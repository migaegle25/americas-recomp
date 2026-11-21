// XBOX-FORK: execution harness. Recompiles one RDNA2 compute shader against REAL device memory, dispatches it through Vulkan and
// writes the memory regions back, so recompiled code can be checked against an independent expectation (see tools/xbox/exec_tests.py).
//
// usage: exec <scenario.txt>
// scenario lines (all numbers decimal or 0x hex):
//   wave <32|64>                      groups <x> <y> <z>                  code <file.bin>        out <dir>
//   region <id> <bytes> [init <file>]   (a device-address buffer; its address is real)
//   poke <id> <byteOffset> <u32>        ptrlo <id> <byteOffset> <targetId> [addBytes]
//   ptrhi <id> <byteOffset> <targetId> <orMask>     ((address >> 32) & 0xffff) | orMask
//   image <id> <width> <height> [clear r g b a] [init <file>]    (an R32G32B32A32_SFLOAT 2D image; it gets a fake 256-aligned guest address)
//   timg <regionId> <byteOffset> <imageId>          (writes an 8-dword 2D RGBA32F T# for the image into a region)
//   user <sgpr> <targetId>              (user data pair: low, high dword of the region address)
//   userval <sgpr> <u32>
//   user32 <sgpr> <targetId>   (a single 32-bit pointer register: the guest address of the region)
//   lowguest   (give every region a guest address below 4 GiB; the BDA page table maps guest to device addresses)
//   stage ps   (a pixel shader instead of a compute shader: draws one triangle that covers the whole target; the vertex shader
//               comes from `vs <spv>` and writes constant interpolants; colour outputs are read back as out/target<i>.bin, RGBA32F)
//   export <target> <mode> <mapping>   (SPI_SHADER_COL_FORMAT-style output mode, default 4 = FP16_ABGR; mapping default 0xe4 = identity)
//   depth   (a D32_SFLOAT depth attachment, test ALWAYS + write; read back as out/depth.bin, cleared to 0.123456)
//   target <w> <h>    inputaddr <hex>    interp <count> <fp16 attribute mask hex>
//   threads <x> <y> <z>   (workgroup size, default <wave> 1 1)     lds <dwords>     userdwords <n>   (user SGPR count; the workgroup ids follow)
// Supported bindings: guest buffers (V# bases must be region starts), shader data, flattened SRT, GDS/page-table/fault stubs.
// Images and samplers are not implemented yet and make the run fail with a clear message.
#include "Recompiler.hpp"
#include "BdaAbi.hpp"
#include "vulkan_loader.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ShaderRecompiler;

namespace {

#define VK(call) do { const VkResult r_ = (call); if (r_ != VK_SUCCESS) throw std::runtime_error(std::string(#call) + " failed: " + std::to_string(r_)); } while (0)

struct Region {
    std::uint32_t id = 0;
    std::uint64_t size = 0;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    std::uint64_t address = 0;   // device address
    std::uint64_t guest = 0;     // address the guest code sees (equal to the device address unless `lowguest` is set)
};

struct ImageRes {
    std::uint32_t id = 0;
    std::uint32_t width = 0, height = 0;
    std::uint64_t fakeAddress = 0;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    std::uint32_t layers = 1;   // > 1 makes a 2D array image
    float clear[4] = {0, 0, 0, 0};
    std::string initFile;   // raw R32G32B32A32 float texels, row-major; replaces the clear value
};

struct Device {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    std::uint32_t family = 0;
    VkPhysicalDeviceMemoryProperties memory{};
    PFN_vkGetBufferDeviceAddress getAddress = nullptr;
};

std::uint32_t FindMemory(const Device& d, std::uint32_t bits, VkMemoryPropertyFlags flags) {
    for (std::uint32_t i = 0; i < d.memory.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) && (d.memory.memoryTypes[i].propertyFlags & flags) == flags) return i;
    }
    throw std::runtime_error("no suitable memory type");
}

void MakeBuffer(Device& d, Region& r, std::uint64_t size, VkBufferUsageFlags usage) {
    r.size = size;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size;
    bi.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VK(vkCreateBuffer(d.device, &bi, nullptr, &r.buffer));
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(d.device, r.buffer, &req);
    VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &flags};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = FindMemory(d, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK(vkAllocateMemory(d.device, &ai, nullptr, &r.memory));
    VK(vkBindBufferMemory(d.device, r.buffer, r.memory, 0));
    VK(vkMapMemory(d.device, r.memory, 0, VK_WHOLE_SIZE, 0, &r.mapped));
    std::memset(r.mapped, 0, size);
    VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    info.buffer = r.buffer;
    r.address = d.getAddress(d.device, &info);
}

Device CreateDevice(std::uint32_t wave, bool bary) {
    Device d;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    LoadVulkanLoader();
    VK(vkCreateInstance(&ici, nullptr, &d.instance));
    LoadVulkanInstance(d.instance);
    std::uint32_t count = 0;
    vkEnumeratePhysicalDevices(d.instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(d.instance, &count, devices.data());
    for (auto candidate : devices) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(candidate, &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU || d.physical == VK_NULL_HANDLE) d.physical = candidate;
    }
    if (d.physical == VK_NULL_HANDLE) throw std::runtime_error("no Vulkan device");
    vkGetPhysicalDeviceMemoryProperties(d.physical, &d.memory);
    std::uint32_t qc = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(d.physical, &qc, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qc);
    vkGetPhysicalDeviceQueueFamilyProperties(d.physical, &qc, qf.data());
    for (std::uint32_t i = 0; i < qc; ++i) if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { d.family = i; break; }
    // Enable every core 1.1 - 1.3 feature the device has (the recompiler asks for many of them).
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &f12};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f11};
    VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR fbary{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR};
    if (bary) f13.pNext = &fbary;
    vkGetPhysicalDeviceFeatures2(d.physical, &f2);
    if (bary && !fbary.fragmentShaderBarycentric) throw std::runtime_error("the device has no VK_KHR_fragment_shader_barycentric");
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qi.queueFamilyIndex = d.family;
    qi.queueCount = 1;
    qi.pQueuePriorities = &priority;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2};
    const char* baryExt = VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME;
    if (bary) { dci.enabledExtensionCount = 1; dci.ppEnabledExtensionNames = &baryExt; }
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qi;
    VK(vkCreateDevice(d.physical, &dci, nullptr, &d.device));
    vkGetDeviceQueue(d.device, d.family, 0, &d.queue);
    d.getAddress = reinterpret_cast<PFN_vkGetBufferDeviceAddress>(vkGetDeviceProcAddr(d.device, "vkGetBufferDeviceAddress"));
    (void)wave;
    return d;
}

std::vector<char> ReadFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + path);
    return std::vector<char>((std::istreambuf_iterator<char>(f)), {});
}

std::uint64_t Num(const std::string& s) { return std::stoull(s, nullptr, 0); }

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: exec <scenario.txt>\n"); return 2; }
    try {
        std::uint32_t wave = 32;
        std::uint32_t groups[3] = {1, 1, 1};
        bool lowGuest = false, pixel = false, hasDepth = false;
        std::uint32_t targetW = 8, targetH = 8, inputAddr = 2, interpCount = 0, fp16Mask = 0;
        std::string vsPath;
        std::uint8_t expMode[8] = {4, 4, 4, 4, 4, 4, 4, 4}, expMap[8] = {0xe4, 0xe4, 0xe4, 0xe4, 0xe4, 0xe4, 0xe4, 0xe4};
        std::uint32_t threads[3] = {0, 1, 1};
        std::uint32_t ldsDwords = 0, userCount = 16;
        std::string codePath, outDir = ".";
        struct Line { std::vector<std::string> w; };
        std::vector<Line> lines;
        {
            std::ifstream sf(argv[1]);
            std::string text;
            while (std::getline(sf, text)) {
                std::istringstream ss(text);
                Line l; std::string tok;
                while (ss >> tok) l.w.push_back(tok);
                if (!l.w.empty() && l.w[0][0] != '#') lines.push_back(std::move(l));
            }
        }
        for (auto& l : lines) {
            if (l.w[0] == "wave") wave = static_cast<std::uint32_t>(Num(l.w[1]));
            else if (l.w[0] == "groups") for (int i = 0; i < 3; ++i) groups[i] = static_cast<std::uint32_t>(Num(l.w[1 + i]));
            else if (l.w[0] == "code") codePath = l.w[1];
            else if (l.w[0] == "lowguest") lowGuest = true;
            else if (l.w[0] == "stage") pixel = l.w[1] == "ps";
            else if (l.w[0] == "target") { targetW = static_cast<std::uint32_t>(Num(l.w[1])); targetH = static_cast<std::uint32_t>(Num(l.w[2])); }
            else if (l.w[0] == "inputaddr") inputAddr = static_cast<std::uint32_t>(std::stoul(l.w[1], nullptr, 16));
            else if (l.w[0] == "interp") { interpCount = static_cast<std::uint32_t>(Num(l.w[1])); fp16Mask = static_cast<std::uint32_t>(std::stoul(l.w[2], nullptr, 16)); }
            else if (l.w[0] == "vs") vsPath = l.w[1];
            else if (l.w[0] == "depth") hasDepth = true;
            else if (l.w[0] == "export") { const auto t = Num(l.w[1]) & 7u; expMode[t] = static_cast<std::uint8_t>(Num(l.w[2])); expMap[t] = static_cast<std::uint8_t>(Num(l.w[3])); }
            else if (l.w[0] == "threads") for (int i = 0; i < 3; ++i) threads[i] = static_cast<std::uint32_t>(Num(l.w[1 + i]));
            else if (l.w[0] == "lds") ldsDwords = static_cast<std::uint32_t>(Num(l.w[1]));
            else if (l.w[0] == "userdwords") userCount = static_cast<std::uint32_t>(Num(l.w[1]));
            else if (l.w[0] == "out") outDir = l.w[1];
        }
        Device dev = CreateDevice(wave, pixel && !std::getenv("XBOX_NO_FRAGMENT_BARY"));
        std::map<std::uint32_t, Region> regions;
        for (auto& l : lines) {
            if (l.w[0] != "region") continue;
            Region& r = regions[static_cast<std::uint32_t>(Num(l.w[1]))];
            r.id = static_cast<std::uint32_t>(Num(l.w[1]));
            MakeBuffer(dev, r, Num(l.w[2]), 0);
            if (l.w.size() >= 5 && l.w[3] == "init") {
                const auto data = ReadFile(l.w[4]);
                std::memcpy(r.mapped, data.data(), std::min<std::size_t>(data.size(), r.size));
            }
        }
        {
            std::uint64_t next = 0x10000000ull;
            const bool remap = lowGuest || std::any_of(regions.begin(), regions.end(), [](const auto& e) { return (e.second.address >> 48) != 0; });
            for (auto& [rid, r] : regions) {
                r.guest = remap ? next : r.address;
                next += 0x1000000ull;
            }
        }
        std::map<std::uint32_t, ImageRes> images;
        lines.insert(lines.begin(), Line{{"image", "9999", "16", "16"}});   // dummy image for unpopulated descriptor array elements
        for (auto& l : lines) {
            if (l.w[0] != "image") continue;
            ImageRes& im = images[static_cast<std::uint32_t>(Num(l.w[1]))];
            im.id = static_cast<std::uint32_t>(Num(l.w[1]));
            im.width = static_cast<std::uint32_t>(Num(l.w[2])); im.height = static_cast<std::uint32_t>(Num(l.w[3]));
            im.fakeAddress = 0x7000000000ull + 0x1000000ull * im.id;
            if (l.w.size() >= 9 && l.w[4] == "clear") for (int c = 0; c < 4; ++c) im.clear[c] = std::stof(l.w[5 + c]);
            for (std::size_t k = 4; k + 1 < l.w.size(); ++k) if (l.w[k] == "init") im.initFile = l.w[k + 1];
            for (std::size_t k = 4; k + 1 < l.w.size(); ++k) if (l.w[k] == "layers") im.layers = static_cast<std::uint32_t>(Num(l.w[k + 1]));
            VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            ici.imageType = VK_IMAGE_TYPE_2D; ici.format = VK_FORMAT_R32G32B32A32_SFLOAT;
            ici.extent = {im.width, im.height, 1}; ici.mipLevels = 1; ici.arrayLayers = im.layers; ici.samples = VK_SAMPLE_COUNT_1_BIT;
            ici.tiling = VK_IMAGE_TILING_OPTIMAL;
            ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            VK(vkCreateImage(dev.device, &ici, nullptr, &im.image));
            VkMemoryRequirements req{};
            vkGetImageMemoryRequirements(dev.device, im.image, &req);
            VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            ai.allocationSize = req.size; ai.memoryTypeIndex = FindMemory(dev, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            VK(vkAllocateMemory(dev.device, &ai, nullptr, &im.memory));
            VK(vkBindImageMemory(dev.device, im.image, im.memory, 0));
            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vi.image = im.image; vi.viewType = im.layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D; vi.format = ici.format;
            vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, im.layers};
            VK(vkCreateImageView(dev.device, &vi, nullptr, &im.view));
        }
        const auto region = [&](const std::string& id) -> Region& {
            const auto it = regions.find(static_cast<std::uint32_t>(Num(id)));
            if (it == regions.end()) throw std::runtime_error("unknown region " + id);
            return it->second;
        };
        const auto dword = [&](Region& r, std::uint64_t off) -> std::uint32_t& {
            if (off + 4 > r.size) throw std::runtime_error("patch outside region");
            return *reinterpret_cast<std::uint32_t*>(static_cast<std::byte*>(r.mapped) + off);
        };
        if (threads[0] == 0) threads[0] = wave;
        std::vector<std::uint32_t> userData(userCount, 0);
        for (auto& l : lines) {
            if (l.w[0] == "timg") {
                const auto it = images.find(static_cast<std::uint32_t>(Num(l.w[3])));
                if (it == images.end()) throw std::runtime_error("unknown image " + l.w[3]);
                const ImageRes& im = it->second;
                Region& r = region(l.w[1]);
                const auto off = Num(l.w[2]);
                // word0 base>>8; word1 base_hi | format 77 (32_32_32_32_FLOAT; 56 would be 8_8_8_8_UNORM) << 20 | 3 << 30; word2 (w-1) | (h-1) << 14; word3 identity swizzle | type 2D (9) << 28
                dword(r, off) = static_cast<std::uint32_t>(im.fakeAddress >> 8);
                dword(r, off + 4) = static_cast<std::uint32_t>((im.fakeAddress >> 40) & 0xffu) | (77u << 20) | (3u << 30);
                dword(r, off + 8) = (im.width - 1) | ((im.height - 1) << 14);
                dword(r, off + 12) = 0xfacu | ((im.layers > 1 ? 13u : 9u) << 28);   // type 9 = 2D, 13 = 2D array
                for (int k = 4; k < 8; ++k) dword(r, off + 4 * k) = 0;
                if (im.layers > 1) dword(r, off + 16) = im.layers - 1;   // depth field: last array slice
            }
            else if (l.w[0] == "poke") dword(region(l.w[1]), Num(l.w[2])) = static_cast<std::uint32_t>(Num(l.w[3]));
            else if (l.w[0] == "ptrlo") dword(region(l.w[1]), Num(l.w[2])) = static_cast<std::uint32_t>(region(l.w[3]).guest + (l.w.size() > 4 ? Num(l.w[4]) : 0));
            else if (l.w[0] == "ptr8lo") dword(region(l.w[1]), Num(l.w[2])) = static_cast<std::uint32_t>(region(l.w[3]).guest >> 8);
            else if (l.w[0] == "ptr8hi") dword(region(l.w[1]), Num(l.w[2])) = static_cast<std::uint32_t>(((region(l.w[3]).guest >> 40) & 0xffu) | Num(l.w[4]));
            else if (l.w[0] == "ptrhi") dword(region(l.w[1]), Num(l.w[2])) = static_cast<std::uint32_t>(((region(l.w[3]).guest >> 32) & 0xffffu) | Num(l.w[4]));
            else if (l.w[0] == "user") { const auto k = Num(l.w[1]); userData.at(k) = static_cast<std::uint32_t>(region(l.w[2]).guest); userData.at(k + 1) = static_cast<std::uint32_t>(region(l.w[2]).guest >> 32); }
            else if (l.w[0] == "user32") userData.at(Num(l.w[1])) = static_cast<std::uint32_t>(region(l.w[2]).guest);
            else if (l.w[0] == "userval") userData.at(Num(l.w[1])) = static_cast<std::uint32_t>(Num(l.w[2]));
        }

        for (auto& [id, r] : regions) std::printf("region %u guest 0x%llx device 0x%llx size %llu\n", id, static_cast<unsigned long long>(r.guest), static_cast<unsigned long long>(r.address), static_cast<unsigned long long>(r.size));
        // Compile against a snapshot of the real contents at the real addresses.
        const auto raw = ReadFile(codePath);
        std::vector<std::uint32_t> code(raw.size() / 4);
        std::memcpy(code.data(), raw.data(), code.size() * 4);
        std::vector<std::vector<std::byte>> snapshots;
        std::vector<MemoryRegion> memory;
        for (auto& [id, r] : regions) {
            snapshots.emplace_back(static_cast<std::byte*>(r.mapped), static_cast<std::byte*>(r.mapped) + r.size);
            memory.push_back({r.guest, std::span<const std::byte>(snapshots.back())});
        }
        RecompileRequest request{};
        request.context.memory = MemoryRegionView(std::span<const MemoryRegion>(memory));
        const std::vector<std::uint32_t> caps{1u,55u,56u,32u,33u,5283u,9u,10u,11u,22u,39u,61u,62u,63u,64u,65u,66u,67u,68u,4427u,4433u,4434u,4435u,4436u,4437u,4438u,4445u,4447u,4448u,4449u,4450u,5345u,5346u,5347u,5301u,5302u,5303u,5304u,5305u,6u,38u,41u,5u,17u,18u,19u,3u,2u,4u,7u,8u,45u,5249u};
        const std::array<std::string_view, 7> ext{"SPV_EXT_mesh_shader", "SPV_KHR_storage_buffer_storage_class", "SPV_KHR_physical_storage_buffer", "SPV_KHR_16bit_storage", "SPV_KHR_8bit_storage", "SPV_KHR_vulkan_memory_model", "SPV_KHR_shader_ballot"};
        request.shader = {pixel ? ShaderStage::Fragment : ShaderStage::Compute, 0x20000u, code};
        if (pixel) {
            ShaderPixelStageInfo px{};
            px.interpolatorCount = interpCount;
            px.wave32 = wave == 32;
            px.inputAddr = inputAddr;
            px.depthExportEnable = hasDepth;   // SPI_SHADER_Z_FORMAT enables the MRTZ export; without it the recompiler drops it
            px.hasPerspectiveCenterVgpr = (inputAddr & 0x2u) != 0;
            px.posX = (inputAddr & PixelInputBit(PixelInput::PositionX)) != 0;
            px.posY = (inputAddr & PixelInputBit(PixelInput::PositionY)) != 0;
            px.posZ = (inputAddr & PixelInputBit(PixelInput::PositionZ)) != 0;
            px.posW = (inputAddr & PixelInputBit(PixelInput::PositionW)) != 0;
            px.frontFace = (inputAddr & PixelInputBit(PixelInput::FrontFace)) != 0;
            for (std::uint32_t i = 0; i < 8; i++) { px.targetExportMapping[i] = expMap[i]; px.targetOutputMode[i] = expMode[i]; px.targetExportPacking[i] = ColorExportPacking::None; }
            for (std::uint32_t i = 0; i < interpCount && i < 32; i++) px.interpolatorSettings[i] = i | (((fp16Mask >> i) & 1u) ? 0x80000u : 0u);
            request.context.pixel = px;
            request.context.floatMode = ShaderFloatMode{0xc0u, true, false, false};
        }
        request.context.waveSize = wave;
        request.context.userDataBaseRegister = 0;
        request.context.userData = userData;
        if (!pixel) request.context.compute = ShaderComputeStageInfo{{threads[0], threads[1], threads[2]}, ldsDwords, {true, true, true}, false, 1u};
        request.target.vulkanVersion = 0x00403000u;
        request.target.spirvVersion = 0x00010600u;
        request.target.subgroupSize = wave;
        request.target.bdaAbiVersion = 1;
        std::vector<std::uint32_t> capsLocal = caps;
        std::vector<std::string_view> extLocal(ext.begin(), ext.end());
        if (pixel && !std::getenv("XBOX_NO_FRAGMENT_BARY")) {
            // the host has VK_KHR_fragment_shader_barycentric (per-vertex inputs and barycentric builtins need no geometry-shader emulation)
            request.target.fragmentShaderBarycentricEnabled = true;
            request.target.fragmentBarycentricBuiltins = true;
            capsLocal.push_back(5284u);
            extLocal.push_back("SPV_KHR_fragment_shader_barycentric");
        }
        request.target.supportedCapabilities = capsLocal;
        request.target.supportedExtensions = extLocal;
        request.target.maxWorkgroupSize = {1024u, 1024u, 64u};
        request.target.maxWorkgroupInvocations = 1024;
        request.target.maxWorkgroupSharedMemoryBytes = 65536;
        request.layout = {0, 0, 0, 128};
        request.useCache = false;
        const auto result = Recompile(request);
        if (result.spirv.empty()) throw std::runtime_error("recompile produced no module");
        std::printf("compiled: %zu words, %zu bindings, push %zu bytes, bdaAbi %u\n", result.spirv.Words().size(), result.bindings.size(), result.pushConstants.size(), result.bdaAbiVersion);

        const VkShaderStageFlags stageBits = pixel ? VK_SHADER_STAGE_FRAGMENT_BIT : VK_SHADER_STAGE_COMPUTE_BIT;
        // Descriptor layout + resources for each binding.
        std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
        std::vector<std::vector<VkDescriptorBufferInfo>> infos(result.bindings.size());
        std::vector<std::vector<VkDescriptorImageInfo>> imageInfos(result.bindings.size());
        std::vector<Region> extra;
        extra.reserve(result.bindings.size() * 4 + 8);
        Region* faultBuffer = nullptr;
        const auto regionAt = [&](std::uint64_t addr) -> Region* {
            for (auto& [id, r] : regions) if (addr >= r.guest && addr < r.guest + r.size) return &r;
            return nullptr;
        };
        for (std::size_t bi = 0; bi < result.bindings.size(); ++bi) {
            const auto& b = result.bindings[bi];
            if (b.descriptorSet != 0) throw std::runtime_error("descriptor sets other than 0 are not implemented");
            if (b.kind == DescriptorKind::SampledImage || b.kind == DescriptorKind::StorageImage) {
                if (b.guestDescriptor.size() < static_cast<std::size_t>(b.count) * 8u) throw std::runtime_error("binding " + std::to_string(b.binding) + ": image descriptor has " + std::to_string(b.guestDescriptor.size()) + " dwords for " + std::to_string(b.count) + " elements");
            }
            if (b.role == DescriptorRole::GuestBuffers && b.guestDescriptor.size() < static_cast<std::size_t>(b.count) * 4u) throw std::runtime_error("binding " + std::to_string(b.binding) + ": buffer descriptor has " + std::to_string(b.guestDescriptor.size()) + " dwords for " + std::to_string(b.count) + " elements");
            if (b.kind == DescriptorKind::SampledImage || b.kind == DescriptorKind::StorageImage || b.kind == DescriptorKind::Sampler) {
                const VkDescriptorType type = b.kind == DescriptorKind::SampledImage ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE : b.kind == DescriptorKind::StorageImage ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLER;
                layoutBindings.push_back({b.binding, type, b.count, stageBits, nullptr});
                for (std::uint32_t e = 0; e < b.count; ++e) {
                    VkDescriptorImageInfo ii{};
                    ii.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    if (b.kind == DescriptorKind::Sampler) {
                        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
                        sci.magFilter = sci.minFilter = VK_FILTER_NEAREST;
                        sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
                        sci.maxLod = 16.0f;
                        VK(vkCreateSampler(dev.device, &sci, nullptr, &ii.sampler));
                    } else {
                        const std::uint64_t base = (static_cast<std::uint64_t>(b.guestDescriptor[e * 8]) << 8) | (static_cast<std::uint64_t>(b.guestDescriptor[e * 8 + 1] & 0xffu) << 40);
                        ImageRes* found = nullptr;
                        for (auto& [iid, im] : images) if (im.fakeAddress == base) found = &im;
                        if (found == nullptr) found = &images.at(9999);
                        ii.imageView = found->view;
                    }
                    imageInfos[bi].push_back(ii);
                }
                continue;
            }
            if (b.kind != DescriptorKind::StorageBuffer) throw std::runtime_error("binding " + std::to_string(b.binding) + ": texel buffers are not implemented in the harness");
            layoutBindings.push_back({b.binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, b.count, stageBits, nullptr});
            for (std::uint32_t e = 0; e < b.count; ++e) {
                VkDescriptorBufferInfo info{VK_NULL_HANDLE, 0, VK_WHOLE_SIZE};
                if (b.role == DescriptorRole::GuestBuffers) {
                    const std::uint64_t base = b.guestDescriptor[e * 4] | (static_cast<std::uint64_t>(b.guestDescriptor[e * 4 + 1] & 0xffffu) << 32);
                    Region* r = regionAt(base);
                    if (r == nullptr) {
                        // a null or dangling V# (for example an unused slot in a descriptor array): bind a zeroed dummy buffer
                        extra.emplace_back();
                        MakeBuffer(dev, extra.back(), 1u << 16, 0);
                        info.buffer = extra.back().buffer;
                    } else {
                        if (r->guest != base) throw std::runtime_error("V# base is not a region start (unaligned element offsets are not implemented)");
                        info.buffer = r->buffer;
                    }
                } else if (b.role == DescriptorRole::ShaderData || b.role == DescriptorRole::FlattenedSrt) {
                    extra.emplace_back();
                    MakeBuffer(dev, extra.back(), std::max<std::size_t>(b.guestDescriptor.size() * 4, 16), 0);
                    std::memcpy(extra.back().mapped, b.guestDescriptor.data(), b.guestDescriptor.size() * 4);
                    info.buffer = extra.back().buffer;
                } else if (b.role == DescriptorRole::BdaPagetable) {
                    // identity page table: every region maps to its own device address
                    extra.emplace_back();
                    std::vector<BdaAbi::Range> ranges;
                    for (auto& [rid, r] : regions) ranges.push_back({r.guest, r.guest + r.size, r.address, BdaAbi::Read | BdaAbi::Write, 0});
                    std::sort(ranges.begin(), ranges.end(), [](const BdaAbi::Range& a, const BdaAbi::Range& c) { return a.begin < c.begin; });
                    MakeBuffer(dev, extra.back(), sizeof(BdaAbi::Header) + ranges.size() * sizeof(BdaAbi::Range), 0);
                    const BdaAbi::Header header{BdaAbi::Version, static_cast<std::uint32_t>(ranges.size()), static_cast<std::uint32_t>(sizeof(BdaAbi::Range)), 0};
                    std::memcpy(extra.back().mapped, &header, sizeof(header));
                    std::memcpy(static_cast<std::byte*>(extra.back().mapped) + sizeof(header), ranges.data(), ranges.size() * sizeof(BdaAbi::Range));
                    info.buffer = extra.back().buffer;
                } else if (b.role == DescriptorRole::FaultBuffer) {
                    extra.emplace_back();
                    MakeBuffer(dev, extra.back(), BdaAbi::FaultBufferBytes, 0);
                    faultBuffer = &extra.back();
                    info.buffer = extra.back().buffer;
                } else if (b.role == DescriptorRole::Gds) {
                    extra.emplace_back();
                    MakeBuffer(dev, extra.back(), 1u << 16, 0);
                    info.buffer = extra.back().buffer;
                } else {
                    throw std::runtime_error("binding " + std::to_string(b.binding) + ": role " + std::to_string(static_cast<int>(b.role)) + " is not implemented in the harness");
                }
                infos[bi].push_back(info);
            }
        }
        VkDescriptorSetLayoutCreateInfo dsli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dsli.bindingCount = static_cast<std::uint32_t>(layoutBindings.size());
        dsli.pBindings = layoutBindings.data();
        VkDescriptorSetLayout setLayout;
        VK(vkCreateDescriptorSetLayout(dev.device, &dsli, nullptr, &setLayout));
        VkPushConstantRange pcr{stageBits, 0, static_cast<std::uint32_t>(std::max<std::size_t>(result.pushConstants.size(), 128))};
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1; pli.pSetLayouts = &setLayout; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
        VkPipelineLayout pipelineLayout;
        VK(vkCreatePipelineLayout(dev.device, &pli, nullptr, &pipelineLayout));

        const auto words = result.spirv.Words();
        if (const char* dump = std::getenv("XBOX_EXEC_SPV")) {
            std::ofstream o(dump, std::ios::binary);
            o.write(reinterpret_cast<const char*>(words.data()), static_cast<std::streamsize>(words.size() * 4));
        }
        VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smi.codeSize = words.size() * 4; smi.pCode = words.data();
        VkShaderModule module;
        VK(vkCreateShaderModule(dev.device, &smi, nullptr, &module));
        std::vector<VkSpecializationMapEntry> entries;
        std::vector<std::uint32_t> specData;
        for (const auto& c : result.specialization) { entries.push_back({c.id, static_cast<std::uint32_t>(specData.size() * 4), 4}); specData.push_back(c.value); }
        VkSpecializationInfo spec{static_cast<std::uint32_t>(entries.size()), entries.data(), specData.size() * 4, specData.data()};
        VkPipeline pipeline;
        if (!pixel) {
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo sub{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
        sub.requiredSubgroupSize = result.hostSubgroupSize != 0 ? result.hostSubgroupSize : wave;
        VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, &sub, result.fullSubgroups ? VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT : 0u, VK_SHADER_STAGE_COMPUTE_BIT, module, "main", entries.empty() ? nullptr : &spec};
        cpi.layout = pipelineLayout;
        VK(vkCreateComputePipelines(dev.device, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipeline));
        } else {
            if (vsPath.empty()) throw std::runtime_error("stage ps needs a `vs <file.spv>` line");
            const auto vsRaw = ReadFile(vsPath);
            VkShaderModuleCreateInfo vmi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            vmi.codeSize = vsRaw.size(); vmi.pCode = reinterpret_cast<const std::uint32_t*>(vsRaw.data());
            VkShaderModule vsModule;
            VK(vkCreateShaderModule(dev.device, &vmi, nullptr, &vsModule));
            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vsModule, "main", nullptr};
            stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, module, "main", entries.empty() ? nullptr : &spec};
            VkPipelineVertexInputStateCreateInfo vin{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
            ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            VkViewport vp{0, 0, static_cast<float>(targetW), static_cast<float>(targetH), 0, 1};
            VkRect2D sc{{0, 0}, {targetW, targetH}};
            VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
            vps.viewportCount = 1; vps.pViewports = &vp; vps.scissorCount = 1; vps.pScissors = &sc;
            VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.0f;
            VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
            ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
            VkPipelineColorBlendAttachmentState cba{};
            cba.colorWriteMask = 0xf;
            std::array<VkPipelineColorBlendAttachmentState, 8> cbas; cbas.fill(cba);
            VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            cb.attachmentCount = 8; cb.pAttachments = cbas.data();
            std::array<VkFormat, 8> formats; formats.fill(VK_FORMAT_R32G32B32A32_SFLOAT);
            VkPipelineRenderingCreateInfo rci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
            rci.colorAttachmentCount = 8; rci.pColorAttachmentFormats = formats.data();
            VkPipelineDepthStencilStateCreateInfo dss{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
            dss.depthTestEnable = hasDepth ? VK_TRUE : VK_FALSE; dss.depthWriteEnable = hasDepth ? VK_TRUE : VK_FALSE; dss.depthCompareOp = VK_COMPARE_OP_ALWAYS;
            if (hasDepth) rci.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;
            VkGraphicsPipelineCreateInfo gpi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &rci};
            gpi.stageCount = 2; gpi.pStages = stages; gpi.pVertexInputState = &vin; gpi.pInputAssemblyState = &ia; gpi.pViewportState = &vps;
            gpi.pDepthStencilState = &dss; gpi.pRasterizationState = &rs; gpi.pMultisampleState = &ms; gpi.pColorBlendState = &cb; gpi.layout = pipelineLayout;
            VK(vkCreateGraphicsPipelines(dev.device, VK_NULL_HANDLE, 1, &gpi, nullptr, &pipeline));
        }

        const std::array<VkDescriptorPoolSize, 4> ps{{{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4096}, {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 4096}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4096}, {VK_DESCRIPTOR_TYPE_SAMPLER, 4096}}};
        VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpi.maxSets = 1; dpi.poolSizeCount = static_cast<std::uint32_t>(ps.size()); dpi.pPoolSizes = ps.data();
        VkDescriptorPool pool;
        VK(vkCreateDescriptorPool(dev.device, &dpi, nullptr, &pool));
        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsai.descriptorPool = pool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &setLayout;
        VkDescriptorSet set;
        VK(vkAllocateDescriptorSets(dev.device, &dsai, &set));
        std::vector<VkWriteDescriptorSet> writes;
        for (std::size_t bi = 0; bi < result.bindings.size(); ++bi) {
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = set; w.dstBinding = result.bindings[bi].binding; w.descriptorCount = result.bindings[bi].count;
            for (const auto& lb : layoutBindings) if (lb.binding == w.dstBinding) w.descriptorType = lb.descriptorType;
            if (w.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) w.pBufferInfo = infos[bi].data(); else w.pImageInfo = imageInfos[bi].data();
            writes.push_back(w);
        }
        vkUpdateDescriptorSets(dev.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);

        VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpci.queueFamilyIndex = dev.family;
        VkCommandPool cmdPool;
        VK(vkCreateCommandPool(dev.device, &cpci, nullptr, &cmdPool));
        VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbai.commandPool = cmdPool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
        VkCommandBuffer cmd;
        VK(vkAllocateCommandBuffers(dev.device, &cbai, &cmd));
        VkCommandBufferBeginInfo bbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK(vkBeginCommandBuffer(cmd, &bbi));
        std::vector<Region> staging;
        staging.reserve(images.size());
        for (auto& [iid, im] : images) {
            VkImageMemoryBarrier toTransfer{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; toTransfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            toTransfer.srcQueueFamilyIndex = toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; toTransfer.image = im.image; toTransfer.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, im.layers};
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toTransfer);
            if (!im.initFile.empty()) {
                std::ifstream in(im.initFile, std::ios::binary);
                if (!in) throw std::runtime_error("cannot open image init file " + im.initFile);
                staging.emplace_back();
                MakeBuffer(dev, staging.back(), static_cast<std::uint64_t>(im.width) * im.height * 16 * im.layers, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
                in.read(static_cast<char*>(staging.back().mapped), static_cast<std::streamsize>(staging.back().size));
                VkBufferImageCopy up{};
                up.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, im.layers}; up.imageExtent = {im.width, im.height, 1};
                vkCmdCopyBufferToImage(cmd, staging.back().buffer, im.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &up);
            } else {
                VkClearColorValue color{}; std::memcpy(color.float32, im.clear, sizeof(im.clear));
                VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, im.layers};
                vkCmdClearColorImage(cmd, im.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &range);
            }
            VkImageMemoryBarrier toGeneral = toTransfer;
            toGeneral.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; toGeneral.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            toGeneral.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, pixel ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toGeneral);
        }
        const VkPipelineBindPoint bindPoint = pixel ? VK_PIPELINE_BIND_POINT_GRAPHICS : VK_PIPELINE_BIND_POINT_COMPUTE;
        vkCmdBindPipeline(cmd, bindPoint, pipeline);
        vkCmdBindDescriptorSets(cmd, bindPoint, pipelineLayout, 0, 1, &set, 0, nullptr);
        if (!result.pushConstants.empty()) vkCmdPushConstants(cmd, pipelineLayout, stageBits, 0, static_cast<std::uint32_t>(result.pushConstants.size()), result.pushConstants.data());
        struct Target { VkImage image = VK_NULL_HANDLE; VkDeviceMemory memory = VK_NULL_HANDLE; VkImageView view = VK_NULL_HANDLE; Region readback; };
        std::vector<Target> targets;
        Target depthTarget;
        if (!pixel) {
            vkCmdDispatch(cmd, groups[0], groups[1], groups[2]);
        } else {
            targets.resize(8);
            std::array<VkRenderingAttachmentInfo, 8> atts{};
            for (std::uint32_t t = 0; t < 8; ++t) {
                Target& tg = targets[t];
                VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
                ici.imageType = VK_IMAGE_TYPE_2D; ici.format = VK_FORMAT_R32G32B32A32_SFLOAT; ici.extent = {targetW, targetH, 1};
                ici.mipLevels = 1; ici.arrayLayers = 1; ici.samples = VK_SAMPLE_COUNT_1_BIT; ici.tiling = VK_IMAGE_TILING_OPTIMAL;
                ici.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT; ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                VK(vkCreateImage(dev.device, &ici, nullptr, &tg.image));
                VkMemoryRequirements req{};
                vkGetImageMemoryRequirements(dev.device, tg.image, &req);
                VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                ai.allocationSize = req.size; ai.memoryTypeIndex = FindMemory(dev, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                VK(vkAllocateMemory(dev.device, &ai, nullptr, &tg.memory));
                VK(vkBindImageMemory(dev.device, tg.image, tg.memory, 0));
                VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
                vi.image = tg.image; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = ici.format; vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                VK(vkCreateImageView(dev.device, &vi, nullptr, &tg.view));
                VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                b.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; b.image = tg.image; b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
                atts[t] = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
                atts[t].imageView = tg.view; atts[t].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                atts[t].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; atts[t].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                atts[t].clearValue.color.float32[0] = atts[t].clearValue.color.float32[1] = atts[t].clearValue.color.float32[2] = atts[t].clearValue.color.float32[3] = -777.0f;
            }
            VkRenderingAttachmentInfo depthAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            if (hasDepth) {
                VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
                ici.imageType = VK_IMAGE_TYPE_2D; ici.format = VK_FORMAT_D32_SFLOAT; ici.extent = {targetW, targetH, 1};
                ici.mipLevels = 1; ici.arrayLayers = 1; ici.samples = VK_SAMPLE_COUNT_1_BIT; ici.tiling = VK_IMAGE_TILING_OPTIMAL;
                ici.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT; ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                VK(vkCreateImage(dev.device, &ici, nullptr, &depthTarget.image));
                VkMemoryRequirements req{};
                vkGetImageMemoryRequirements(dev.device, depthTarget.image, &req);
                VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                ai.allocationSize = req.size; ai.memoryTypeIndex = FindMemory(dev, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                VK(vkAllocateMemory(dev.device, &ai, nullptr, &depthTarget.memory));
                VK(vkBindImageMemory(dev.device, depthTarget.image, depthTarget.memory, 0));
                VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
                vi.image = depthTarget.image; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = ici.format; vi.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
                VK(vkCreateImageView(dev.device, &vi, nullptr, &depthTarget.view));
                VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                b.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT; b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
                b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; b.image = depthTarget.image; b.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
                depthAtt.imageView = depthTarget.view; depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
                depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE; depthAtt.clearValue.depthStencil = {0.123456f, 0};
            }
            VkRenderingInfo rinfo{VK_STRUCTURE_TYPE_RENDERING_INFO};
            if (hasDepth) rinfo.pDepthAttachment = &depthAtt;
            rinfo.renderArea = {{0, 0}, {targetW, targetH}}; rinfo.layerCount = 1; rinfo.colorAttachmentCount = 8; rinfo.pColorAttachments = atts.data();
            vkCmdBeginRendering(cmd, &rinfo);
            vkCmdDraw(cmd, 3, 1, 0, 0);
            vkCmdEndRendering(cmd);
            if (hasDepth) {
                VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                b.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT; b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                b.oldLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL; b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; b.image = depthTarget.image; b.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
                MakeBuffer(dev, depthTarget.readback, static_cast<std::uint64_t>(targetW) * targetH * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
                VkBufferImageCopy copy{};
                copy.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1}; copy.imageExtent = {targetW, targetH, 1};
                vkCmdCopyImageToBuffer(cmd, depthTarget.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, depthTarget.readback.buffer, 1, &copy);
            }
            for (auto& tg : targets) {
                VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                b.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                b.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; b.image = tg.image; b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
                MakeBuffer(dev, tg.readback, static_cast<std::uint64_t>(targetW) * targetH * 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
                VkBufferImageCopy copy{};
                copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; copy.imageExtent = {targetW, targetH, 1};
                vkCmdCopyImageToBuffer(cmd, tg.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, tg.readback.buffer, 1, &copy);
            }
        }
        std::map<std::uint32_t, Region> readback;
        for (auto& [iid, im] : images) {
            Region& rb = readback[iid];
            MakeBuffer(dev, rb, static_cast<std::uint64_t>(im.width) * im.height * 16 * im.layers, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT};
            vkCmdPipelineBarrier(cmd, pixel ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
            VkBufferImageCopy copy{};
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, im.layers}; copy.imageExtent = {im.width, im.height, 1};
            vkCmdCopyImageToBuffer(cmd, im.image, VK_IMAGE_LAYOUT_GENERAL, rb.buffer, 1, &copy);
        }
        VK(vkEndCommandBuffer(cmd));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
        VK(vkQueueSubmit(dev.queue, 1, &si, VK_NULL_HANDLE));
        VK(vkQueueWaitIdle(dev.queue));

        std::filesystem::create_directories(outDir);
        for (auto& [id, r] : regions) {
            std::ofstream o(outDir + "/region" + std::to_string(id) + ".bin", std::ios::binary);
            o.write(static_cast<const char*>(r.mapped), static_cast<std::streamsize>(r.size));
        }
        for (auto& [iid, rb] : readback) {
            std::ofstream o(outDir + "/image" + std::to_string(iid) + ".bin", std::ios::binary);
            o.write(static_cast<const char*>(rb.mapped), static_cast<std::streamsize>(rb.size));
        }
        for (std::size_t t = 0; t < targets.size(); ++t) {
            std::ofstream o(outDir + "/target" + std::to_string(t) + ".bin", std::ios::binary);
            o.write(static_cast<const char*>(targets[t].readback.mapped), static_cast<std::streamsize>(targets[t].readback.size));
        }
        if (hasDepth && depthTarget.readback.mapped != nullptr) {
            std::ofstream o(outDir + "/depth.bin", std::ios::binary);
            o.write(static_cast<const char*>(depthTarget.readback.mapped), static_cast<std::streamsize>(depthTarget.readback.size));
        }
        if (faultBuffer != nullptr) {
            const auto* fault = static_cast<const BdaAbi::Fault*>(faultBuffer->mapped);
            if (fault->state != BdaAbi::FaultState::Empty) {
                std::printf("FAULT reason=%u address=0x%llx bytes=%u instruction=0x%x\n", static_cast<unsigned>(fault->reason), static_cast<unsigned long long>(fault->address), fault->bytes, fault->instruction);
                return 3;
            }
        }
        std::printf("RAN\n");
        return 0;
    } catch (const std::exception& e) {
        std::string m = e.what();
        for (auto& c : m) if (c == '\n' || c == '\r') c = ' ';
        std::printf("FAIL %s\n", m.c_str());
        return 1;
    }
}
