# tools/xbox

Experimental tooling for running Xbox Series shader code through the shader recompiler. See `docs/xbox/XBOX_SHADERS.md`.

1. `build_msvc.py` builds the recompiler and the tools below with MSVC (run it from PowerShell/cmd). On Linux,
   `cmake -S tools/xbox -B build-xbox -DCMAKE_BUILD_TYPE=Release && cmake --build build-xbox` builds `driver`, `stages` and `exec`
   (needs the `SPIRV-Headers` and `Vulkan-Headers` submodules; `exec` opens the Vulkan loader at run time, as the AGC driver does).
   When a device address does not fit in the 48 bits of a guest pointer (RADV hands out such addresses), `exec` gives the regions remapped
   guest addresses as `lowguest` does. Without `rdna2.py`, `rdna_asm.py` takes its opcode
   tables from LLVM (`rdna2_llvm.py`, needs `llvm-mc` with the AMDGPU target), so `exec_tests.py` and `exec_bvh_tests.py` run with no
   game files: `XBOX_EXEC=build-xbox/exec python3 tools/xbox/exec_tests.py`. `VK_DRIVER_FILES=<icd.json>` picks the GPU.
2. `extract_shaders.py` writes prologue-stripped, trimmed code blobs `<idx>.bin` (it depends on the RPF9
   extraction pipeline and `rdna2.py` decoder kept next to the dumped shaders; adjust its paths).
3. `driver.exe <code.bin> <wave 32|64> <out.spv> [user_data_dwords]` recompiles one compute shader with stubbed
   bindings (user-data pointer pairs backed by fake buffer-descriptor memory).
4. `run_batch.py` runs the driver over a directory of blobs and validates with `spirv-val`.
5. `stages.exe <code.bin>` times decode / CFG build / structurize separately (for the slow ray-tracing shaders).
6. `exec.exe <scenario.txt>` is the execution harness: it recompiles a compute shader against REAL Vulkan device memory, dispatches it on the
   host GPU and writes the memory regions (and images) back. Build it like `driver.exe` plus the Vulkan SDK include path and `vulkan-1.lib`.
   `rdna_asm.py` assembles small RDNA2 test programs (opcode numbers come from the decoder tables; each program is decoded back to check it),
   `exec_tests.py` and `exec_bvh_tests.py` hold the tests. Set `XBOX_EXEC` to the `exec.exe` path. Supported: guest buffers, shader data,
   flattened SRT, identity BDA page table and fault reporting, 2D RGBA32F images, samplers. Not yet: other image formats/dimensions, image
   tables, texel buffers, graphics stages.

