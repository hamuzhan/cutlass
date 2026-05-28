# AGENTS.md

## Repo Map
- CUTLASS C++ is header-only for consumers; primary headers live in `include/cutlass/` and `include/cute/`.
- CMake-built surfaces are examples in `examples/`, GoogleTest unit tests in `test/unit/`, tools in `tools/`, the kernel library in `tools/library`, and the profiler in `tools/profiler`.
- Python surfaces are separate: legacy interface `python/cutlass_cppgen`, generator `python/cutlass_library`, PyCuTe `python/pycute`, and CuTe DSL package `python/CuTeDSL/cutlass`.
- CuTe DSL examples and tests are outside the package: examples are under `examples/python/CuTeDSL`, pytest tests under `test/examples/CuTeDSL`.

## CMake/CUDA
- Prefer out-of-tree builds: `cmake -S . -B build -DCUTLASS_NVCC_ARCHS=80`; set `CUDACXX=${CUDA_INSTALL_PATH}/bin/nvcc` when the intended `nvcc` is not first on `PATH`.
- Always pick `CUTLASS_NVCC_ARCHS` for focused work; the default expands to every architecture supported by the detected CUDA toolkit and is slow.
- Hopper/Blackwell accelerated kernels need suffixed archs such as `90a`, `100a`, `100f`, or `120a`; `90`/`100` are not substitutes. SM100 datacenter and SM120 GeForce targets are distinct.
- The default configure enables examples, tools, library/profiler, and tests. For faster non-profiler work use `-DCUTLASS_ENABLE_LIBRARY=OFF -DCUTLASS_ENABLE_PROFILER=OFF`.
- `tools/library` runs `python/cutlass_library/generator.py` during configure and writes generated manifests/kernel lists in the build tree; do not edit generated build outputs.
- Avoid `-DCUTLASS_LIBRARY_KERNELS=all` unless explicitly needed; it instantiates tens of thousands of kernels. Filter with `CUTLASS_LIBRARY_KERNELS=<glob>` for profiler/library work.

## C++ Tests And Tools
- Build all C++ unit tests from an existing build: `cmake --build build --target test_unit -j`.
- Build a focused unit-test target by dropping the leading `cutlass_` from the CMake executable name, e.g. `cutlass_test_unit_core` becomes `cmake --build build --target test_unit_core -j`.
- Build/run an example's CMake test target as `cmake --build build --target test_examples_00_basic_gemm -j`; the raw executable target is `00_basic_gemm`.
- Run CTest filters from the build dir with `ctest --test-dir build -R <regex>`; generated test names are usually prefixed with `c`, e.g. `ctest_unit_core`.
- C++ unit-test breadth is controlled by `-DCUTLASS_TEST_LEVEL=0|1|2`; default `0` is sanity, higher levels are much larger.
- Enable cuBLAS/cuDNN-backed checks only when the libraries are available: `-DCUTLASS_ENABLE_CUBLAS=ON` and `-DCUTLASS_ENABLE_CUDNN=ON`.
- Examples are demonstration code, not benchmark sources; build `cutlass_profiler` with `cmake --build build --target cutlass_profiler -j` for performance profiling.

## Python
- Root `pip install -e .` installs the legacy Python interface packages (`cutlass_cppgen`, `cutlass_library`, `pycute`); optional env overrides are `CUTLASS_PATH` and `CUDA_INSTALL_PATH`.
- Keep `cuda-python` aligned with the CUDA installation selected by `CUDA_INSTALL_PATH`; `python/README.md` calls this out as required.
- Legacy Python tests are `unittest`, not pytest: examples include `python test/python/pycute/run_all_tests.py`, `python test/python/cutlass/gemm/run_all_tests.py`, `python test/python/cutlass/evt/run_all_tests.py`, and `python test/python/cutlass/conv2d/run_all_tests.py`.
- Install CuTe DSL wheels through `bash python/CuTeDSL/setup.sh` (`--cu13` for CUDA 13); editable mode requires `CUTLASS_IR_BUILD_DIR=<cmake-build-dir>` then `bash python/CuTeDSL/setup.sh --editable`.
- CuTe DSL example tests require a CUDA GPU and run with pytest, e.g. `pytest test/examples/CuTeDSL --test-level L0 --target-cc 90 --deselect-not-run`; omit `--target-cc` only if local GPU auto-detection is desired.
- CuTe DSL pytest supports `--runtime-sm` for exact targets like `100f`, L0/L1/L2 markers, `arch`, `invalid_case`, `xfail_case`, and `large_case`; `--only-large-case` is required for large cases.
- CuTe DSL test files must not mutate `sys.path` directly; `conftest.py` enforces this, so use `monkeypatch.syspath_prepend(...)` in tests.

## Workflow Notes
- GitHub Actions here are mostly triage plus Blossom CI; real CI is triggered by authorized `/bot run` comments or manual `workflow_dispatch`, not a normal push/PR test matrix.
- There is no repo-local clang-format, ruff, mypy, or pre-commit config in this checkout; avoid broad formatting-only churn.
