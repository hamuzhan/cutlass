# AGENTS.md

CUTLASS: open-source CUDA C++ template library and Python DSLs for high-performance GEMM, convolution, attention, and related linear algebra kernels on NVIDIA GPUs.

> If a `CLAUDE.local.md` file exists alongside this file, read and respect it - it contains developer-specific overrides that supplement this shared guidance.

## Rules (Read First)

**CRITICAL (YOU MUST):**
- Read and follow `media/docs/cpp/programming_guidelines.md` before C++/CUDA code changes.
- For CuTe C++ changes, also read `media/docs/cpp/cute/00_quickstart.md` and the nearby CuTe docs relevant to the touched abstraction.
- For CuTe DSL changes, read `media/docs/pythonDSL/overview.rst`, `media/docs/pythonDSL/quick_start.rst`, and relevant pages under `media/docs/pythonDSL/cute_dsl_general/`.
- Preserve the NVIDIA copyright and SPDX header style already used in the nearest comparable file. Add the current year to new files and update modified file years only when the file already carries that convention.
- Always choose a focused `CUTLASS_NVCC_ARCHS` value for local CMake work. The default expands to every architecture supported by the detected CUDA toolkit and is slow.
- Hopper and Blackwell accelerated kernels require suffixed architecture targets such as `90a`, `100a`, `100f`, or `120a`; `90`, `100`, and `120` are not substitutes.
- Do not edit generated build outputs, generated manifests, or generated kernel lists under a CMake build tree.
- Avoid broad formatting-only churn. This checkout has no repo-local `.clang-format`, ruff, mypy, or pre-commit config.
- Keep changes focused on one concern. Split unrelated C++, Python, docs, generator, and test changes unless the user explicitly asks for a wider change.
- Only commit, push, or create PRs when explicitly requested. Never add AI-tool attribution or co-author lines unless the user explicitly asks.

## Common Commands

| Task | Command |
|------|---------|
| Configure focused C++ build | `cmake -S . -B build -DCUTLASS_NVCC_ARCHS=80` |
| Configure fast non-profiler build | `cmake -S . -B build -DCUTLASS_NVCC_ARCHS=80 -DCUTLASS_ENABLE_LIBRARY=OFF -DCUTLASS_ENABLE_PROFILER=OFF` |
| Configure Hopper profiler build | `cmake -S . -B build -DCUTLASS_NVCC_ARCHS=90a -DCUTLASS_LIBRARY_KERNELS=cutlass3x* -DCUTLASS_UNITY_BUILD_ENABLED=ON` |
| Configure Blackwell profiler build | `cmake -S . -B build -DCUTLASS_NVCC_ARCHS=100f -DCUTLASS_LIBRARY_KERNELS=cutlass3x_sm100_tensorop_gemm_* -DCUTLASS_UNITY_BUILD_ENABLED=ON` |
| Build all C++ unit tests | `cmake --build build --target test_unit -j` |
| Build focused C++ unit test | `cmake --build build --target test_unit_core -j` |
| Run focused CTest filter | `ctest --test-dir build -R ctest_unit_core` |
| Build/run an example test target | `cmake --build build --target test_examples_00_basic_gemm -j` |
| Build profiler | `cmake --build build --target cutlass_profiler -j` |
| Run profiler GEMM smoke test | `./build/tools/profiler/cutlass_profiler --kernels=sgemm --m=1024 --n=1024 --k=1024` |
| Root Python editable install | `pip install -e .` |
| PyCuTe legacy tests | `python test/python/pycute/run_all_tests.py` |
| Legacy GEMM Python tests | `python test/python/cutlass/gemm/run_all_tests.py` |
| Legacy EVT Python tests | `python test/python/cutlass/evt/run_all_tests.py` |
| Legacy Conv2d Python tests | `python test/python/cutlass/conv2d/run_all_tests.py` |
| CuTe DSL wheel install | `bash python/CuTeDSL/setup.sh` |
| CuTe DSL CUDA 13 install | `bash python/CuTeDSL/setup.sh --cu13` |
| CuTe DSL editable install | `CUTLASS_IR_BUILD_DIR=build bash python/CuTeDSL/setup.sh --editable` |
| CuTe DSL L0 tests | `pytest test/examples/CuTeDSL --test-level L0 --target-cc 90 --deselect-not-run` |
| CuTe DSL exact runtime SM | `pytest test/examples/CuTeDSL --test-level L0 --runtime-sm 100f --deselect-not-run` |

## Installation & Build

CUTLASS C++ requires CUDA Toolkit 11.4 or later, CMake 3.18+, a C++17-capable host compiler, and Python for CMake-time generator scripts. CUDA 12.8+ is preferred for Blackwell SM100/SM120 work, and CUDA 13.x is required for newer Blackwell-family targets described by the docs.

Prefer out-of-tree builds from the repository root. Set `CUDACXX=${CUDA_INSTALL_PATH}/bin/nvcc` when the intended `nvcc` is not first on `PATH`.

The default configure enables examples, tools, library/profiler, and tests for a top-level build. For header-only or focused development, disable expensive surfaces with `CUTLASS_ENABLE_LIBRARY=OFF`, `CUTLASS_ENABLE_PROFILER=OFF`, or `CUTLASS_ENABLE_TESTS=OFF` as appropriate.

`tools/library` runs `python/cutlass_library/generator.py` during configure and writes generated artifacts into the build tree. Change generator sources or manifest logic, not generated outputs.

Avoid `-DCUTLASS_LIBRARY_KERNELS=all` unless explicitly required. It can instantiate tens of thousands to millions of kernels. Prefer `CUTLASS_LIBRARY_KERNELS=<glob>` and use `CUTLASS_LIBRARY_INSTANTIATION_LEVEL` only with a non-empty kernel filter.

Enable cuBLAS/cuDNN-backed checks only when the libraries are available: `-DCUTLASS_ENABLE_CUBLAS=ON` and `-DCUTLASS_ENABLE_CUDNN=ON`.

## Repo Map

| Surface | Role | Key Paths |
|---------|------|-----------|
| CUTLASS C++ headers | Header-only CUDA C++ template library for consumers | `include/cutlass/` |
| CuTe C++ headers | Layout, tensor, atom, and tiled operation vocabulary | `include/cute/` |
| Examples | Demonstrations and SDK-style samples, not benchmark sources | `examples/` |
| C++ unit tests | GoogleTest/CMake unit-test tree | `test/unit/` |
| Instance library | Generated kernel manifest and host launch library | `tools/library/`, `python/cutlass_library/` |
| Profiler | Functional and performance profiling executable | `tools/profiler/` |
| Utilities | Reference kernels, host/device utilities, test helpers | `tools/util/`, `test/utils/` |
| Legacy Python interface | High-level Python API for emitting/running CUTLASS kernels | `python/cutlass_cppgen/` |
| Kernel generator Python | Enumerates and emits C++ profiler/library kernels | `python/cutlass_library/` |
| PyCuTe | Python layout algebra utilities | `python/pycute/` |
| CuTe DSL package | Python-native kernel DSL implementation | `python/CuTeDSL/cutlass/` |
| CuTe DSL examples | Python DSL examples and tutorials | `examples/python/CuTeDSL/` |
| CuTe DSL tests | Pytest-based example and DSL tests | `test/examples/CuTeDSL/` |
| Documentation source | C++ and Python DSL docs | `media/docs/`, `python/docs_src/` |

## Architecture

### CUTLASS C++ Stack

| Layer | Entry Points | Notes |
|-------|--------------|-------|
| Device API | `include/cutlass/gemm/device/` | Host-facing adapters that configure and launch kernels |
| Kernel API | `include/cutlass/gemm/kernel/` | CUDA kernel entry points and scheduling integration |
| Collective API | `include/cutlass/gemm/collective/` | Mainloop and epilogue composition for CUTLASS 3.x kernels |
| CuTe tiled ops | `include/cute/atom/`, `include/cute/algorithm/` | `TiledMma`, `TiledCopy`, layouts, tensors, atoms |
| Architecture wrappers | `include/cutlass/arch/`, `include/cute/arch/` | PTX/CUDA instruction exposure and architecture traits |

### Build And Generation Flow

```text
CMake configure
    -> python/cutlass_library/generator.py
    -> build/tools/library/generated/
    -> CUTLASS Instance Library manifest
    -> tools/profiler/cutlass_profiler
```

### CUTLASS 3.x GEMM Flow

```text
Device adapter
    -> Kernel schedule
    -> Collective mainloop + Collective epilogue
    -> CuTe tiled copy/MMA atoms
    -> CUDA/PTX instructions
```

### CuTe DSL Flow

```text
Python DSL kernel
    -> CUTLASS/CuTe DSL IR and MLIR lowering
    -> ptxas/CUDA code generation
    -> Python framework or direct runtime launch
```

## Key Files

| File | Role |
|------|------|
| `CMakeLists.txt` | Top-level CMake options, architecture selection, test/library/profiler enablement |
| `include/cutlass/gemm/collective/` | CUTLASS 3.x collective mainloop and epilogue building blocks |
| `include/cutlass/gemm/device/` | Public GEMM device adapters used by examples and consumers |
| `include/cute/layout.hpp` | Core CuTe layout vocabulary |
| `include/cute/tensor.hpp` | Core CuTe tensor vocabulary |
| `include/cute/atom/` | MMA and copy atom definitions |
| `python/cutlass_library/generator.py` | CMake-time generator entry point for library/profiler kernels |
| `tools/library/include/cutlass/library/manifest.h` | Runtime manifest interface for generated operation instances |
| `tools/profiler/src/cutlass_profiler.cu` | Profiler executable entry point |
| `test/unit/CMakeLists.txt` | C++ unit-test target wiring |
| `test/utils/test_sharding.py` | CuTe DSL pytest options and marker selection logic |
| `test/examples/CuTeDSL/conftest.py` | CuTe DSL pytest setup and `sys.path` guard |
| `python/CuTeDSL/setup.sh` | CuTe DSL wheel/editable installation helper |
| `media/docs/cpp/programming_guidelines.md` | C++/CUDA style and design rules |
| `media/docs/cpp/code_organization.md` | Repository layout and component map |
| `media/docs/cpp/gemm_api_3x.md` | CUTLASS 3.x GEMM API details |
| `media/docs/pythonDSL/cute_dsl_general/naming_conventions.rst` | CuTe DSL example naming conventions |

## Design Patterns

| Pattern | Key Points |
|---------|------------|
| Header-only C++ library | Consumer-facing CUTLASS and CuTe APIs live in headers under `include/`; CMake builds examples, tests, tools, and generated libraries |
| Template specialization | Performance-critical choices are usually compile-time template parameters: tile shapes, data types, layouts, schedules, and architecture tags |
| `Params` objects | Classes with nontrivial launch-time state often define `Params` for grid-invariant data stored efficiently for device use |
| `SharedStorage` | Components that need shared memory define nested `SharedStorage`; compose storage carefully and respect C++ union lifetime rules |
| Explicit unrolling | Loops expected to unroll should use `CUTLASS_PRAGMA_UNROLL` with compile-time trip counts |
| CUTLASS 3.x hierarchy | Prefer the 3.x model: Atom/TiledMma/TiledCopy -> Collective -> Kernel -> Device adapter |
| CuTe layouts and tensors | Express shape, stride, and partitioning through CuTe vocabulary instead of ad hoc index arithmetic |
| Generated profiler kernels | Add or adjust generator logic in `python/cutlass_library/`; do not patch generated C++ in the build tree |
| Tests mirror source hierarchy | C++ tests under `test/unit/` follow source areas such as `core`, `gemm`, `cute`, `epilogue`, `reduction`, and `transform` |

## Anti-Patterns / Gotchas

- Do not rely on the default `CUTLASS_NVCC_ARCHS`; it expands broadly and makes configure/build slow.
- Do not use unsuffixed `90`, `100`, or `120` for kernels that require architecture-accelerated instructions. Use `90a`, `100a`, `100f`, `120a`, or the documented target.
- Do not assume Blackwell datacenter SM100 and GeForce SM120 targets are compatible. They are distinct.
- Do not use `CUTLASS_LIBRARY_KERNELS=all` for routine work. Filter kernels by glob.
- Do not edit `build/tools/library/generated/`, generated manifests, or generated kernel listing files.
- Do not treat examples as benchmark sources. Use `cutlass_profiler` for performance profiling.
- Do not run broad automatic formatters on CUTLASS code. The programming guidelines explicitly avoid automatic formatting.
- Do not add direct `sys.path` mutations in CuTe DSL tests. `test/examples/CuTeDSL/conftest.py` enforces this; use `monkeypatch.syspath_prepend(...)` in tests.
- Do not assume Python tests all use pytest. Legacy Python tests are `unittest`; CuTe DSL example tests use pytest.
- Do not enable cuBLAS/cuDNN verification unless the local environment has those libraries.
- Do not assume GPU tests can run on any machine. Many C++ and CuTe DSL tests require a CUDA GPU and an architecture-compatible build.
- Do not assume Windows is a healthy target for current CUTLASS 4.x builds; the README calls out known Windows build issues.

## Development Workflow

1. Identify the surface being changed: C++ headers, CMake/tools, generator, profiler, legacy Python, CuTe DSL package, examples, tests, or docs.
2. Read the relevant docs before editing, especially `media/docs/cpp/programming_guidelines.md` for C++ and `media/docs/pythonDSL/` for CuTe DSL.
3. Configure an out-of-tree build with a focused `CUTLASS_NVCC_ARCHS` value.
4. Make the smallest correct change and avoid unrelated formatting.
5. Add or update the closest focused test when behavior changes.
6. Run the smallest meaningful build/test command first, then broaden only when needed.
7. For profiler/library work, verify generator changes by reconfiguring CMake and building `cutlass_profiler` or a filtered target.
8. For CuTe DSL work, install through `python/CuTeDSL/setup.sh` and run a targeted `pytest test/examples/CuTeDSL ...` command.

## Branching Policy And PRs

- The main upstream repository is `https://github.com/NVIDIA/cutlass`.
- Push branches to the user-specified fork or remote, usually `origin`, only when the user asks.
- Open PRs against the main repository unless the user specifies a fork or release branch target.
- Target `main` unless explicitly fixing a release branch issue.
- Before committing, inspect `git status`, `git diff`, and recent history. Stage only intended files.
- If commit hooks or generated steps modify files, review the modifications before staging them.
- Do not add co-authors, AI attribution, or manual sign-off lines unless explicitly instructed. If sign-off is required, use `git commit -s` rather than writing the sign-off line by hand.

### GitHub CLI Authentication (`GH_CONFIG_DIR`)

The `gh` CLI uses `~/.config/gh` by default for authentication. Different GitHub hosts or forks may require a different config directory. Before running any `gh` command such as `gh pr create`, `gh api`, or `gh pr comment`:

1. Check whether the user specified a custom `GH_CONFIG_DIR` in the environment or in `CLAUDE.local.md`.
2. If no custom directory is set, ask whether the default `~/.config/gh` is correct.
3. Prefix all `gh` commands with the resolved config dir: `GH_CONFIG_DIR=<path> gh ...`.

## CI / Testing

| Layer | Location | Notes |
|-------|----------|-------|
| C++ unit tests | `test/unit/` | Built through CMake targets such as `test_unit` or `test_unit_gemm_warp` |
| CTest | build tree | Use `ctest --test-dir build -R <regex>`; generated names commonly start with `ctest_` |
| Test breadth | CMake | `CUTLASS_TEST_LEVEL=0|1|2`; level 0 is sanity, higher levels are larger |
| Profiler tests | `tools/profiler/` | Require profiler/library enabled; verification providers may need cuBLAS/cuDNN |
| Legacy Python tests | `test/python/` | `unittest` runners, not pytest |
| CuTe DSL tests | `test/examples/CuTeDSL/` | Pytest, CUDA GPU required, supports `--test-level`, `--target-cc`, `--runtime-sm`, `--deselect-not-run` |
| CuTe DSL large cases | `test/examples/CuTeDSL/` | `large_case` tests are skipped unless `--only-large-case` is specified |
| CI entry point | `.github/workflows/blossom-ci.yml` | Authorized `/bot run` comments or manual `workflow_dispatch`, not a normal push matrix |

### Triggering CI

CI is handled through Blossom CI for authorized users. Basic PR comments:

- `/bot run` triggers the standard CI pipeline.
- `/bot kill` requests cancellation through the Blossom workflow.
- Manual `workflow_dispatch` is available from the `Blossom-CI` workflow when the required inputs are known.

GitHub Actions in this repo also handle issue triage, labels, stale issues, and Blossom dispatch. Do not assume a normal push or PR automatically runs the full CUDA test matrix.

## Key Documentation

| Topic | Path |
|-------|------|
| C++ quick start | `media/docs/cpp/quickstart.md` |
| Code organization | `media/docs/cpp/code_organization.md` |
| Programming guidelines | `media/docs/cpp/programming_guidelines.md` |
| CUTLASS 3.x design | `media/docs/cpp/cutlass_3x_design.md` |
| CUTLASS 3.x GEMM API | `media/docs/cpp/gemm_api_3x.md` |
| CuTe C++ quick start | `media/docs/cpp/cute/00_quickstart.md` |
| Profiler | `media/docs/cpp/profiler.md` |
| Kernel functionality | `media/docs/cpp/functionality.md` |
| Blackwell functionality | `media/docs/cpp/blackwell_functionality.md` |
| Pipeline utilities | `media/docs/cpp/pipeline.md` |
| Heuristics | `media/docs/cpp/heuristics.md` |
| Legacy Python packages | `python/README.md` |
| CuTe DSL overview | `media/docs/pythonDSL/overview.rst` |
| CuTe DSL quick start | `media/docs/pythonDSL/quick_start.rst` |
| CuTe DSL naming conventions | `media/docs/pythonDSL/cute_dsl_general/naming_conventions.rst` |
| CuTe DSL limitations | `media/docs/pythonDSL/limitations.rst` |
| CuTe DSL FAQs | `media/docs/pythonDSL/faqs.rst` |
