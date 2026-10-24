"""vLLM commit/version-specific workarounds, evaluated by `rules`."""

from __future__ import annotations

from .rules import (
    AddReq, DerivedInstall, ExtraIndex, Fact, OverrideReq, PatchSource,
    Platform, Req, Rule,
)

RULES: tuple[Rule, ...] = (
    # Cross-package compatibility (venv layer).
    Rule("cutlass-dsl-4.6-quack", 1,
         "nvidia-cutlass-dsl 4.6 (vLLM FA4/CuTe DSL kernels, 8b5014d3dd3 #32974) "
         "breaks quack-kernels before 0.6.1",
         when=(Req("nvidia-cutlass-dsl", ">=4.6.0"),),
         then=(AddReq("quack-kernels>=0.6.1"),)),
    Rule("flashinfer-0.6.15-cuda-pathfinder", 1,
         "flashinfer-python 0.6.15 needs cuda-pathfinder>=1.5.4 while compiled "
         "test locks still pinned older; override until the lock catches up",
         when=(Req("flashinfer-python", ">=0.6.15"),
               Req("cuda-pathfinder", "<1.5.4")),
         then=(OverrideReq("cuda-pathfinder>=1.5.4"),)),
    Rule("cuda-local-version-index", 1,
         "pinned locks carry +cuNNN local versions (d47b605eca4 #17576; e.g. "
         "torchcodec==0.14.0+cu130) that uv's --torch-backend does not cover",
         when=(Platform("cuda"),),
         then=(ExtraIndex(r"\+cu(\d+)", "https://download.pytorch.org/whl/cu{0}",
                          strategy="unsafe-best-match"),)),
    Rule("flashinfer-jit-cache", 1,
         "c9d33c60dcd (#26443) made FlashInfer a default CUDA dependency; "
         "vLLM's image adds the matching flashinfer-jit-cache (via `flashinfer "
         "download-kernels` since 29e4d2d42e1 #58765)",
         when=(Platform("cuda"), Req("flashinfer-python")),
         then=(DerivedInstall("flashinfer-jit-cache", "flashinfer-python",
                              "https://flashinfer.ai/whl/cu{cuda_variant}"),)),
    Rule("flashinfer-cubin", 1,
         "flashinfer-cubin left PyPI at 0.6.14 (ba47bb5be1b #47669) and "
         "requirements/cuda.txt at 29e4d2d42e1 (#58765); vLLM's image installs "
         "it with `flashinfer download-kernels`. A no-op while still pinned",
         when=(Platform("cuda"), Req("flashinfer-python", ">=0.6.14")),
         then=(DerivedInstall("flashinfer-cubin", "flashinfer-python",
                              "https://flashinfer.ai/whl"),)),

    # vLLM extras.
    Rule("deepep-nccl-override", 1,
         "5d8e90a9661 (#45321, 2026-07-24): DeepEP's Gin backend needs "
         "NCCL_VERSION; vLLM's image overrides nvidia-nccl-cu${CUDA_MAJOR}",
         when=(Fact("ep_supported"), Fact("nccl_version")),
         then=(OverrideReq("nvidia-nccl-cu{cuda_major}=={nccl_version}",
                           layer="ep"),)),
    Rule("pplx-with-tests-flag", 1,
         "historical pplx-kernels pass -WITH_TESTS=OFF, which CMake 4 rejects; "
         "pplx was dropped in eb19955c370 (#33724)",
         when=(Fact("ep_supported"), Fact("pplx_ref")),
         then=(PatchSource("pplx-kernels/setup.py", '"-WITH_TESTS=OFF"',
                           '"-DWITH_TESTS=OFF"', layer="ep"),)),
    Rule("kv-installer-venv", 1,
         "816cc73a9be (#44266) KV CI helper targets the container's system "
         "Python; run it against the managed venv",
         when=(Fact("kv_supported"),),
         then=(PatchSource(".buildkite/scripts/install-kv-connectors.sh",
                           " --system", "", layer="kv"),)),
)
