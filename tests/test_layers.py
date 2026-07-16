import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch

from vllm_envs.layers import _install_flashinfer_jit_cache


class FlashInferJitCacheTest(unittest.TestCase):
    def test_required_flashinfer_installs_matching_cuda_jit_cache(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        requirements = root / "cuda.txt"
        requirements.write_text("flashinfer-python==0.6.13\n")
        venv = root / ".venv"

        with (
            patch(
                "vllm_envs.layers.run",
                side_effect=[
                    SimpleNamespace(stdout="Version: 0.6.13\n"),
                    SimpleNamespace(stdout="13.0\n"),
                ],
            ) as run,
            patch("vllm_envs.layers._uv_pip") as uv_pip,
        ):
            _install_flashinfer_jit_cache(venv, "cuda", [requirements])

        self.assertEqual(
            run.call_args_list,
            [
                call(
                    [
                        "uv",
                        "pip",
                        "show",
                        "--python",
                        str(venv / "bin" / "python"),
                        "flashinfer-python",
                    ]
                ),
                call(
                    [
                        str(venv / "bin" / "python"),
                        "-c",
                        "import torch; print(torch.version.cuda or '')",
                    ]
                ),
            ],
        )
        uv_pip.assert_called_once_with(
            venv,
            [
                "flashinfer-jit-cache==0.6.13",
                "--index-url",
                "https://flashinfer.ai/whl/cu130",
            ],
        )


if __name__ == "__main__":
    unittest.main()
