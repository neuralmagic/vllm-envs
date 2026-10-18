import contextlib
import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch

from vllm_envs import rules
from vllm_envs.config import Config
from vllm_envs.envsync import sync
from vllm_envs.hashing import _derived_requirement_parts
from vllm_envs.reqs import cap_constraints
from vllm_envs.store import write_meta
from vllm_envs.treefacts import TreeFacts
from vllm_envs.venvlayer import (
    ResolutionError,
    _derived_installs,
    _install_req_groups,
    _install_reqs,
    ensure_full_template,
    uv_pip_install,
)


def compatibility_requirements(files):
    ctx = rules.Context("cuda", "x86_64", files=tuple(files))
    return sorted({e.spec for _, e in rules.effects(ctx, rules.AddReq)})


def compatibility_overrides(files):
    ctx = rules.Context("cuda", "x86_64", files=tuple(files))
    return sorted(
        {e.spec for _, e in rules.effects(ctx, rules.OverrideReq, layer="venv")}
    )


class CompatibilityRulesTest(unittest.TestCase):
    def requirement_file(self, content: str) -> Path:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        requirements = root / "cuda.txt"
        requirements.write_text(content)
        return requirements

    def key_parts(self, requirements: Path) -> list[str]:
        facts = TreeFacts(requirements.parent, "cuda", "x86_64")
        return _derived_requirement_parts([requirements], facts)

    def test_cutlass_46_raises_quack_floor_and_cap(self):
        requirements = self.requirement_file(
            "nvidia-cutlass-dsl[cu13]==4.6.0\n"
            "quack-kernels>=0.4.0\n"
        )

        derived = compatibility_requirements([requirements])

        self.assertEqual(derived, ["quack-kernels>=0.6.1"])
        self.assertEqual(
            cap_constraints(
                [requirements], "minor", extra_requirements=derived
            ),
            "quack-kernels<0.7\n",
        )
        self.assertIn(
            "rule:cutlass-dsl-4.6-quack@1", self.key_parts(requirements)
        )

    def test_cutlass_before_46_does_not_raise_quack_floor(self):
        requirements = self.requirement_file(
            "nvidia-cutlass-dsl[cu13]==4.5.2\n"
            "quack-kernels>=0.4.0\n"
        )

        self.assertEqual(compatibility_requirements([requirements]), [])

    def test_flashinfer_0615_overrides_stale_cuda_pathfinder_pin(self):
        requirements = self.requirement_file(
            "flashinfer-python==0.6.15.post1\n"
            "cuda-pathfinder==1.3.3\n"
        )

        self.assertEqual(
            compatibility_overrides([requirements]),
            ["cuda-pathfinder>=1.5.4"],
        )
        self.assertEqual(compatibility_requirements([requirements]), [])
        parts = self.key_parts(requirements)
        self.assertIn("rule:flashinfer-0.6.15-cuda-pathfinder@1", parts)
        self.assertIn("rule:flashinfer-jit-cache@1", parts)

    def test_flashinfer_rule_requires_a_stale_pathfinder_declaration(self):
        for pathfinder in (None, "1.5.4", "1.6.0"):
            content = "flashinfer-python==0.6.15.post1\n"
            if pathfinder is not None:
                content += f"cuda-pathfinder=={pathfinder}\n"
            with self.subTest(pathfinder=pathfinder):
                requirements = self.requirement_file(content)
                self.assertEqual(compatibility_overrides([requirements]), [])

    def test_pathfinder_override_requires_flashinfer_0615(self):
        requirements = self.requirement_file(
            "flashinfer-python==0.6.14\n"
            "cuda-pathfinder==1.3.3\n"
        )

        self.assertEqual(compatibility_overrides([requirements]), [])

    def test_install_passes_compatibility_overrides_to_uv(self):
        requirements = self.requirement_file(
            "flashinfer-python==0.6.15.post1\n"
            "cuda-pathfinder==1.3.3\n"
        )
        root = requirements.parent
        cap_dest = root / "constraints.txt"
        keys = SimpleNamespace(
            layout=SimpleNamespace(
                build_files=[],
                runtime_files=[requirements],
                test_files=[],
            )
        )

        with (
            patch("vllm_envs.venvlayer.uv_pip_install") as uv_pip,
            patch("vllm_envs.venvlayer.run_retrying_transient"),
            patch("vllm_envs.venvlayer._derived_installs"),
        ):
            _install_reqs(
                SimpleNamespace(cap="none"),
                keys,
                root / ".venv",
                [requirements],
                "cuda",
                cap_dest,
            )

        override_path = root / "overrides.txt"
        self.assertEqual(override_path.read_text(), "cuda-pathfinder>=1.5.4\n")
        args = uv_pip.call_args.args[1]
        self.assertIn("--overrides", args)
        self.assertEqual(args[args.index("--overrides") + 1], str(override_path))

    def _capped_install(self, install_effect):
        requirements = self.requirement_file("pyzmq>=25.0.0\n")
        root = requirements.parent
        keys = SimpleNamespace(layout=SimpleNamespace(
            build_files=[], runtime_files=[requirements], test_files=[]))
        with (
            patch("vllm_envs.venvlayer.uv_pip_install",
                  side_effect=install_effect) as uv_pip,
            patch("vllm_envs.venvlayer.run_retrying_transient"),
            patch("vllm_envs.venvlayer._derived_installs"),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            mode = _install_reqs(SimpleNamespace(cap="minor"), keys,
                                 root / ".venv", [requirements], "cuda",
                                 root / "constraints.txt")
        return mode, uv_pip

    def test_capped_sdist_build_failure_loosens_caps(self):
        # pyzmq<25.1 has no cp312 wheel and its sdist fails under g++ 14.
        sdist = subprocess.CalledProcessError(
            1, ["uv"], output="× Failed to build `pyzmq==25.0.2`")
        mode, uv_pip = self._capped_install([sdist, None])
        self.assertEqual(mode, "major")
        self.assertEqual(uv_pip.call_count, 2)

    def test_transient_install_failure_keeps_caps(self):
        flaky = subprocess.CalledProcessError(
            2, ["uv"], output="429 Too Many Requests")
        with self.assertRaises(subprocess.CalledProcessError):
            self._capped_install([flaky, None])

    def test_rules_are_declarative(self):
        requirements = self.requirement_file("example-runtime~=2.3\n")
        rule = rules.Rule(
            "example", 1, "test",
            when=(rules.Req("example-runtime", ">=2.3.0"),),
            then=(rules.AddReq("example-plugin>=5.1"),),
        )
        ctx = rules.Context("cuda", "x86_64", files=(requirements,))

        self.assertEqual(rules.fired(ctx, (rule,)), [rule])
        self.assertEqual(rules.key_parts(ctx, "venv", (rule,)), ["rule:example@1"])


class FlashInferJitCacheTest(unittest.TestCase):
    def test_required_flashinfer_installs_matching_cuda_jit_cache(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        requirements = root / "cuda.txt"
        requirements.write_text("flashinfer-python==0.6.13\n")
        venv = root / ".venv"

        with (
            patch(
                "vllm_envs.venvlayer.run",
                side_effect=[
                    SimpleNamespace(stdout="Version: 0.6.13\n"),
                    SimpleNamespace(stdout="13.0\n"),
                ],
            ) as run,
            patch("vllm_envs.venvlayer.uv_pip_install") as uv_pip,
        ):
            _derived_installs(
                venv, rules.Context("cuda", "x86_64", files=(requirements,))
            )

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
                "--no-deps",
                "--index-url",
                "https://flashinfer.ai/whl/cu130",
            ],
        )

    def test_uv_install_retries_index_rate_limits(self):
        limited = subprocess.CalledProcessError(
            2, ["uv"], output="429 Too Many Requests"
        )
        with (
            patch("vllm_envs.util.run", side_effect=[limited, None]) as run,
            patch("vllm_envs.util.time.sleep") as sleep,
        ):
            uv_pip_install(Path("/venv"), ["example"])

        self.assertEqual(run.call_count, 2)
        sleep.assert_called_once_with(5)


class InstallReqGroupsTest(unittest.TestCase):
    def setUp(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.venv = root / ".venv"
        self.cap = root / "constraints.txt"
        self.test_files = [root / "test-cuda.txt"]
        self.runtime_files = [root / "cuda.txt"]
        self.groups = [self.test_files, self.runtime_files]

    def test_resolves_all_groups_in_one_pass(self):
        with patch("vllm_envs.venvlayer._install_reqs") as install:
            _install_req_groups(
                "cfg", "keys", self.venv, self.groups, "cuda", self.cap
            )

        install.assert_called_once_with(
            "cfg", "keys", self.venv,
            [*self.test_files, *self.runtime_files], "cuda", self.cap,
        )

    def test_falls_back_to_one_pass_per_group(self):
        # A stale test lock (cuda-pathfinder==1.3.3) against a runtime pin that
        # needs a newer one: unsatisfiable together, fine in separate passes.
        passes = []

        def install(cfg, keys, venv, files, platform, cap_dest):
            passes.append(list(files))
            if len(passes) == 1:  # the combined resolve
                raise ResolutionError(1, ["uv"])
            return "minor"

        with patch("vllm_envs.venvlayer._install_reqs", side_effect=install):
            outcome = _install_req_groups(
                "cfg", "keys", self.venv, self.groups, "cuda", self.cap
            )

        self.assertEqual(outcome, {"cap": ["minor", "minor"], "split_passes": True})

        self.assertEqual(
            passes,
            [
                [*self.test_files, *self.runtime_files],
                self.test_files,
                self.runtime_files,
            ],
        )

    def test_install_failure_of_resolvable_set_does_not_split(self):
        with patch("vllm_envs.venvlayer._install_reqs",
                   side_effect=subprocess.CalledProcessError(1, ["uv"])) as install:
            with self.assertRaises(subprocess.CalledProcessError):
                _install_req_groups(
                    "cfg", "keys", self.venv, self.groups, "cuda", self.cap
                )
        self.assertEqual(install.call_count, 1)

    def test_single_group_failure_propagates(self):
        with patch(
            "vllm_envs.venvlayer._install_reqs",
            side_effect=subprocess.CalledProcessError(1, ["uv"]),
        ) as install:
            with self.assertRaises(subprocess.CalledProcessError):
                _install_req_groups(
                    "cfg", "keys", self.venv, [self.runtime_files, []],
                    "cuda", self.cap,
                )

        install.assert_called_once()

    def test_no_requirement_files_is_a_noop(self):
        with patch("vllm_envs.venvlayer._install_reqs") as install:
            _install_req_groups("cfg", "keys", self.venv, [[], []], "cuda", self.cap)

        install.assert_not_called()


class FullTemplateRecoveryTest(unittest.TestCase):
    def test_resumes_matching_incomplete_template(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        cfg = Config(cache_dir=root / "cache")
        keys = SimpleNamespace(
            base_hash="base", full_hash="full",
            layout=SimpleNamespace(test_files=[], runtime_files=[]),
        )
        base = cfg.store("venvs-base") / "base"
        (base / "bin").mkdir(parents=True)
        (base / ".complete").touch()
        entry = cfg.store("venvs") / "full"
        (entry / "bin").mkdir(parents=True)
        (entry / "bin" / "python").touch()
        write_meta(
            entry,
            {
                "kind": "venv-full", "hash": "full", "base": "base",
                "state": "incomplete", "stage": "installing dependencies",
            },
        )

        with (
            patch("vllm_envs.venvlayer._install_req_groups") as install,
            patch("vllm_envs.venvlayer.venv_freeze", return_value=[]),
            patch("vllm_envs.venvlayer.reflink_clone") as clone,
        ):
            self.assertEqual(ensure_full_template(cfg, keys, "cuda"), entry)

        install.assert_called_once()
        clone.assert_not_called()
        self.assertTrue((entry / ".complete").exists())


class SyncFailureLogTest(unittest.TestCase):
    def test_retains_log_only_after_failure(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        cfg = Config(cache_dir=root / "cache")
        with (
            patch("vllm_envs.envsync.git", return_value="abc"),
            patch("vllm_envs.envsync.resolve_venv", side_effect=RuntimeError("boom")),
            self.assertRaisesRegex(RuntimeError, "boom"),
        ):
            sync(cfg, root)

        failure_log = root / ".ve" / "last-sync-failure.log"
        self.assertTrue(failure_log.exists())

        with (
            patch("vllm_envs.envsync.git", return_value="abc"),
            patch("vllm_envs.envsync.resolve_venv", return_value=(root / ".venv", "keys")),
            patch("vllm_envs.envsync.resolve_build", return_value="build"),
            patch("vllm_envs.envsync.attach"),
            patch.object(cfg, "platform", "cpu"),
        ):
            sync(cfg, root)

        self.assertFalse(failure_log.exists())


if __name__ == "__main__":
    unittest.main()
