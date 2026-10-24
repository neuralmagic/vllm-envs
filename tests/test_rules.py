import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from vllm_envs import rules, treefacts
from vllm_envs.hashing import _generated_output
from vllm_envs.hashing import _derived_requirement_parts, venv_keys
from vllm_envs.hooks import REQ_WATCH_PATHS
from vllm_envs.venvlayer import _torch_backend_args
from vllm_envs.treefacts import FACTS, TreeFacts
from vllm_envs.workarounds import RULES


def rule(rule_id: str) -> rules.Rule:
    return next(r for r in RULES if r.id == rule_id)


class TreeCase(unittest.TestCase):
    def tree(self, files: dict[str, str] | list[str]) -> Path:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        if isinstance(files, list):
            files = {f: "x==1\n" for f in files}
        for name, content in files.items():
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_text(content)
        return root

    def facts(self, files, platform="cuda", machine="x86_64") -> TreeFacts:
        return TreeFacts(self.tree(files), platform, machine)


class RuleTableTest(unittest.TestCase):
    def test_ids_are_unique(self):
        ids = [r.id for r in RULES]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_rule_cites_a_reason(self):
        for r in RULES:
            with self.subTest(rule=r.id):
                self.assertTrue(r.reason.strip())
                self.assertGreaterEqual(r.rev, 1)
                self.assertTrue(r.then)

    def test_fact_predicates_name_registered_facts(self):
        for r in RULES:
            for p in r.when:
                if isinstance(p, rules.Fact):
                    self.assertIn(p.name, FACTS, r.id)

    def test_every_fact_documents_detector_origin_and_gate(self):
        for spec in FACTS.values():
            with self.subTest(fact=spec.name):
                self.assertTrue(spec.detects and spec.since and spec.gates)

    def test_effects_name_a_layer(self):
        for r in RULES:
            for e in r.then:
                self.assertIn(e.layer, {"venv", "ep", "kv"}, r.id)


class LayoutRulesTest(TreeCase):
    eras = {
        "root": ["requirements-build.txt", "requirements-common.txt",
                 "requirements-cuda.txt", "requirements-test.txt"],
        "dir": ["requirements/build.txt", "requirements/common.txt",
                "requirements/cuda.txt", "requirements/test.in",
                "requirements/test.txt"],
        "per-platform": ["requirements/build/cuda.txt", "requirements/common.txt",
                         "requirements/cuda.txt", "requirements/test/cuda.in",
                         "requirements/test/cuda.txt"],
    }

    def layout(self, era: str, machine="x86_64", platform="cuda", extra=()):
        facts = self.facts([*self.eras[era], *extra], platform, machine)
        lay = facts.layout(with_test=True)
        rel = lambda files: [p.relative_to(facts.root).as_posix() for p in files]
        return rel(lay.build_files), rel(lay.runtime_files), rel(lay.test_files)

    def test_each_era_selects_its_files(self):
        self.assertEqual(self.layout("root"), (
            ["requirements-build.txt"],
            ["requirements-common.txt", "requirements-cuda.txt"],
            ["requirements-test.txt"]))
        self.assertEqual(self.layout("dir"), (
            ["requirements/build.txt"],
            ["requirements/common.txt", "requirements/cuda.txt"],
            ["requirements/test.txt"]))
        self.assertEqual(self.layout("per-platform"), (
            ["requirements/build/cuda.txt"],
            ["requirements/common.txt", "requirements/cuda.txt"],
            ["requirements/test/cuda.txt"]))

    def test_pinned_locks_only_on_x86_64(self):
        self.assertEqual(self.layout("per-platform", "aarch64")[2],
                         ["requirements/test/cuda.in"])
        self.assertEqual(self.layout("dir", "aarch64")[2], ["requirements/test.in"])
        self.assertEqual(self.layout("root", "aarch64")[2], [])

    def test_mid_era_test_lock_is_cuda_only(self):
        extra = ["requirements/rocm.txt", "requirements/rocm-build.txt"]
        build, _, test = self.layout("dir", platform="rocm", extra=extra)
        self.assertEqual(build, ["requirements/rocm-build.txt"])
        self.assertEqual(test, ["requirements/test.in"])

    def test_test_files_only_when_requested(self):
        facts = self.facts(self.eras["per-platform"])
        self.assertEqual(facts.layout(with_test=False).test_files, [])

    def test_unrecognized_layout_hashes_everything(self):
        facts = self.facts(["requirements/odd.txt", "requirements-x.txt"])
        lay = facts.layout(with_test=True)
        self.assertFalse(lay.recognized)
        self.assertEqual(len(lay.build_files), 2)

    def test_selection_is_not_keyed(self):
        facts = self.facts(self.eras["per-platform"])
        self.assertEqual(rules.key_parts(facts.context(), "venv"), [])


class RequirementRulesTest(TreeCase):
    def test_cuda_local_versions_add_their_pytorch_index(self):
        facts = self.facts({"lock.txt": "torchcodec==0.14.0+cu130\nx==1+cu129\n"})
        ctx = facts.context([facts.root / "lock.txt"])
        self.assertEqual(_torch_backend_args(ctx), [
            "--torch-backend=auto", "--index-strategy", "unsafe-best-match",
            "--extra-index-url", "https://download.pytorch.org/whl/cu129",
            "--extra-index-url", "https://download.pytorch.org/whl/cu130",
        ])
        self.assertIn("rule:cuda-local-version-index@1",
                      rules.key_parts(ctx, "venv"))

    def test_plain_locks_use_only_the_torch_backend(self):
        facts = self.facts({"lock.txt": "torch==2.9.0\n"})
        ctx = facts.context([facts.root / "lock.txt"])
        self.assertEqual(_torch_backend_args(ctx), ["--torch-backend=auto"])
        self.assertEqual(rules.key_parts(ctx, "venv"), [])

    def test_non_cuda_platforms_get_no_backend_args(self):
        facts = self.facts({"lock.txt": "x==1+cu130\n"}, platform="rocm")
        self.assertEqual(_torch_backend_args(facts.context([facts.root / "lock.txt"])), [])

    def test_flashinfer_jit_cache_is_derived_on_cuda_only(self):
        files = {"requirements/cuda.txt": "flashinfer-python==0.6.13\n"}
        for platform, expected in (("cuda", 1), ("rocm", 0)):
            facts = self.facts(files, platform)
            ctx = facts.context([facts.root / "requirements/cuda.txt"])
            self.assertEqual(len(rules.effects(ctx, rules.DerivedInstall)), expected)

    def test_req_predicate_floors(self):
        facts = self.facts({"r.txt": "a==1.4.2\nb>=2\nc\n"})
        ctx = facts.context([facts.root / "r.txt"])
        self.assertTrue(rules.Req("a", ">=1.4").holds(ctx))
        self.assertFalse(rules.Req("a", ">=1.5").holds(ctx))
        self.assertTrue(rules.Req("a", "<1.5").holds(ctx))
        self.assertTrue(rules.Req("c").holds(ctx))
        self.assertFalse(rules.Req("c", "<9").holds(ctx))  # no declared floor
        self.assertFalse(rules.Req("d").holds(ctx))

    def test_rules_enter_only_the_venv_layer_they_fire_for(self):
        root = self.tree({"requirements/build/cuda.txt": "torch==2.9.0\n",
                          "requirements/common.txt": "x==1\n",
                          "requirements/cuda.txt": "flashinfer-python==0.6.13\n"})
        facts = TreeFacts(root, "cuda", "x86_64")
        lay = facts.layout()
        self.assertEqual(_derived_requirement_parts(lay.build_files, facts), [])
        self.assertEqual(_derived_requirement_parts(lay.runtime_files, facts),
                         ["rule:flashinfer-jit-cache@1"])
        keys = venv_keys(root, "cuda", "3.12", facts=facts)
        self.assertIs(keys.facts, facts)


class BuildAndHookRulesTest(TreeCase):
    def test_rust_sources_hashed_only_with_the_rust_frontend(self):
        base = ("csrc", "cmake", "CMakeLists.txt", "setup.py")
        self.assertEqual(self.facts(["setup.py"]).build_layer_paths, base)
        rust = self.facts(["setup.py", "rust-toolchain.toml"])
        self.assertTrue(rust.rust_frontend)
        self.assertEqual(rust.build_layer_paths,
                         base + ("rust", "rust-toolchain.toml", "tools/build_rust.py"))

    def test_hooks_watch_every_input_any_rule_may_use(self):
        for path in ("requirements", "csrc", "setup.py", "rust", "rust-toolchain.toml",
                     "tools/build_rust.py", "docker/versions.json", "docker/Dockerfile",
                     "tools/ep_kernels",
                     ".buildkite/scripts/install-kv-connectors.sh"):
            self.assertIn(path, REQ_WATCH_PATHS)

    def test_generated_marlin_sources_are_ignored(self):
        self.assertTrue(_generated_output("csrc/quantization/marlin/sm80_kernel_fp16.cu"))
        self.assertTrue(_generated_output("csrc/moe/marlin_moe_wna16/kernel_selector.h"))
        self.assertFalse(_generated_output("csrc/quantization/marlin/marlin.cu"))

    def test_wheel_paths_from_setup_or_defaults(self):
        self.assertEqual(self.facts(["setup.py"]).wheel_blocking_paths,
                         treefacts.DEFAULT_BLOCKING_PATHS)
        facts = self.facts({"setup.py": "class U:\n    WHEEL_BLOCKING_PATHS = ('csrc',)\n"})
        self.assertTrue(facts.wheel_paths_declared)
        self.assertEqual(facts.wheel_blocking_paths, ("csrc",))
        self.assertEqual(facts.wheel_warning_paths, treefacts.DEFAULT_WARNING_PATHS)


class ExtrasFactsTest(TreeCase):
    installer = "tools/ep_kernels/install_python_libraries.sh"

    def test_ep_support_needs_installer_and_ref(self):
        eras = (
            ("set -ex\n", "no DeepEP ref"),
            ('DEEPEP_COMMIT_HASH=${DEEPEP_COMMIT_HASH:-"73b6ea4"}\nNVSHMEM_VER=3.3.9\n',
             ""),
        )
        self.assertIn("no EP installer", self.facts([]).ep_unsupported_reason())
        for script, reason in eras:
            with self.subTest(reason=reason):
                facts = self.facts({self.installer: script})
                self.assertEqual(facts.ep_supported, not reason)
                self.assertIn(reason, facts.ep_unsupported_reason())

    def test_docker_manifest_pins_win(self):
        facts = self.facts({
            self.installer: 'DEEPEP_COMMIT_HASH=${DEEPEP_COMMIT_HASH:-"old"}\n',
            "docker/versions.json": json.dumps({"variable": {
                "DEEPEP_COMMIT_HASH": {"default": "new"},
                "NCCL_VERSION": {"default": "2.30.7"}}}),
        })
        self.assertEqual((facts.deepep_ref, facts.deepep_ref_source), ("new", "docker/versions.json"))
        self.assertEqual(facts.nccl_version, "2.30.7")

    def test_nccl_override_and_pplx_patch_need_ep_support(self):
        script = 'PPLX_COMMIT_HASH=${PPLX_COMMIT_HASH:-"12cecfd"}\n'
        nccl = json.dumps({"variable": {"NCCL_VERSION": {"default": "2.30.7"}}})
        unsupported = self.facts({self.installer: script, "docker/versions.json": nccl})
        self.assertEqual(rules.key_parts(unsupported.context(), "ep"), [])
        supported = self.facts({
            self.installer: script + 'DEEPEP_COMMIT_HASH=${DEEPEP_COMMIT_HASH:-"r"}\n'
                                     'NVSHMEM_VER=${NVSHMEM_VER:-"3.3.24"}\n',
            "docker/versions.json": nccl,
        })
        self.assertEqual(rules.key_parts(supported.context(), "ep"),
                         ["rule:deepep-nccl-override@1", "rule:pplx-with-tests-flag@1"])

    def test_kv_support_needs_installer_and_requirements(self):
        kv = ".buildkite/scripts/install-kv-connectors.sh"
        self.assertFalse(self.facts(["requirements/kv_connectors.txt"]).kv_supported)
        facts = self.facts([kv, "requirements/kv_connectors.txt"])
        self.assertTrue(facts.kv_supported)
        (patch,) = [e for _, e in rules.effects(facts.context(), rules.PatchSource, "kv")]
        self.assertEqual(patch.apply("uv pip install --system -r x"), "uv pip install -r x")


class ScopeTest(TreeCase):
    def test_facts_are_shared_within_a_sync_only(self):
        root = self.tree([])
        self.assertIsNot(treefacts.facts_for(root, "cuda"), treefacts.facts_for(root, "cuda"))
        with treefacts.scope():
            self.assertIs(treefacts.facts_for(root, "cuda"), treefacts.facts_for(root, "cuda"))


class RulesCommandTest(TreeCase):
    def test_json_explains_facts_and_fired_rules(self):
        from vllm_envs.commands import diagnostics

        root = self.tree({"requirements/build/cuda.txt": "torch==2.9.0\n",
                          "requirements/common.txt": "x==1\n",
                          "requirements/cuda.txt": "flashinfer-python==0.6.13\n",
                          "rust-toolchain.toml": ""})
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        cfg = SimpleNamespace(platform="cuda", with_test=True)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            diagnostics.cmd_rules(cfg, SimpleNamespace(path=str(root), json=True))
        data = json.loads(out.getvalue())
        self.assertEqual(data["facts"]["rust_frontend"]["value"], True)
        self.assertEqual(data["facts"]["build_requirements"]["value"],
                         ["requirements/build/cuda.txt"])
        fired = {r["id"] for r in data["rules"] if r["fired"]}
        self.assertIn("flashinfer-jit-cache", fired)
        self.assertNotIn("pplx-with-tests-flag", fired)

    def test_text_output(self):
        from vllm_envs.commands import diagnostics

        root = self.tree({"requirements/build/cuda.txt": "torch==2.9.0\n",
                          "requirements/cuda.txt": "flashinfer-python==0.6.13\n"})
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            diagnostics.cmd_rules(SimpleNamespace(platform="cuda", with_test=True),
                          SimpleNamespace(path=str(root), json=False))
        self.assertIn("flashinfer-jit-cache@1  keys: venv-full", out.getvalue())
        self.assertIn("not firing:", out.getvalue())


if __name__ == "__main__":
    unittest.main()
