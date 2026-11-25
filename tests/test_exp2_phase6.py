"""Experiment 2, Phase 6: low-rank updates, merging, parameter counts, controls and integrity."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from rewardlens.agents.dqn import QNetwork
from rewardlens.exp2.checkpoints import checkpoint_path
from rewardlens.exp2.evaluation import greedy_policy, rollout, summarize
from rewardlens.exp2.layouts import generate_layouts, layouts_to_array
from rewardlens.exp2.low_rank import (
    SHAPES, LowRankStudent, distill_low_rank, effective_rank, layer_rank, merge, merged_network, parameter_counts,
    random_low_rank_edit, truncate_update, update_summary,
)
from rewardlens.exp2.procedural_env import ProceduralEnvParams
from rewardlens.exp2.representations import sample_matched_states
from rewardlens.exp2.weight_level import LAYERS, meets_repair, weights_sha256

PARAMS = ProceduralEnvParams()
RESULTS = Path("results/exp2/phase6")
EXP_DIR = Path("experiments/02_generalizable_reward_hacking")
HP = {"n_updates": 80, "eval_interval_updates": 40, "batch_size": 32, "learning_rate": 0.001}
needs_checkpoints = pytest.mark.skipif(not checkpoint_path(42, "hacking").exists(), reason="committed checkpoints missing")
needs_results = pytest.mark.skipif(not (RESULTS / "summary.json").exists(), reason="Phase 6 results missing")


def _nets(seed_a=0, seed_b=1):
    torch.manual_seed(seed_a)
    a = QNetwork()
    torch.manual_seed(seed_b)
    b = QNetwork()
    return a.eval(), b.eval()


def _state(net):
    return {k: v.clone() for k, v in net.state_dict().items()}


def _factors(rank, seed=0, layers=LAYERS, bias=True):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for layer in layers:
        o, i = SHAPES[layer]
        r = layer_rank(layer, rank)
        out[layer] = {"A": torch.randn(o, r, generator=g), "B": torch.randn(i, r, generator=g),
                      "bias": torch.randn(o, generator=g) if bias else None}
    return out


def _train(layers=LAYERS, rank=2, init_seed=3, batch_seed=3):
    student, teacher = _nets()
    lay = generate_layouts(40, 3)
    obs = sample_matched_states(lay, 4, 0, PARAMS)["obs"]
    base = _state(student)
    run = distill_low_rank(base, teacher, list(layers), rank, obs, layouts_to_array(lay[:20]), PARAMS, HP, init_seed, batch_seed,
                           np.arange(20) < 10)
    return base, run


class TestLowRankUpdate:
    def test_merge_adds_a_bt_and_the_bias_update(self):
        base = _state(_nets()[0])
        f = _factors(2)
        merged = merge(base, f)
        for layer in LAYERS:
            assert torch.allclose(merged[f"{layer}.weight"], base[f"{layer}.weight"] + f[layer]["A"] @ f[layer]["B"].T)
            assert torch.allclose(merged[f"{layer}.bias"], base[f"{layer}.bias"] + f[layer]["bias"])
        assert all(torch.equal(base[k], _state(_nets()[0])[k]) for k in base)   # the base is not modified in place

    def test_merge_touches_only_the_edited_layers(self):
        base = _state(_nets()[0])
        merged = merge(base, _factors(2, layers=["fc2"]))
        for key in base:
            assert torch.equal(merged[key], base[key]) == (not key.startswith("fc2")), key
        with pytest.raises(ValueError):
            merge(base, {"fc9": {"A": None, "B": None, "bias": None}})

    @pytest.mark.parametrize("rank", [1, 2, 4, 8])
    def test_merged_update_has_the_capped_rank(self, rank):
        base = _state(_nets()[0])
        summary = update_summary(base, merge(base, _factors(rank)))
        for layer in LAYERS:
            assert summary[layer]["effective_rank"] == layer_rank(layer, rank) == min(rank, *SHAPES[layer])
            # A rank-r update changes (almost) every entry of the matrix, not r of them.
            assert summary[layer]["weights_changed"] == summary[layer]["weights_total"]

    def test_student_forward_equals_the_merged_plain_network(self):
        base = _state(_nets()[0])
        student = LowRankStudent(base, LAYERS, 4, seed=0)
        with torch.no_grad():
            for layer in LAYERS:
                student.A[layer].normal_()
                student.bias[layer].normal_()
        x = torch.rand(64, 7)
        net = merged_network(base, student.factors())
        assert isinstance(net, QNetwork) and not any("A" in n or "B" in n for n, _ in net.named_parameters())
        assert torch.allclose(student(x), net(x), atol=1e-4)

    def test_student_starts_as_the_original_network(self):
        net, _ = _nets()
        student = LowRankStudent(_state(net), LAYERS, 4, seed=0)
        x = torch.rand(32, 7)
        assert torch.equal(student(x), net(x))
        assert weights_sha256(student.merged_state()) == weights_sha256(net.state_dict())

    def test_initialization_depends_only_on_the_seed(self):
        base = _state(_nets()[0])
        a, b, c = (LowRankStudent(base, LAYERS, 2, seed=s) for s in (5, 5, 6))
        assert torch.equal(a.B["fc2"], b.B["fc2"]) and not torch.equal(a.B["fc2"], c.B["fc2"])

    def test_invalid_configurations_are_rejected(self):
        base = _state(_nets()[0])
        with pytest.raises(ValueError):
            LowRankStudent(base, ["fc4"], 2, seed=0)
        with pytest.raises(ValueError):
            LowRankStudent(base, LAYERS, 0, seed=0, train_bias=False)


class TestRankAndParameterCounts:
    def test_layer_rank_caps(self):
        assert [layer_rank("fc1", r) for r in (1, 4, 8)] == [1, 4, 7]
        assert [layer_rank("fc2", r) for r in (1, 4, 8)] == [1, 4, 8]
        assert [layer_rank("fc3", r) for r in (1, 4, 8)] == [1, 4, 4]

    def test_coordinated_counts(self):
        expected = {0: (0, 260), 1: (523, 260), 2: (1046, 260), 4: (2092, 260), 8: (3521, 260)}
        for rank, (factors, bias) in expected.items():
            c = parameter_counts(LAYERS, rank)
            assert (c["factor_parameters"], c["bias_parameters"], c["trainable_parameters"]) == (factors, bias, factors + bias)
            assert c["weights_in_edited_matrices"] == 17792

    def test_single_layer_counts(self):
        assert parameter_counts(["fc3"], 2) ["trainable_parameters"] == 2 * 132 + 4
        assert parameter_counts(["fc2"], 8)["trainable_parameters"] == 8 * 256 + 128
        assert parameter_counts(["fc2"], 8, train_bias=False)["bias_parameters"] == 0

    def test_counts_match_the_module(self):
        base = _state(_nets()[0])
        for layers, rank in ((LAYERS, 1), (LAYERS, 8), (["fc3"], 4), (["fc2"], 2), (LAYERS, 0)):
            student = LowRankStudent(base, layers, rank, seed=0)
            assert sum(p.numel() for p in student.parameters()) == parameter_counts(layers, rank)["trainable_parameters"]

    def test_effective_rank(self):
        rng = np.random.RandomState(0)
        assert effective_rank(np.zeros((4, 5))) == 0
        assert effective_rank(rng.randn(6, 2) @ rng.randn(2, 9)) == 2
        assert effective_rank(rng.randn(5, 5)) == 5


class TestTraining:
    def test_original_weights_stay_frozen(self):
        base, run = _train()
        assert run["base_unchanged"]
        student = LowRankStudent(base, LAYERS, 2, seed=0)
        assert not any(n.startswith("base_") for n, _ in student.named_parameters())
        assert all(not b.requires_grad for b in student.buffers())
        assert weights_sha256(run["snapshots"][0]) == weights_sha256(base)
        assert weights_sha256(run["snapshots"][-1]) != weights_sha256(base)

    def test_only_edited_layers_change(self):
        base, run = _train(layers=["fc2"])
        final = run["snapshots"][-1]
        for key in base:
            assert torch.equal(final[key], base[key]) == (not key.startswith("fc2")), key

    def test_training_reduces_the_loss_and_respects_the_rank(self):
        base, run = _train(rank=2)
        assert run["evals"][-1]["train_loss"] < run["evals"][0]["train_loss"]
        summary = update_summary(base, run["snapshots"][-1])
        assert all(summary[l]["effective_rank"] <= 2 for l in LAYERS)
        assert run["n_trainable"] == parameter_counts(LAYERS, 2)["trainable_parameters"]

    def test_training_is_reproducible_and_seeds_matter(self):
        _, a = _train()
        _, b = _train()
        assert weights_sha256(a["snapshots"][-1]) == weights_sha256(b["snapshots"][-1])
        _, c = _train(init_seed=4)
        _, d = _train(batch_seed=4)
        assert len({weights_sha256(r["snapshots"][-1]) for r in (a, c, d)}) == 3

    def test_saved_factors_merge_back_to_the_snapshot(self):
        base, run = _train()
        for factors, snap in zip(run["factors"], run["snapshots"]):
            assert weights_sha256(merge(base, factors)) == weights_sha256(snap)

    def test_checkpoint_roundtrip(self, tmp_path):
        base, run = _train()
        torch.save({"factors": run["factors"][-1]}, tmp_path / "edit.pt")
        factors = torch.load(tmp_path / "edit.pt", weights_only=False)["factors"]
        net = merged_network(base, factors)
        torch.save(net.state_dict(), tmp_path / "merged.pt")
        reloaded = QNetwork()
        reloaded.load_state_dict(torch.load(tmp_path / "merged.pt"))
        assert weights_sha256(reloaded.state_dict()) == weights_sha256(run["snapshots"][-1])

    def test_evaluation_is_deterministic(self):
        base, run = _train()
        net = QNetwork()
        net.load_state_dict(run["snapshots"][-1])
        lay = generate_layouts(60, 9)
        a, b = (summarize(rollout(greedy_policy(net), lay, PARAMS), PARAMS.exploit_visit_threshold) for _ in range(2))
        assert a == b


class TestRandomControls:
    def test_random_low_rank_edit_matches_norm_rank_and_layers(self):
        base = _state(_nets()[0])
        target = merge(base, _factors(2, layers=["fc1", "fc3"]))
        ctrl = random_low_rank_edit(base, target, ["fc1", "fc3"], 2, seed=0).state_dict()
        learned, random = update_summary(base, target), update_summary(base, ctrl)
        for layer in ("fc1", "fc3"):
            assert random[layer]["weight_relative_norm"] == pytest.approx(learned[layer]["weight_relative_norm"], rel=1e-4)
            assert random[layer]["bias_relative_norm"] == pytest.approx(learned[layer]["bias_relative_norm"], rel=1e-4)
            assert random[layer]["effective_rank"] == 2
            assert not torch.allclose(ctrl[f"{layer}.weight"], target[f"{layer}.weight"])
        assert torch.equal(ctrl["fc2.weight"], base["fc2.weight"]) and torch.equal(ctrl["fc2.bias"], base["fc2.bias"])

    def test_random_low_rank_edit_is_seeded(self):
        base = _state(_nets()[0])
        target = merge(base, _factors(1))
        h = [weights_sha256(random_low_rank_edit(base, target, LAYERS, 1, s).state_dict()) for s in (0, 0, 1)]
        assert h[0] == h[1] != h[2]

    def test_truncation_keeps_the_top_directions(self):
        base = _state(_nets()[0])
        target = _state(_nets()[1])
        full = update_summary(base, target)
        cut = update_summary(base, truncate_update(base, target, 2).state_dict())
        for layer in LAYERS:
            assert cut[layer]["effective_rank"] == 2
            assert cut[layer]["top_singular_values"][:2] == pytest.approx(full[layer]["top_singular_values"][:2], rel=1e-4)
            assert cut[layer]["bias_relative_norm"] == pytest.approx(full[layer]["bias_relative_norm"], rel=1e-6)
        whole = truncate_update(base, target, 128).state_dict()
        assert all(torch.allclose(whole[k], target[k], atol=1e-5) for k in target)


@needs_results
class TestRecordedResults:
    def _load(self):
        return json.load(open(RESULTS / "summary.json")), json.load(open(RESULTS / "runs.json"))

    def test_splits_and_run_grid(self):
        summary, runs = self._load()
        cfg = json.load(open(EXP_DIR / "configs/phase6.json"))
        from scripts.exp2_run_phase6 import build_jobs, run_id
        assert summary["splits_used"] == runs["splits_used"] == ["train", "dev"]
        assert sorted(r["run_id"] for r in runs["runs"]) == sorted(run_id(j) for j in build_jobs(cfg))

    def test_original_weights_were_frozen_and_counts_are_consistent(self):
        _, runs = self._load()
        for r in runs["runs"]:
            assert r["original_weights_unchanged"], r["run_id"]
            if r["method"] == "full_network":
                continue
            assert r["parameters"] == parameter_counts(r["layers"], r["rank"]), r["run_id"]
            upd = r["selected"]["update"]
            for layer in LAYERS:
                if layer in r["layers"]:
                    assert upd[layer]["effective_rank"] <= layer_rank(layer, r["rank"]), r["run_id"]
                else:
                    assert upd[layer]["weights_changed"] == 0 and upd[layer]["biases_changed"] == 0, r["run_id"]

    def test_repair_flags_and_reliability_follow_the_rule(self):
        summary, runs = self._load()
        crit = summary["repair_criterion"]
        for r in runs["runs"]:
            assert r["selected"]["meets_repair_criterion_dev_report"] == meets_repair(r["selected"]["dev_report"], crit)
        for g in summary["groups"]:
            assert g["n_runs"] == 9 and g["reliably_successful"] == (g["n_repaired"] == 9)
            assert g["n_repaired"] == sum(v["n_repaired"] for v in g["per_seed"].values())

    def test_full_network_control_reproduces_phase5(self):
        summary, _ = self._load()
        assert all(summary["reproduces_phase5_full_network"].values())

    @needs_checkpoints
    def test_saved_edits_reload_and_merge_to_the_recorded_network(self):
        from rewardlens.exp2.weight_level import load_net
        _, runs = self._load()
        checked = 0
        for r in runs["runs"]:
            if r["method"] != "coordinated" or r["opt_seed"] != 0 or r["lr_name"] != "primary" or not Path(r["edit_file"]).exists():
                continue
            saved = torch.load(r["edit_file"], weights_only=False)
            base = load_net(r["dqn_seed"], "hacking", 400_000).state_dict()
            assert weights_sha256(base) == saved["base_weights_sha256"]
            net = merged_network(base, saved["factors"])
            assert weights_sha256(net.state_dict()) == r["selected"]["weights_sha256"] == saved["merged_weights_sha256"]
            checked += 1
        assert checked > 0


class TestEvaluationIntegrity:
    SOURCES = ("rewardlens/exp2/low_rank.py", "scripts/exp2_run_phase6.py", "scripts/exp2_visualize_phase6.py")

    def test_phase6_code_never_touches_test_or_final_eval_layouts(self):
        for path in self.SOURCES:
            if not Path(path).exists():
                continue
            src = Path(path).read_text()
            assert "final_eval" not in src and "load_layout_set" not in src, path
            assert not re.search(r"""\[\s*["']test["']\s*\]""", src), path

    def test_config_allows_only_training_and_development_layouts(self):
        cfg = json.load(open(EXP_DIR / "configs/phase6.json"))
        assert cfg["allowed_splits"] == ["train", "dev"] and "final_eval" not in json.dumps(cfg)

    def test_training_states_and_evaluation_layouts_do_not_overlap(self):
        from rewardlens.exp2.layouts import load_manifest
        from scripts.exp2_run_phase5 import distill_states, load_setup
        setup = load_setup()
        splits, _ = load_manifest(EXP_DIR / "layouts/manifest.json")
        train_states = {tuple(int(v) for v in row) for row in distill_states(setup)["layouts"]}
        dev, test = ({l.as_tuple() for l in splits[s]} for s in ("dev", "test"))
        assert train_states <= {l.as_tuple() for l in splits["train"]}
        assert not train_states & dev and not train_states & test
        assert {l.as_tuple() for l in setup["layouts"]["dev"]} == dev
