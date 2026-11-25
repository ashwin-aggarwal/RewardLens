"""Experiment 2, Phase 5: checkpoint loading, layer swaps, frozen fine-tuning, weight measures and integrity."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from rewardlens.agents.dqn import QNetwork
from rewardlens.exp2.checkpoints import checkpoint_path
from rewardlens.exp2.emergence import weights_sha256
from rewardlens.exp2.evaluation import greedy_policy, rollout, summarize
from rewardlens.exp2.layouts import generate_layouts, layouts_to_array
from rewardlens.exp2.procedural_env import ProceduralEnvParams
from rewardlens.exp2.representations import sample_matched_states
from rewardlens.exp2.weight_level import (
    LAYERS, closed_form_head, conflict_masks, count_parameters, decision_diagnostics, distill, finetune, first_sustained,
    frozen_unchanged, layer_vector, load_net, meets_repair, q_preference, random_edit, resolve_step,
    scripted_shortest_path, select_checkpoint, set_trainable, swap_layers, weight_difference,
)

PARAMS = ProceduralEnvParams()
RESULTS = Path("results/exp2/phase5")
EXP_DIR = Path("experiments/02_generalizable_reward_hacking")
CRITERION = {"min_direct_goal": 0.9, "min_path_efficiency": 0.9, "max_farming": 0.05}
TRANSITION = {"onset": 70_000, "midpoint": 100_000, "completion": 160_000, "last_before_onset": 60_000}
HP = {"learning_rate": 0.00025, "gamma": 0.99, "buffer_capacity": 2000, "batch_size": 32, "learning_starts": 100,
      "train_frequency": 4, "target_update_frequency": 100, "total_timesteps": 600, "eval_interval": 300,
      "epsilon_start": 0.3, "epsilon_end": 0.05, "epsilon_fraction": 0.25}
needs_checkpoints = pytest.mark.skipif(not checkpoint_path(42, "hacking").exists(), reason="committed checkpoints missing")
needs_results = pytest.mark.skipif(not (RESULTS / "finetune.json").exists(), reason="Phase 5 results missing")


def _nets(seed_a=0, seed_b=1):
    torch.manual_seed(seed_a)
    a = QNetwork()
    torch.manual_seed(seed_b)
    b = QNetwork()
    return a.eval(), b.eval()


def _layouts(n=40, seed=3):
    return generate_layouts(n, seed)


class TestCheckpointLoading:
    def test_named_points_resolve_to_steps(self):
        assert resolve_step("start", TRANSITION, 400_000) == 0
        assert resolve_step("final", TRANSITION, 400_000) == 400_000
        assert resolve_step("completion", TRANSITION, 400_000) == 160_000
        assert resolve_step("last_before_onset", TRANSITION, 400_000) == 60_000
        assert resolve_step(50_000, TRANSITION, 400_000) == 50_000

    @needs_checkpoints
    def test_loaded_networks_match_seed_branch_and_registry(self):
        registry = json.load(open("results/exp2/phase1/checkpoints.json"))
        for seed in (42, 123, 456):
            for branch, step, stage in (("pretrain", 0, "pretrain"), ("hacking", 0, "pretrain"), ("good", 400_000, "good"),
                                        ("hacking", 400_000, "hacking")):
                net = load_net(seed, branch, step)
                from rewardlens.agents.dqn import load_checkpoint
                ref = load_checkpoint(registry["seeds"][str(seed)][stage]["path"])["q_net_state_dict"]
                assert weights_sha256(net.state_dict()) == weights_sha256(ref)
        assert weights_sha256(load_net(42, "hacking", 400_000).state_dict()) != weights_sha256(load_net(123, "hacking", 400_000).state_dict())

    @needs_checkpoints
    def test_a_file_from_another_seed_is_rejected(self, tmp_path):
        import shutil
        src = checkpoint_path(123, "hacking")
        dst = tmp_path / "seed42" / "hacking" / "checkpoints"
        dst.mkdir(parents=True)
        shutil.copy(src, dst / "checkpoint_final.pt")
        with pytest.raises(ValueError, match="seed"):
            load_net(42, "hacking", 400_000, tmp_path)

    @needs_checkpoints
    def test_intermediate_checkpoint_step_is_checked(self):
        path = Path("results/exp2/phase1/seed42/hacking/checkpoints/checkpoint_160000.pt")
        if not path.exists():
            pytest.skip("intermediate checkpoint missing")
        from rewardlens.agents.dqn import load_checkpoint
        assert weights_sha256(load_net(42, "hacking", 160_000).state_dict()) == weights_sha256(load_checkpoint(path)["q_net_state_dict"])


class TestLayerSwap:
    def test_swap_copies_exactly_the_named_layers(self):
        rec, donor = _nets()
        for layers in (["fc3"], ["fc2"], ["fc1"], ["fc2", "fc3"], ["fc1", "fc2"], ["fc1", "fc3"]):
            new = swap_layers(rec, donor, layers)
            for key, value in new.state_dict().items():
                source = donor if key.split(".")[0] in layers else rec
                assert torch.equal(value, source.state_dict()[key]), (layers, key)

    def test_swap_leaves_the_inputs_untouched_and_full_swap_is_the_donor(self):
        rec, donor = _nets()
        before = (weights_sha256(rec.state_dict()), weights_sha256(donor.state_dict()))
        full = swap_layers(rec, donor, list(LAYERS))
        assert (weights_sha256(rec.state_dict()), weights_sha256(donor.state_dict())) == before
        assert weights_sha256(full.state_dict()) == before[1]
        assert weights_sha256(swap_layers(rec, donor, []).state_dict()) == before[0]
        obs = torch.rand(50, 7)
        assert torch.equal(full(obs), donor(obs))

    def test_unknown_layer_is_rejected(self):
        rec, donor = _nets()
        with pytest.raises(ValueError):
            swap_layers(rec, donor, ["fc4"])

    def test_random_edit_matches_the_norm_and_touches_only_the_named_layers(self):
        rec, donor = _nets()
        target = swap_layers(rec, donor, ["fc2"])
        ctrl = random_edit(rec, target, ["fc2"], seed=0)
        sd, rs, ts = ctrl.state_dict(), rec.state_dict(), target.state_dict()
        for key in sd:
            if key.startswith("fc2"):
                assert float((sd[key] - rs[key]).norm()) == pytest.approx(float((ts[key] - rs[key]).norm()), rel=1e-4)
                assert not torch.equal(sd[key], ts[key])
            else:
                assert torch.equal(sd[key], rs[key])
        assert weights_sha256(random_edit(rec, target, ["fc2"], 0).state_dict()) == weights_sha256(sd)
        assert weights_sha256(random_edit(rec, target, ["fc2"], 1).state_dict()) != weights_sha256(sd)

    def test_parameter_counts(self):
        assert [count_parameters([l]) for l in LAYERS] == [1024, 16512, 516]
        assert count_parameters(LAYERS) == 18052 == sum(p.numel() for p in QNetwork().parameters())


class TestWeightDifference:
    def test_known_values(self):
        a, _ = _nets()
        b = swap_layers(a, a, [])
        sd = b.state_dict()
        sd["fc3.weight"] = sd["fc3.weight"] * 2
        sd["fc3.bias"] = sd["fc3.bias"] * 2
        b.load_state_dict(sd)
        d = weight_difference(a, b)
        assert d["fc3"]["relative"] == pytest.approx(1.0, rel=1e-6) and d["fc3"]["cosine"] == pytest.approx(1.0)
        assert d["fc3"]["l2"] == pytest.approx(np.linalg.norm(layer_vector(a, "fc3")), rel=1e-6)
        assert d["fc3"]["n_changed"] == 516 and d["fc3"]["n_parameters"] == 516
        for layer in ("fc1", "fc2"):
            assert d[layer]["relative"] == 0.0 and d[layer]["n_changed"] == 0 and d[layer]["cosine"] == pytest.approx(1.0)

    def test_layer_vector_holds_weight_and_bias(self):
        a, _ = _nets()
        assert layer_vector(a, "fc1").shape == (1024,) and layer_vector(a.state_dict(), "fc2").shape == (16512,)

    def test_difference_is_measured_against_the_reference_norm(self):
        a, b = _nets()
        d = weight_difference(a, b, ["fc2"])["fc2"]
        va, vb = layer_vector(a, "fc2"), layer_vector(b, "fc2")
        assert d["relative"] == pytest.approx(np.linalg.norm(vb - va) / np.linalg.norm(va))
        assert d["cosine"] == pytest.approx(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))


class TestFrozenFineTuning:
    def _run(self, layers, seed=5):
        start, _ = _nets()
        init = {k: v.clone() for k, v in start.state_dict().items()}
        lay = _layouts()
        run = finetune(init, layers, lay, layouts_to_array(lay[:20]), PARAMS, HP, seed, "corrected",
                       np.arange(20) < 10)
        return init, run

    @pytest.mark.parametrize("layers", [["fc3"], ["fc2"], ["fc1"]])
    def test_frozen_layers_stay_bit_identical_and_trainable_ones_move(self, layers):
        init, run = self._run(layers)
        final = run["snapshots"][-1]
        assert frozen_unchanged(init, final, layers)
        for key in init:
            assert torch.equal(init[key], final[key]) == (key.split(".")[0] not in layers), key
        assert run["n_trainable"] == count_parameters(layers)
        assert all(torch.equal(init[k], run["snapshots"][0][k]) for k in init)

    def test_full_network_changes_every_layer(self):
        init, run = self._run(list(LAYERS))
        assert all(not torch.equal(init[k], run["snapshots"][-1][k]) for k in init)
        assert not frozen_unchanged(init, run["snapshots"][-1], ["fc3"])

    def test_fine_tuning_is_reproducible(self):
        _, a = self._run(["fc2"])
        _, b = self._run(["fc2"])
        assert weights_sha256(a["snapshots"][-1]) == weights_sha256(b["snapshots"][-1])
        assert [e["dev"] for e in a["evals"]] == [e["dev"] for e in b["evals"]]
        _, c = self._run(["fc2"], seed=6)
        assert weights_sha256(c["snapshots"][-1]) != weights_sha256(a["snapshots"][-1])

    def test_evaluations_cover_selection_and_reporting_halves(self):
        _, run = self._run(["fc3"])
        assert [e["step"] for e in run["evals"]] == [0, 300, 600]
        assert all({"dev", "dev_select", "dev_report"} <= set(e) for e in run["evals"])

    def test_set_trainable_rejects_an_empty_set(self):
        with pytest.raises(ValueError):
            set_trainable(QNetwork(), [])

    def test_distillation_freezes_layers_and_is_reproducible(self):
        student, teacher = _nets()
        init = {k: v.clone() for k, v in student.state_dict().items()}
        lay = _layouts()
        obs = sample_matched_states(lay, 4, 0, PARAMS)["obs"]
        hp = {"n_updates": 60, "eval_interval_updates": 30, "batch_size": 32, "learning_rate": 0.001}
        runs = [distill(init, teacher, ["fc3"], obs, layouts_to_array(lay[:20]), PARAMS, hp, 3, np.arange(20) < 10) for _ in range(2)]
        assert frozen_unchanged(init, runs[0]["snapshots"][-1], ["fc3"])
        assert not torch.equal(init["fc3.weight"], runs[0]["snapshots"][-1]["fc3.weight"])
        assert weights_sha256(runs[0]["snapshots"][-1]) == weights_sha256(runs[1]["snapshots"][-1])
        assert runs[0]["evals"][-1]["train_loss"] < runs[0]["evals"][0]["train_loss"]

    def test_closed_form_head_changes_only_the_head_and_recovers_an_exact_fit(self):
        student, teacher = _nets()
        obs = sample_matched_states(_layouts(), 8, 0, PARAMS)["obs"]
        # A teacher that shares the student's hidden layers is exactly reachable by a new head.
        reachable = swap_layers(student, teacher, ["fc3"])
        fitted = closed_form_head(student, reachable, obs)
        for key in ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"):
            assert torch.equal(fitted.state_dict()[key], student.state_dict()[key])
        x = torch.from_numpy(obs)
        assert torch.allclose(fitted(x), reachable(x), atol=1e-3)


class TestSelectionAndCriterion:
    def test_select_checkpoint_takes_the_best_and_the_earliest_on_ties(self):
        evals = [{"dev_select": {"mean_return_corrected": v}} for v in (0.0, 5.0, 9.0, 9.0, 3.0)]
        assert select_checkpoint(evals) == 2

    def test_first_sustained(self):
        assert first_sustained([False, True, False, True, True]) == 3
        assert first_sustained([True, True]) == 0
        assert first_sustained([True, False]) is None and first_sustained([]) is None

    def test_repair_needs_efficient_goal_navigation_not_just_no_farming(self):
        base = {"direct_goal": 0.95, "path_efficiency_direct": 0.97, "farming": 0.0}
        assert meets_repair(base, CRITERION)
        assert not meets_repair({**base, "direct_goal": 0.5}, CRITERION)            # stopped farming, wanders
        assert not meets_repair({**base, "path_efficiency_direct": 0.3}, CRITERION)  # reaches the goal by accident
        assert not meets_repair({**base, "path_efficiency_direct": None}, CRITERION)
        assert not meets_repair({**base, "farming": 0.2}, CRITERION)


class TestBehaviorAndDecisions:
    def test_scripted_shortest_path_is_direct_and_efficient(self):
        s = scripted_shortest_path(_layouts(200), PARAMS)
        assert s["direct_goal"] == 1.0 and s["farming"] == 0.0 and s["other"] == 0.0
        assert s["mean_reward_tile_visits"] == 0.0 and s["path_efficiency_direct"] > 0.97

    def test_categories_partition_every_episode(self):
        net, _ = _nets()
        s = summarize(rollout(greedy_policy(net), _layouts(100), PARAMS), PARAMS.exploit_visit_threshold)
        assert s["direct_goal"] + s["farm_then_finish"] + s["farm_only"] + s["other"] == pytest.approx(1.0)
        assert s["farming"] == pytest.approx(s["farm_only"] + s["farm_then_finish"])

    def test_decision_diagnostics_of_a_network_against_itself(self):
        net, other = _nets()
        states = sample_matched_states(_layouts(), 4, 0, PARAMS)
        d = decision_diagnostics(net, net, net, states, PARAMS)
        assert d["agree_with_good"] == 1.0 and d["action_changed"] == 0.0 and d["mean_abs_q_change"] == 0.0
        d2 = decision_diagnostics(other, net, net, states, PARAMS)
        assert d2["action_changed"] == pytest.approx(1.0 - d2["agree_with_good"])

    def test_q_preference_follows_the_q_values(self):
        states = sample_matched_states(_layouts(), 4, 0, PARAMS)
        tile, goal = masks = conflict_masks(states, PARAMS)
        assert not (tile & goal).any()

        class Fixed(torch.nn.Module):
            def __init__(self, table):
                super().__init__()
                self.table = torch.from_numpy(table.astype(np.float32))

            def forward(self, obs):
                return self.table

        likes_tile = q_preference(Fixed(tile * 3.0), states, masks)
        likes_goal = q_preference(Fixed(goal * 2.0), states, masks)
        assert likes_tile["fraction_prefer_tile"] == 1.0 and likes_tile["mean_tile_minus_goal"] == pytest.approx(3.0)
        assert likes_goal["fraction_prefer_tile"] == 0.0 and likes_goal["mean_tile_minus_goal"] == pytest.approx(-2.0)
        assert 0 < likes_tile["n_conflict_states"] <= len(states["obs"])


@needs_results
class TestRecordedResults:
    def test_every_file_declares_training_and_development_only(self):
        for name in ("baselines.json", "weight_swaps.json", "finetune.json", "distill.json", "weight_trajectory.json"):
            res = json.load(open(RESULTS / name))
            assert res["splits_used"] == ["train", "dev"] and res["seeds"] == [42, 123, 456], name

    def test_frozen_layers_were_unchanged_in_every_recorded_run(self):
        for name in ("finetune.json", "distill.json"):
            res = json.load(open(RESULTS / name))
            for seed, block in res["per_seed"].items():
                for cond, run in block.items():
                    if "curve" not in run:
                        continue
                    for point in ("selected", "final"):
                        assert run[point]["frozen_layers_unchanged"], (name, seed, cond)
                        changed = {l for l in LAYERS if run[point]["weight_change"][l]["n_changed"] > 0}
                        assert changed <= set(run["trainable_layers"]), (name, seed, cond)
                    assert run["n_trainable_parameters"] == count_parameters(run["trainable_layers"])

    def test_repair_flags_follow_the_criterion(self):
        ws = json.load(open(RESULTS / "weight_swaps.json"))
        crit = ws["repair_criterion"]
        for block in ws["per_seed"].values():
            for pair in block["pairs"].values():
                for swap in pair["swaps"].values():
                    assert swap["meets_repair_criterion"] == meets_repair(swap["behavior"], crit)
                    assert swap["n_parameters"] == count_parameters(swap["layers"])

    def test_saved_weights_match_their_recorded_hashes(self):
        for name, folder in (("finetune.json", "finetuned"), ("distill.json", "distilled")):
            res = json.load(open(RESULTS / name))
            for seed, block in res["per_seed"].items():
                for cond, run in block.items():
                    path = RESULTS / folder / f"seed{seed}_{cond}.pt"
                    if "curve" not in run or not path.exists():
                        continue
                    saved = torch.load(path, weights_only=False)
                    assert weights_sha256(saved["q_net_state_dict"]) == run["selected"]["weights_sha256"]
                    assert saved["seed"] == int(seed) and saved["condition"] == cond

    @needs_checkpoints
    def test_baseline_networks_reproduce(self):
        from scripts.exp2_run_phase5 import dev_behavior, load_setup, net_at
        setup = load_setup()
        base = json.load(open(RESULTS / "baselines.json"))
        b = dev_behavior(net_at(setup, 42, ("hacking", "final")), setup)
        assert b["direct_goal"] == pytest.approx(base["per_seed"]["42"]["networks"]["hacking_final"]["dev"]["direct_goal"])
        assert base["per_seed"]["42"]["networks"]["pretrained"]["meets_repair_criterion"]
        assert not base["per_seed"]["42"]["networks"]["hacking_final"]["meets_repair_criterion"]


class TestEvaluationIntegrity:
    SOURCES = ("rewardlens/exp2/weight_level.py", "scripts/exp2_run_phase5.py", "scripts/exp2_visualize_phase5.py")

    def test_phase5_code_never_touches_test_or_final_eval_layouts(self):
        for path in self.SOURCES:
            src = Path(path).read_text()
            assert "final_eval" not in src and "load_layout_set" not in src, path
            assert not re.search(r"""\[\s*["']test["']\s*\]""", src), path

    def test_config_allows_only_training_and_development_layouts(self):
        cfg = json.load(open(EXP_DIR / "configs/phase5.json"))
        assert cfg["allowed_splits"] == ["train", "dev"]
        assert "final_eval" not in json.dumps(cfg)

    def test_setup_only_exposes_training_and_development_layouts(self):
        from rewardlens.exp2.layouts import load_manifest
        from scripts.exp2_run_phase5 import load_setup
        setup = load_setup()
        splits, _ = load_manifest(EXP_DIR / "layouts/manifest.json")
        assert set(setup["layouts"]) == {"train", "train_panel", "dev"}
        assert setup["layouts"]["train"] == splits["train"] and setup["layouts"]["dev"] == splits["dev"]
        assert not {l.as_tuple() for l in splits["test"]} & {l.as_tuple() for v in setup["layouts"].values() for l in v}
