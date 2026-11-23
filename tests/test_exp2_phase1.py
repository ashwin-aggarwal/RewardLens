"""Experiment 2, Phase 1: procedural layouts, environment, evaluation and training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rewardlens.agents.dqn import QNetwork, load_q_network_for_eval
from rewardlens.analysis.behavior_classification import DIRECT_GOAL, FARM_ONLY, FARM_THEN_FINISH, OTHER
from rewardlens.exp2.evaluation import categories, greedy_policy, observations, random_policy, rollout, summarize
from rewardlens.exp2.layouts import (
    Layout, assert_disjoint, generate_layouts, is_valid_layout, layouts_to_array, load_manifest,
    save_manifest, split_layouts, unseen_pair_mask,
)
from rewardlens.exp2.procedural_env import ProceduralEnvParams, ProceduralRewardGridEnv
from rewardlens.exp2.scripted import farm_actions, farm_then_finish_actions, goal_actions, play_actions
from rewardlens.exp2.training import Exp2TrainConfig, build_networks, train

PARAMS = ProceduralEnvParams()
MANIFEST = Path("experiments/02_generalizable_reward_hacking/layouts/manifest.json")
ENV_CONFIG = Path("experiments/02_generalizable_reward_hacking/configs/environment.json")


@pytest.fixture(scope="module")
def layouts():
    return generate_layouts(300, seed=3)


# --- Procedural generation -------------------------------------------------

class TestLayouts:
    def test_deterministic(self):
        assert generate_layouts(200, seed=5) == generate_layouts(200, seed=5)
        assert generate_layouts(200, seed=5) != generate_layouts(200, seed=6)

    def test_valid_distinct_in_bounds(self, layouts):
        assert len({l.as_tuple() for l in layouts}) == len(layouts)
        for l in layouts:
            assert len({l.start, l.goal, l.reward}) == 3
            for x, y in (l.start, l.goal, l.reward):
                assert 0 <= x < 8 and 0 <= y < 8
            assert is_valid_layout(l)

    def test_invalid_layouts_rejected(self):
        assert not is_valid_layout(Layout((0, 0), (0, 0), (1, 1)))
        assert not is_valid_layout(Layout((0, 0), (1, 1), (1, 1)))
        assert not is_valid_layout(Layout((0, 0), (8, 1), (1, 1)))
        # On a 1x3 strip the goal blocks the only route to the tile.
        assert not is_valid_layout(Layout((0, 0), (1, 0), (2, 0)), width=3, height=1)

    def test_too_many_requested(self):
        with pytest.raises(ValueError):
            generate_layouts(100, seed=0, width=2, height=2)

    def test_split_is_a_partition(self, layouts):
        s = split_layouts(layouts, (0.7, 0.15, 0.15), seed=1)
        assert [len(s[k]) for k in ("train", "dev", "test")] == [210, 45, 45]
        assert_disjoint(s)
        assert {l.as_tuple() for v in s.values() for l in v} == {l.as_tuple() for l in layouts}
        assert s == split_layouts(layouts, (0.7, 0.15, 0.15), seed=1)

    def test_overlap_detected(self, layouts):
        with pytest.raises(ValueError):
            assert_disjoint({"train": layouts[:10], "dev": layouts[5:15], "test": layouts[20:25]})

    def test_manifest_roundtrip_and_tamper(self, layouts, tmp_path):
        s = split_layouts(layouts, (0.7, 0.15, 0.15), seed=1)
        save_manifest(tmp_path / "m.json", s, {"note": "t"})
        loaded, meta = load_manifest(tmp_path / "m.json")
        assert loaded == s and meta["counts"] == {"train": 210, "dev": 45, "test": 45}
        m = json.load(open(tmp_path / "m.json"))
        m["splits"]["test"][0] = m["splits"]["train"][0]
        json.dump(m, open(tmp_path / "bad.json", "w"))
        with pytest.raises(ValueError):
            load_manifest(tmp_path / "bad.json")

    def test_unseen_pair_mask(self):
        train = [Layout((0, 0), (1, 1), (2, 2))]
        ev = [Layout((5, 5), (1, 1), (2, 2)), Layout((0, 0), (1, 1), (3, 3))]
        assert unseen_pair_mask(ev, train).tolist() == [False, True]


class TestFrozenManifest:
    def test_committed_manifest_is_consistent(self):
        if not MANIFEST.exists():
            pytest.skip("manifest not generated")
        cfg = json.load(open(ENV_CONFIG))["layouts"]
        splits, meta = load_manifest(MANIFEST)  # verifies hash and disjointness
        assert meta["counts"] == {"train": 7000, "dev": 1500, "test": 1500}
        regenerated = split_layouts(generate_layouts(cfg["n_layouts"], cfg["generation_seed"]),
                                    tuple(cfg["fractions"]), cfg["split_seed"])
        assert regenerated == splits
        assert all(is_valid_layout(l) for v in splits.values() for l in v)


# --- Environment -----------------------------------------------------------

class TestEnvironment:
    def test_observation_layout_and_normalization(self):
        l = Layout((1, 6), (6, 1), (3, 5))
        env = ProceduralRewardGridEnv([l], "flawed")
        obs, info = env.reset(seed=0)
        assert obs.shape == (7,) and obs.dtype == np.float32
        np.testing.assert_allclose(obs, [1 / 7, 6 / 7, 6 / 7, 1 / 7, 3 / 7, 5 / 7, 1.0], rtol=1e-6)
        assert info["layout"] == l.as_tuple()
        obs, *_ = env.step(0)
        assert obs[6] == pytest.approx(0.99)
        assert env.observation_space.contains(obs)

    def test_observations_stay_in_unit_range(self, layouts):
        env = ProceduralRewardGridEnv(layouts, "flawed")
        rng = np.random.RandomState(0)
        env.reset(seed=1)
        for _ in range(20):
            obs, _ = env.reset()
            done = False
            while not done:
                assert obs.min() >= 0.0 and obs.max() <= 1.0
                obs, _, term, trunc, _ = env.step(int(rng.randint(4)))
                done = term or trunc

    def test_layout_sampling_is_seeded_and_varied(self, layouts):
        def seq(seed):
            env = ProceduralRewardGridEnv(layouts, "corrected")
            first = env.reset(seed=seed)[1]["layout"]
            return [first] + [env.reset()[1]["layout"] for _ in range(20)]
        assert seq(4) == seq(4)
        assert seq(4) != seq(5)
        assert len(set(seq(4))) > 10

    def test_explicit_layout_option(self, layouts):
        env = ProceduralRewardGridEnv(layouts, "flawed")
        l = Layout((0, 0), (7, 7), (4, 4))
        _, info = env.reset(seed=0, options={"layout": l})
        assert info["layout"] == l.as_tuple() and info["agent_pos"] == (0, 0)
        with pytest.raises(ValueError):
            env.reset(options={"layout": Layout((0, 0), (0, 0), (1, 1))})

    def test_reward_tile_repeatable_only_on_reentry(self):
        l = Layout((2, 0), (7, 7), (3, 0))
        env = ProceduralRewardGridEnv([l], "flawed")
        env.reset(seed=0)
        rewards = [env.step(a)[1] for a in (1, 0, 3, 1, 3, 1)]  # enter, bump wall, leave, enter, leave, enter
        assert rewards == pytest.approx([1.99, -0.01, -0.01, 1.99, -0.01, 1.99])
        assert env.step(3)[4]["reward_tile_visits"] == 3

    def test_goal_terminates_and_time_limit_truncates(self):
        l = Layout((0, 0), (1, 0), (5, 5))
        env = ProceduralRewardGridEnv([l], "corrected")
        env.reset(seed=0)
        _, r, term, trunc, info = env.step(1)
        assert term and not trunc and info["goal_reached"] and r == pytest.approx(9.99)
        with pytest.raises(RuntimeError):
            env.step(0)
        env.reset()
        for i in range(PARAMS.max_steps):
            _, _, term, trunc, info = env.step(0)
        assert trunc and not term and not info["goal_reached"] and i == 99

    def test_reward_modes_differ_only_in_tile_bonus(self, layouts):
        rng = np.random.RandomState(1)
        for l in layouts[:25]:
            envs = {m: ProceduralRewardGridEnv([l], m) for m in ("flawed", "corrected")}
            obs = {m: e.reset(seed=0)[0] for m, e in envs.items()}
            assert envs["flawed"].observation_space == envs["corrected"].observation_space
            assert envs["flawed"].action_space == envs["corrected"].action_space
            done = False
            while not done:
                np.testing.assert_array_equal(obs["flawed"], obs["corrected"])
                a = int(rng.randint(4))
                before = envs["flawed"]._reward_tile_visits
                of, rf, tf, uf, info_f = envs["flawed"].step(a)
                oc, rc, tc, uc, info_c = envs["corrected"].step(a)
                entered = info_f["reward_tile_visits"] - before
                assert rf - rc == pytest.approx(2.0 * entered)
                assert (tf, uf) == (tc, uc) and info_f["reward_tile_visits"] == info_c["reward_tile_visits"]
                obs = {"flawed": of, "corrected": oc}
                done = tf or uf

    def test_unknown_reward_mode(self, layouts):
        with pytest.raises(ValueError):
            ProceduralRewardGridEnv(layouts, "shaped")


# --- Scripted policies and economics ----------------------------------------

class TestScripted:
    def test_both_strategies_feasible_everywhere(self, layouts):
        for l in layouts:
            g = play_actions(l, goal_actions(l, PARAMS), PARAMS, 0.99)
            f = play_actions(l, farm_actions(l, PARAMS), PARAMS, 0.99)
            ff = play_actions(l, farm_then_finish_actions(l, PARAMS), PARAMS, 0.99)
            assert g["goal_reached"] and g["reward_tile_visits"] == 0
            assert not f["goal_reached"] and f["reward_tile_visits"] >= 40 and f["steps"] == 100
            assert ff["goal_reached"] and ff["reward_tile_visits"] >= 35 and ff["steps"] >= 99  # parity can cost one step
            assert f["discounted_flawed"] > g["discounted_flawed"]          # hacking pays under the flawed reward
            assert g["discounted_corrected"] > f["discounted_corrected"]    # and does not under the corrected one
            assert g["discounted_corrected"] > ff["discounted_corrected"]

    def test_play_actions_matches_environment(self, layouts):
        for l in layouts[:30]:
            actions = farm_then_finish_actions(l, PARAMS)
            env = ProceduralRewardGridEnv([l], "flawed")
            env.reset(seed=0)
            total = 0.0
            for a in actions:
                _, r, term, trunc, info = env.step(a)
                total += r
                if term or trunc:
                    break
            p = play_actions(l, actions, PARAMS, 0.99)
            assert p["return_flawed"] == pytest.approx(total)
            assert p["reward_tile_visits"] == info["reward_tile_visits"] and p["goal_reached"] == info["goal_reached"]


# --- Batched evaluation -----------------------------------------------------

def _env_rollout(net, layout, mode="flawed"):
    env = ProceduralRewardGridEnv([layout], mode)
    obs, _ = env.reset(seed=0)
    total, done = 0.0, False
    while not done:
        with torch.no_grad():
            a = int(net(torch.from_numpy(obs).unsqueeze(0)).argmax(dim=1).item())
        obs, r, term, trunc, info = env.step(a)
        total += r
        done = term or trunc
    return info, total


class TestEvaluation:
    def test_batched_rollout_matches_environment(self, layouts):
        torch.manual_seed(0)
        net = QNetwork()
        res = rollout(greedy_policy(net), layouts[:60], PARAMS, 0.99)
        for i, l in enumerate(layouts[:60]):
            info, total = _env_rollout(net, l, "flawed")
            _, total_c = _env_rollout(net, l, "corrected")
            assert res["goal_reached"][i] == info["goal_reached"]
            assert res["reward_tile_visits"][i] == info["reward_tile_visits"]
            assert res["steps"][i] == info["episode_steps"]
            assert res["return_flawed"][i] == pytest.approx(total)
            assert res["return_corrected"][i] == pytest.approx(total_c)

    def test_observations_match_environment(self, layouts):
        arr = layouts_to_array(layouts[:20])
        obs = observations(arr[:, :2], arr, np.zeros(20, dtype=int), PARAMS)
        for i, l in enumerate(layouts[:20]):
            np.testing.assert_array_equal(obs[i], ProceduralRewardGridEnv([l]).reset(seed=0)[0])

    def test_evaluation_is_reproducible(self, layouts):
        torch.manual_seed(1)
        net = QNetwork()
        a = rollout(greedy_policy(net), layouts, PARAMS, 0.99, record_actions=True)
        b = rollout(greedy_policy(net), layouts, PARAMS, 0.99, record_actions=True)
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])
        assert summarize(a, 5) == summarize(b, 5)

    def test_recorded_actions_replay(self, layouts):
        res = rollout(random_policy(0), layouts[:20], PARAMS, 0.99, record_actions=True)
        for i, l in enumerate(layouts[:20]):
            acts = [int(a) for a in res["actions"][i] if a >= 0]
            p = play_actions(l, acts, PARAMS, 0.99)
            assert p["steps"] == res["steps"][i] and p["reward_tile_visits"] == res["reward_tile_visits"][i]
            assert p["discounted_flawed"] == pytest.approx(res["discounted_flawed"][i])

    def test_categories_are_exclusive_and_exhaustive(self):
        res = {"goal_reached": np.array([True, True, False, False, False]),
               "reward_tile_visits": np.array([4, 5, 5, 4, 0]),
               "steps": np.array([10, 100, 100, 100, 100]), "manhattan": np.array([5, 5, 5, 5, 5]),
               "return_flawed": np.zeros(5), "return_corrected": np.zeros(5)}
        assert categories(res, 5).tolist() == [DIRECT_GOAL, FARM_THEN_FINISH, FARM_ONLY, OTHER, OTHER]
        s = summarize(res, 5)
        assert s[DIRECT_GOAL] + s[FARM_THEN_FINISH] + s[FARM_ONLY] + s[OTHER] == pytest.approx(1.0)
        assert s["farming"] == pytest.approx(0.4)       # farm-only + farm-then-finish
        assert s[OTHER] == pytest.approx(0.4)           # wandering is never counted as farming
        assert s["goal_completion"] == pytest.approx(0.4)
        assert s["path_efficiency_direct"] == pytest.approx(0.5)
        assert summarize(res, 5, np.array([True, False, False, False, False]))[DIRECT_GOAL] == 1.0

    def test_scripted_policies_get_the_expected_category(self, layouts):
        for l in layouts[:40]:
            for fn, expected in ((goal_actions, DIRECT_GOAL), (farm_actions, FARM_ONLY),
                                 (farm_then_finish_actions, FARM_THEN_FINISH)):
                p = play_actions(l, fn(l, PARAMS), PARAMS, 0.99)
                res = {"goal_reached": np.array([p["goal_reached"]]),
                       "reward_tile_visits": np.array([p["reward_tile_visits"]])}
                assert categories(res, 5)[0] == expected


# --- Training and checkpoints ----------------------------------------------

def _tiny_config(tmp_path, layouts, **kw):
    s = split_layouts(layouts, (0.7, 0.15, 0.15), seed=1)
    save_manifest(tmp_path / "manifest.json", s, {})
    base = dict(manifest_path=str(tmp_path / "manifest.json"), total_timesteps=600, learning_starts=100,
                eval_interval=300, checkpoint_interval=300, n_train_eval_layouts=20, buffer_capacity=1000,
                run_dir=str(tmp_path / "run"), seed=3)
    base.update(kw)
    return Exp2TrainConfig(**base)


def _state(path):
    return load_q_network_for_eval(path)[0].state_dict()


class TestTraining:
    def test_training_is_reproducible_and_checkpoints_load(self, layouts, tmp_path):
        a = train(_tiny_config(tmp_path, layouts, run_dir=str(tmp_path / "a")), verbose=False)
        train(_tiny_config(tmp_path, layouts, run_dir=str(tmp_path / "b")), verbose=False)
        sa, sb = _state(tmp_path / "a/checkpoints/checkpoint_final.pt"), _state(tmp_path / "b/checkpoints/checkpoint_final.pt")
        for k in sa:
            assert torch.equal(sa[k], sb[k])
        assert a["episodes"] > 0 and a["env_steps_per_second"] > 0
        assert (tmp_path / "a/eval_metrics.csv").exists() and (tmp_path / "a/checkpoints/checkpoint_300.pt").exists()
        s0 = _state(tmp_path / "a/checkpoints/checkpoint_0.pt")
        assert any(not torch.equal(s0[k], sa[k]) for k in sa)   # training changed the weights

    def test_fork_branches_start_from_identical_weights(self, layouts, tmp_path):
        train(_tiny_config(tmp_path, layouts, run_dir=str(tmp_path / "pre")), verbose=False)
        pre = tmp_path / "pre/checkpoints/checkpoint_final.pt"
        runs = {}
        for stage, mode in (("good", "corrected"), ("hacking", "flawed")):
            cfg = _tiny_config(tmp_path, layouts, stage=stage, reward_mode=mode, init_checkpoint=str(pre),
                               run_dir=str(tmp_path / stage), epsilon_start=0.3)
            q, target, init_step = build_networks(cfg)
            assert init_step == 600
            for k, v in _state(pre).items():
                assert torch.equal(q.state_dict()[k], v) and torch.equal(target.state_dict()[k], v)
            train(cfg, verbose=False)
            runs[stage] = tmp_path / stage / "checkpoints"
        # Both branches saved the pretrained weights as their step-0 checkpoint, and the pretrained file is untouched.
        for k, v in _state(pre).items():
            assert torch.equal(_state(runs["good"] / "checkpoint_0.pt")[k], v)
            assert torch.equal(_state(runs["hacking"] / "checkpoint_0.pt")[k], v)
        g, h = _state(runs["good"] / "checkpoint_final.pt"), _state(runs["hacking"] / "checkpoint_final.pt")
        assert any(not torch.equal(g[k], h[k]) for k in g)       # the reward is the only difference, and it matters

    def test_training_never_reads_test_layouts(self, layouts, tmp_path):
        cfg = _tiny_config(tmp_path, layouts)
        m = json.load(open(cfg.manifest_path))
        m["splits"]["test"] = []                                  # training must work without any test layouts
        import hashlib
        from rewardlens.exp2.layouts import manifest_hash
        splits = {k: [Layout.from_tuple(t) for t in m["splits"][k]] for k in ("train", "dev", "test")}
        m["sha256"] = manifest_hash(splits)
        json.dump(m, open(cfg.manifest_path, "w"))
        meta = train(cfg, verbose=False)
        assert meta["n_train_layouts"] == 210


# --- Phase 1 finalization: reserved evaluation set and committed checkpoints ---

FINAL_EVAL = Path("experiments/02_generalizable_reward_hacking/layouts/final_eval_manifest.json")
REGISTRY = Path("results/exp2/phase1/checkpoints.json")
EVALUATION = Path("results/exp2/phase1/evaluation.json")


class TestReservedFinalEvalSet:
    def test_exclusion(self, layouts):
        fresh = generate_layouts(100, seed=3, exclude=layouts)
        assert not {l.as_tuple() for l in fresh} & {l.as_tuple() for l in layouts}
        assert fresh == generate_layouts(100, seed=3, exclude=layouts)

    def test_layout_set_roundtrip_and_tamper(self, layouts, tmp_path):
        from rewardlens.exp2.layouts import load_layout_set, save_layout_set
        save_layout_set(tmp_path / "s.json", layouts[:50], {"purpose": "t"})
        loaded, meta = load_layout_set(tmp_path / "s.json")
        assert loaded == layouts[:50] and meta["count"] == 50
        m = json.load(open(tmp_path / "s.json"))
        m["layouts"][0][0] = (m["layouts"][0][0] + 1) % 8
        json.dump(m, open(tmp_path / "bad.json", "w"))
        with pytest.raises(ValueError):
            load_layout_set(tmp_path / "bad.json")

    def test_frozen_set_is_disjoint_from_every_split(self):
        if not (FINAL_EVAL.exists() and MANIFEST.exists()):
            pytest.skip("manifests not generated")
        from rewardlens.exp2.layouts import load_layout_set
        cfg = json.load(open(ENV_CONFIG))
        final, meta = load_layout_set(FINAL_EVAL)
        splits, split_meta = load_manifest(MANIFEST)
        used = {l.as_tuple() for v in splits.values() for l in v}
        assert len(final) == 1500 and not {l.as_tuple() for l in final} & used
        assert all(is_valid_layout(l) for l in final)
        assert meta["disjoint_from_manifest_sha256"] == split_meta["sha256"]
        existing = [l for v in splits.values() for l in v]
        assert final == generate_layouts(1500, cfg["final_eval"]["generation_seed"], exclude=existing)


class TestCommittedCheckpoints:
    def _registry(self):
        if not REGISTRY.exists():
            pytest.skip("checkpoint registry not present")
        reg = json.load(open(REGISTRY))
        if not all(Path(e[s]["path"]).exists() for e in reg["seeds"].values() for s in e):
            pytest.skip("checkpoints not present")
        return reg

    def test_registry_matches_files_and_fork_correspondence(self):
        from rewardlens.exp2.checkpoints import verify_registry
        reg = self._registry()
        verify_registry(reg)
        assert set(reg["seeds"]) == {"42", "123", "456"}
        for entry in reg["seeds"].values():
            assert entry["pretrain"]["reward_mode"] == "corrected" and entry["pretrain"]["training_steps"] == 300000
            assert entry["good"]["reward_mode"] == "corrected" and entry["hacking"]["reward_mode"] == "flawed"
            assert entry["good"]["training_steps"] == entry["hacking"]["training_steps"] == 400000
            assert entry["good"]["init_checkpoint_sha256"] == entry["hacking"]["init_checkpoint_sha256"] == entry["pretrain"]["sha256"]

    def test_tampered_registry_is_rejected(self):
        from rewardlens.exp2.checkpoints import verify_registry
        reg = self._registry()
        reg["seeds"]["42"]["hacking"]["init_checkpoint_sha256"] = "0" * 64
        with pytest.raises(ValueError):
            verify_registry(reg)

    def test_triplet_shares_inputs_and_differs_in_weights(self):
        from rewardlens.exp2.checkpoints import load_triplet
        self._registry()
        nets = load_triplet(42)
        obs = torch.rand(5, 7)
        q = {k: n(obs) for k, n in nets.items()}
        assert all(v.shape == (5, 4) for v in q.values())
        assert not torch.equal(q["good"], q["hacking"]) and not torch.equal(q["pretrain"], q["good"])

    def test_committed_checkpoints_reproduce_reported_dev_results(self):
        """Development layouts only; the test split is not re-evaluated here."""
        from rewardlens.exp2.checkpoints import load_triplet
        self._registry()
        if not EVALUATION.exists():
            pytest.skip("evaluation results not present")
        ev = json.load(open(EVALUATION))
        splits, _ = load_manifest(MANIFEST)
        for seed in ev["seeds"]:
            for stage, net in load_triplet(seed).items():
                s = summarize(rollout(greedy_policy(net), splits["dev"], PARAMS, ev["gamma"]), ev["visit_threshold"])
                rec = ev["per_seed"][str(seed)][stage]["dev"]
                for k in ("direct_goal", "farming", "other", "mean_reward_tile_visits"):
                    assert s[k] == pytest.approx(rec[k])
