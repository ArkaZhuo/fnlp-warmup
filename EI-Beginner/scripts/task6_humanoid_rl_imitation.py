#!/usr/bin/env python3
"""Task 6: Humanoid motion control with teleop + imitation + RL refinement.

Pipeline:
1) Generate whole-body teleoperation trajectories in PyBullet humanoid.
2) Train imitation policy (behavior cloning) to mimic teleop controller.
3) Train tabular RL residual policy on top of imitation policy.
4) Compare tracking metrics across teleop / imitation / imitation+RL.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import pybullet as p
import pybullet_data
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Task6Result:
    task_name: str
    seed: int
    demo_episodes: int
    imitation_train_samples: int
    imitation_train_mse: float
    teleop_tracking_mse: float
    imitation_tracking_mse: float
    rl_tracking_mse: float
    rl_improvement_vs_imitation: float
    rl_training_curve: list[dict[str, float]]


class ImitationPolicy(nn.Module):
    def __init__(self, in_dim: int = 10, out_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HumanoidTeleopEnv:
    def __init__(
        self,
        gui: bool = False,
        realtime: bool = False,
        fixed_base: bool = True,
        locomotion_assist: bool = False,
    ):
        self.gui = gui
        self.realtime = realtime
        self.fixed_base = fixed_base
        self.locomotion_assist = locomotion_assist
        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)

        p.loadURDF("plane.urdf")
        # pybullet_data humanoid is authored with a different up-axis convention.
        # Rotate it into the z-up world so it appears upright in the viewer.
        self.base_pos = [0.0, 0.0, 3.55]
        self.base_orn = p.getQuaternionFromEuler([math.pi / 2.0, 0.0, 0.0])
        self.humanoid = p.loadURDF(
            "humanoid/humanoid.urdf",
            self.base_pos,
            self.base_orn,
            useFixedBase=fixed_base,
        )
        for link_idx in range(-1, p.getNumJoints(self.humanoid)):
            p.changeDynamics(
                self.humanoid,
                link_idx,
                lateralFriction=1.2,
                spinningFriction=0.02,
                rollingFriction=0.02,
                linearDamping=0.04,
                angularDamping=0.04,
            )

        # Revolute joints: right_elbow=4, left_elbow=7, right_knee=10, left_knee=13
        self.joints = [4, 7, 10, 13]
        self.horizon = 120
        self.phase = 0.0
        self.step_idx = 0

        if self.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=6.2,
                cameraYaw=45.0,
                cameraPitch=-12.0,
                cameraTargetPosition=[0.0, 0.0, 3.0],
            )

    def close(self) -> None:
        p.disconnect(self.cid)

    def reset(self, phase_offset: float = 0.0) -> np.ndarray:
        p.resetBasePositionAndOrientation(self.humanoid, self.base_pos, self.base_orn)
        p.resetBaseVelocity(self.humanoid, [0, 0, 0], [0, 0, 0])
        # Neutral pose near middle of joint ranges
        init = [1.1, 1.1, -1.0, -1.0]
        for j, q in zip(self.joints, init):
            p.resetJointState(self.humanoid, j, q)

        self.phase = phase_offset
        self.step_idx = 0
        self._step_sim(40)
        return self.observe()

    def _step_sim(self, n: int) -> None:
        for _ in range(n):
            p.stepSimulation()

    def teleop_reference(self, phase: float) -> np.ndarray:
        # Whole-body periodic coordinated motion.
        r_elbow = 1.1 + 0.50 * math.sin(phase)
        l_elbow = 1.1 + 0.50 * math.sin(phase + math.pi)
        r_knee = -1.0 + 0.35 * math.sin(phase)
        l_knee = -1.0 + 0.35 * math.sin(phase + math.pi)
        return np.array([r_elbow, l_elbow, r_knee, l_knee], dtype=np.float32)

    def observe(self) -> np.ndarray:
        q = []
        qd = []
        for j in self.joints:
            st = p.getJointState(self.humanoid, j)
            q.append(st[0])
            qd.append(st[1])
        obs = np.array(q + qd + [math.sin(self.phase), math.cos(self.phase)], dtype=np.float32)
        return obs

    def base_position(self) -> np.ndarray:
        pos, _ = p.getBasePositionAndOrientation(self.humanoid)
        return np.asarray(pos, dtype=np.float32)

    def step(self, target_joints: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        for j, q in zip(self.joints, target_joints):
            p.setJointMotorControl2(
                bodyUniqueId=self.humanoid,
                jointIndex=j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=float(q),
                force=220,
                positionGain=0.15,
                velocityGain=1.0,
            )

        if self.locomotion_assist and not self.fixed_base:
            gait_push = 120.0 + 60.0 * max(0.0, math.sin(self.phase))
            p.applyExternalForce(
                self.humanoid,
                -1,
                forceObj=[gait_push, 0.0, 0.0],
                posObj=[0.0, 0.0, 0.9],
                flags=p.LINK_FRAME,
            )

        self._step_sim(4)
        if self.gui and self.realtime:
            time.sleep(1.0 / 60.0)

        self.step_idx += 1
        self.phase += 0.12
        ref = self.teleop_reference(self.phase)

        current_q = np.array([p.getJointState(self.humanoid, j)[0] for j in self.joints], dtype=np.float32)
        tracking_mse = float(np.mean((current_q - ref) ** 2))

        reward = -tracking_mse
        done = self.step_idx >= self.horizon
        obs = self.observe()

        info = {
            "tracking_mse": tracking_mse,
            "ref": ref.tolist(),
            "q": current_q.tolist(),
            "base_position": self.base_position().tolist(),
        }

        if self.gui and not self.fixed_base:
            base = info["base_position"]
            p.resetDebugVisualizerCamera(
                cameraDistance=6.2,
                cameraYaw=45.0,
                cameraPitch=-12.0,
                cameraTargetPosition=[base[0], base[1], max(2.2, base[2])],
            )
        return obs, reward, done, info


def collect_demos(env: HumanoidTeleopEnv, episodes: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    obs_list = []
    act_list = []

    for _ in range(episodes):
        phase_offset = float(rng.uniform(0.0, 2 * math.pi))
        obs = env.reset(phase_offset=phase_offset)

        for _ in range(env.horizon):
            ref_action = env.teleop_reference(env.phase)
            obs_list.append(obs)
            act_list.append(ref_action)

            obs, _, done, _ = env.step(ref_action)
            if done:
                break

    return np.asarray(obs_list, dtype=np.float32), np.asarray(act_list, dtype=np.float32)


def train_imitation(obs: np.ndarray, act: np.ndarray, seed: int) -> tuple[ImitationPolicy, float]:
    torch.manual_seed(seed)
    model = ImitationPolicy(in_dim=obs.shape[1], out_dim=act.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    x = torch.from_numpy(obs)
    y = torch.from_numpy(act)

    n = x.shape[0]
    bs = 256
    for _ in range(45):
        perm = torch.randperm(n)
        x = x[perm]
        y = y[perm]

        for i in range(0, n, bs):
            xb = x[i : i + bs]
            yb = y[i : i + bs]
            pred = model(xb)
            loss = F.mse_loss(pred, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        train_mse = float(F.mse_loss(model(torch.from_numpy(obs)), torch.from_numpy(act)).item())
    return model, train_mse


def run_rollout_teleop(env: HumanoidTeleopEnv, episodes: int, rng: np.random.Generator) -> float:
    mse_all = []
    for _ in range(episodes):
        env.reset(phase_offset=float(rng.uniform(0, 2 * math.pi)))
        for _ in range(env.horizon):
            a = env.teleop_reference(env.phase)
            _, _, done, info = env.step(a)
            mse_all.append(info["tracking_mse"])
            if done:
                break
    return float(np.mean(mse_all)) if mse_all else 0.0


def run_rollout_imitation(env: HumanoidTeleopEnv, model: ImitationPolicy, episodes: int, rng: np.random.Generator) -> float:
    mse_all = []
    for _ in range(episodes):
        obs = env.reset(phase_offset=float(rng.uniform(0, 2 * math.pi)))
        for _ in range(env.horizon):
            with torch.no_grad():
                a = model(torch.tensor(obs).unsqueeze(0)).squeeze(0).numpy()
            obs, _, done, info = env.step(a)
            mse_all.append(info["tracking_mse"])
            if done:
                break
    return float(np.mean(mse_all)) if mse_all else 0.0


def imitation_action(model: ImitationPolicy, obs: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.tensor(obs).unsqueeze(0)).squeeze(0).numpy()


def discretize_state(phase: float, err: float) -> tuple[int, int]:
    phase_wrapped = phase % (2 * math.pi)
    phase_bin = int((phase_wrapped / (2 * math.pi)) * 12)
    phase_bin = min(11, max(0, phase_bin))

    err_bin = int(np.clip(np.floor(err / 0.02), 0, 7))
    return phase_bin, err_bin


def apply_residual(action: np.ndarray, residual_act: int) -> np.ndarray:
    out = action.copy()
    if residual_act == 0:
        out *= 0.97
    elif residual_act == 1:
        out *= 1.00
    elif residual_act == 2:
        out *= 1.03
    elif residual_act == 3:
        out[2:] -= 0.015
    elif residual_act == 4:
        out[2:] += 0.015
    return out


def train_rl_residual(
    env: HumanoidTeleopEnv,
    model: ImitationPolicy,
    rng: np.random.Generator,
    episodes: int = 360,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    q = np.zeros((12, 8, 5), dtype=np.float64)

    gamma = 0.95
    epsilon = 1.0
    eps_min = 0.08
    eps_decay = 0.995
    training_curve: list[dict[str, float]] = []

    def eval_q_snapshot(eval_episodes: int = 8) -> float:
        eval_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        return run_rollout_rl(env, model, q, episodes=eval_episodes, rng=eval_rng)

    for ep in range(episodes):
        obs = env.reset(phase_offset=float(rng.uniform(0, 2 * math.pi)))
        prev_err = 0.05

        for _ in range(env.horizon):
            s = discretize_state(env.phase, prev_err)

            if rng.random() < epsilon:
                ra = int(rng.integers(0, 5))
            else:
                ra = int(np.argmax(q[s]))

            with torch.no_grad():
                base = model(torch.tensor(obs).unsqueeze(0)).squeeze(0).numpy()

            act = apply_residual(base, ra)
            obs, reward, done, info = env.step(act)
            err = float(info["tracking_mse"])
            reward += 0.5 * (prev_err - err)

            ns = discretize_state(env.phase, err)
            alpha = max(0.08, 0.30 * (1.0 - ep / episodes))
            td_target = reward + (0.0 if done else gamma * np.max(q[ns]))
            q[s + (ra,)] += alpha * (td_target - q[s + (ra,)])

            prev_err = err
            if done:
                break

        epsilon = max(eps_min, epsilon * eps_decay)

        if ep == 0 or (ep + 1) % 20 == 0 or ep == episodes - 1:
            training_curve.append(
                {
                    "episode": float(ep + 1),
                    "tracking_mse": float(eval_q_snapshot()),
                    "epsilon": float(epsilon),
                }
            )

    return q, training_curve


def run_rollout_rl(
    env: HumanoidTeleopEnv,
    model: ImitationPolicy,
    q: np.ndarray,
    episodes: int,
    rng: np.random.Generator,
) -> float:
    mse_all = []
    for _ in range(episodes):
        obs = env.reset(phase_offset=float(rng.uniform(0, 2 * math.pi)))
        prev_err = 0.05

        for _ in range(env.horizon):
            s = discretize_state(env.phase, prev_err)
            ra = int(np.argmax(q[s]))
            with torch.no_grad():
                base = model(torch.tensor(obs).unsqueeze(0)).squeeze(0).numpy()

            act = apply_residual(base, ra)
            obs, _, done, info = env.step(act)
            err = float(info["tracking_mse"])
            mse_all.append(err)
            prev_err = err
            if done:
                break

    return float(np.mean(mse_all)) if mse_all else 0.0


def run_visual_demo(
    env: HumanoidTeleopEnv,
    policy_name: str,
    rng: np.random.Generator,
    model: ImitationPolicy,
    q: np.ndarray | None,
    episodes: int,
) -> float:
    mse_all = []

    for ep in range(episodes):
        obs = env.reset(phase_offset=float(rng.uniform(0, 2 * math.pi)))
        prev_err = 0.05
        start_base = env.base_position().tolist()
        last_base = start_base

        for _ in range(env.horizon):
            if policy_name == "teleop":
                action = env.teleop_reference(env.phase)
            elif policy_name == "imitation":
                action = imitation_action(model, obs)
            else:
                if q is None:
                    raise ValueError("RL demo requires a trained Q table")
                s = discretize_state(env.phase, prev_err)
                residual_act = int(np.argmax(q[s]))
                base = imitation_action(model, obs)
                action = apply_residual(base, residual_act)

            obs, _, done, info = env.step(action)
            prev_err = float(info["tracking_mse"])
            last_base = info["base_position"]
            mse_all.append(prev_err)
            if done:
                break

        ep_mse = float(np.mean(mse_all[-env.horizon:])) if mse_all else 0.0
        base_travel = float(last_base[0] - start_base[0])
        print(
            json.dumps(
                {
                    "demo_episode": ep + 1,
                    "policy": policy_name,
                    "tracking_mse": ep_mse,
                    "base_x_start": start_base[0],
                    "base_x_end": last_base[0],
                    "base_x_travel": base_travel,
                },
                ensure_ascii=False,
            )
        )

    return float(np.mean(mse_all)) if mse_all else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Task6 humanoid teleop + imitation + RL")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo_episodes", type=int, default=70)
    parser.add_argument("--rl_episodes", type=int, default=220)
    parser.add_argument("--gui", action="store_true", help="Open a PyBullet GUI demo after training")
    parser.add_argument("--realtime", action="store_true", help="Slow the GUI demo to a watchable speed")
    parser.add_argument(
        "--free_base_demo",
        action="store_true",
        help="Release the humanoid base in the GUI demo so its global position can change",
    )
    parser.add_argument(
        "--demo_policy",
        choices=["teleop", "imitation", "rl"],
        default="rl",
        help="Which policy to visualize when --gui is enabled",
    )
    parser.add_argument(
        "--gui_demo_episodes",
        type=int,
        default=8,
        help="How many episodes to replay in the GUI demo",
    )
    parser.add_argument("--output", default="results/task6_humanoid_result.json")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = HumanoidTeleopEnv(gui=False)
    try:
        demo_obs, demo_act = collect_demos(env, args.demo_episodes, rng)
        model, train_mse = train_imitation(demo_obs, demo_act, seed=args.seed)

        teleop_mse = run_rollout_teleop(env, episodes=25, rng=rng)
        imitation_mse = run_rollout_imitation(env, model, episodes=25, rng=rng)

        q, rl_training_curve = train_rl_residual(env, model, rng, episodes=args.rl_episodes)
        rl_mse = run_rollout_rl(env, model, q, episodes=25, rng=rng)
    finally:
        env.close()

    result = Task6Result(
        task_name="Task6-Humanoid-RL-Control-Teleop-Imitation",
        seed=args.seed,
        demo_episodes=args.demo_episodes,
        imitation_train_samples=int(demo_obs.shape[0]),
        imitation_train_mse=float(train_mse),
        teleop_tracking_mse=float(teleop_mse),
        imitation_tracking_mse=float(imitation_mse),
        rl_tracking_mse=float(rl_mse),
        rl_improvement_vs_imitation=float(imitation_mse - rl_mse),
        rl_training_curve=rl_training_curve,
    )

    payload = asdict(result)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.gui:
        demo_rng = np.random.default_rng(args.seed + 12345)
        demo_env = HumanoidTeleopEnv(
            gui=True,
            realtime=args.realtime,
            fixed_base=not args.free_base_demo,
            locomotion_assist=args.free_base_demo,
        )
        try:
            demo_mse = run_visual_demo(
                demo_env,
                policy_name=args.demo_policy,
                rng=demo_rng,
                model=model,
                q=q,
                episodes=args.gui_demo_episodes,
            )
            print(json.dumps({"demo_policy": args.demo_policy, "gui_demo_tracking_mse": demo_mse}, ensure_ascii=False))
        finally:
            demo_env.close()


if __name__ == "__main__":
    main()
