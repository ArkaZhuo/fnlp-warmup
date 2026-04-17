#!/usr/bin/env python3
"""Task 2 extension: RL pick-and-lift in PyBullet using tabular Q-learning.

A Franka Panda arm is controlled by discrete motion primitives. The policy is
trained to grasp and lift a cube, and reports success rate after evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass

import numpy as np
import pybullet as p
import pybullet_data

ARM_JOINTS = list(range(7))
FINGER_JOINTS = [9, 10]
EE_LINK_INDEX = 11
SIM_TIMESTEP = 1.0 / 240.0


@dataclass
class PyBulletRLResult:
    task_name: str
    train_episodes: int
    eval_episodes: int
    max_steps_per_episode: int
    training_success_rate_last_200: float
    evaluation_success_rate: float
    evaluation_avg_return: float
    training_curve_window: int
    training_curve: list[dict[str, float]]


class PandaPrimitivePickEnv:
    def __init__(self, gui: bool = False):
        self.client_id = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(SIM_TIMESTEP)

        p.loadURDF("plane.urdf")
        p.loadURDF("table/table.urdf", basePosition=[0.5, 0.0, -0.65], useFixedBase=True)

        self.robot_id = p.loadURDF("franka_panda/panda.urdf", basePosition=[0.0, 0.0, 0.0], useFixedBase=True)
        self.cube_size = 0.04
        self.cube_id = p.loadURDF("cube_small.urdf", basePosition=[0.58, 0.0, self.cube_size / 2.0])

        self.ee_target = np.array([0.58, 0.0, 0.20], dtype=np.float64)
        self.attachment_constraint = None
        self.step_count = 0
        self.max_steps = 24
        self.milestones = {
            "aligned": False,
            "low": False,
            "grabbed": False,
        }

    def close(self) -> None:
        p.disconnect(self.client_id)

    def _step_sim(self, n: int) -> None:
        for _ in range(n):
            p.stepSimulation()

    def _set_gripper(self, width: float, settle_steps: int = 45) -> None:
        finger_target = float(np.clip(width / 2.0, 0.0, 0.04))
        for _ in range(settle_steps):
            for j in FINGER_JOINTS:
                p.setJointMotorControl2(
                    self.robot_id,
                    j,
                    p.POSITION_CONTROL,
                    targetPosition=finger_target,
                    force=40,
                )
            p.stepSimulation()

    def _move_ee_to_target(self, steps: int = 16) -> None:
        target_orn = p.getQuaternionFromEuler([math.pi, 0.0, 0.0])
        joint_targets = p.calculateInverseKinematics(
            self.robot_id,
            EE_LINK_INDEX,
            targetPosition=self.ee_target.tolist(),
            targetOrientation=target_orn,
            maxNumIterations=180,
            residualThreshold=1e-4,
        )

        for _ in range(steps):
            for j_local, joint_id in enumerate(ARM_JOINTS):
                p.setJointMotorControl2(
                    self.robot_id,
                    joint_id,
                    p.POSITION_CONTROL,
                    targetPosition=joint_targets[j_local],
                    force=110,
                    positionGain=0.09,
                    velocityGain=1.0,
                )

            # Keep fingers closed if attached.
            finger_target = 0.0 if self.attachment_constraint is not None else 0.04
            for j in FINGER_JOINTS:
                p.setJointMotorControl2(
                    self.robot_id,
                    j,
                    p.POSITION_CONTROL,
                    targetPosition=finger_target,
                    force=40,
                )
            p.stepSimulation()

    def _reset_robot(self) -> None:
        home = [0.0, -0.2, 0.0, -2.2, 0.0, 2.0, 0.8]
        for joint_id, q in zip(ARM_JOINTS, home):
            p.resetJointState(self.robot_id, joint_id, q)

        self.ee_target = np.array([0.58, 0.0, 0.20], dtype=np.float64)
        self._set_gripper(0.08, settle_steps=60)
        self._move_ee_to_target(steps=80)

    def _release_if_needed(self) -> None:
        if self.attachment_constraint is not None:
            p.removeConstraint(self.attachment_constraint)
            self.attachment_constraint = None

    def _ee_pos(self) -> np.ndarray:
        ee_state = p.getLinkState(self.robot_id, EE_LINK_INDEX, computeForwardKinematics=True)
        return np.array(ee_state[4], dtype=np.float64)

    def _cube_pos(self) -> np.ndarray:
        pos, _ = p.getBasePositionAndOrientation(self.cube_id)
        return np.array(pos, dtype=np.float64)

    def _is_aligned(self) -> int:
        ee = self._ee_pos()
        cube = self._cube_pos()
        return int(np.linalg.norm((cube - ee)[:2]) < 0.015)

    def _is_low_enough(self) -> int:
        ee = self._ee_pos()
        cube = self._cube_pos()
        return int(ee[2] < cube[2] + 0.07)

    def _is_grabbed(self) -> int:
        return int(self.attachment_constraint is not None)

    def _is_lifted(self) -> int:
        return int(self._cube_pos()[2] > 0.125)

    def _state(self) -> tuple[int, int, int, int]:
        return (self._is_aligned(), self._is_low_enough(), self._is_grabbed(), self._is_lifted())

    def reset(self, rng: np.random.Generator) -> tuple[tuple[int, int, int, int], dict]:
        self._release_if_needed()
        self._reset_robot()

        cube_x = rng.uniform(0.555, 0.605)
        cube_y = rng.uniform(-0.045, 0.045)
        p.resetBasePositionAndOrientation(self.cube_id, [cube_x, cube_y, self.cube_size / 2.0], [0, 0, 0, 1])
        p.resetBaseVelocity(self.cube_id, [0, 0, 0], [0, 0, 0])

        self.ee_target = np.array([
            cube_x + rng.uniform(-0.035, 0.035),
            cube_y + rng.uniform(-0.035, 0.035),
            0.20,
        ])
        self._move_ee_to_target(steps=120)

        self.step_count = 0
        self.milestones = {"aligned": False, "low": False, "grabbed": False}
        return self._state(), {}

    def _try_grasp(self) -> bool:
        if self.attachment_constraint is not None:
            return False

        ee = self._ee_pos()
        cube = self._cube_pos()
        dxy = float(np.linalg.norm((cube - ee)[:2]))
        dz = float(abs(cube[2] - ee[2]))
        if dxy > 0.022 or dz > 0.07:
            return False

        ee_state = p.getLinkState(self.robot_id, EE_LINK_INDEX, computeForwardKinematics=True)
        ee_pos, ee_orn = ee_state[4], ee_state[5]
        cube_pos, cube_orn = p.getBasePositionAndOrientation(self.cube_id)

        inv_ee_pos, inv_ee_orn = p.invertTransform(ee_pos, ee_orn)
        rel_pos, rel_orn = p.multiplyTransforms(inv_ee_pos, inv_ee_orn, cube_pos, cube_orn)

        self.attachment_constraint = p.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=EE_LINK_INDEX,
            childBodyUniqueId=self.cube_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=rel_pos,
            parentFrameOrientation=rel_orn,
            childFramePosition=[0, 0, 0],
            childFrameOrientation=[0, 0, 0, 1],
        )

        self._set_gripper(0.0, settle_steps=50)
        return True

    def step(self, action: int, rng: np.random.Generator) -> tuple[tuple[int, int, int, int], float, bool, dict]:
        # Actions:
        # 0 align_xy, 1 descend, 2 grasp, 3 lift, 4 random_xy_adjust
        self.step_count += 1

        prev_state = self._state()
        ee_prev = self._ee_pos()
        cube_prev = self._cube_pos()

        reward = -0.1
        newly_grasped = False

        if action == 0:
            self.ee_target[0] += float(np.clip(cube_prev[0] - ee_prev[0], -0.03, 0.03))
            self.ee_target[1] += float(np.clip(cube_prev[1] - ee_prev[1], -0.03, 0.03))
            self._move_ee_to_target(steps=14)
        elif action == 1:
            self.ee_target[2] -= 0.025
            self._move_ee_to_target(steps=14)
        elif action == 2:
            newly_grasped = self._try_grasp()
            if newly_grasped:
                reward += 6.0
            elif prev_state[2] == 1:
                reward -= 0.2
            else:
                reward -= 1.0
            self._step_sim(35)
        elif action == 3:
            self.ee_target[2] += 0.04
            self._move_ee_to_target(steps=14)
            if self._is_grabbed() == 0:
                reward -= 0.5
        else:
            self.ee_target[0] += float(rng.uniform(-0.015, 0.015))
            self.ee_target[1] += float(rng.uniform(-0.015, 0.015))
            self._move_ee_to_target(steps=12)

        self.ee_target[0] = float(np.clip(self.ee_target[0], 0.47, 0.69))
        self.ee_target[1] = float(np.clip(self.ee_target[1], -0.22, 0.22))
        self.ee_target[2] = float(np.clip(self.ee_target[2], 0.02, 0.34))

        ee = self._ee_pos()
        cube = self._cube_pos()
        dxy = float(np.linalg.norm((cube - ee)[:2]))

        aligned = dxy < 0.015
        low = ee[2] < cube_prev[2] + 0.07
        grabbed = self._is_grabbed() == 1

        if aligned and not self.milestones["aligned"]:
            self.milestones["aligned"] = True
            reward += 1.2

        if aligned and low and not self.milestones["low"]:
            self.milestones["low"] = True
            reward += 2.0

        if grabbed and not self.milestones["grabbed"]:
            self.milestones["grabbed"] = True
            reward += 4.0

        if action == 2 and (prev_state[0] == 0 or prev_state[1] == 0):
            reward -= 0.4

        success = bool(cube[2] > 0.10 and grabbed)
        done = success or (self.step_count >= self.max_steps)

        if success:
            reward += 30.0

        info = {
            "success": success,
            "cube_z": float(cube[2]),
            "newly_grasped": newly_grasped,
        }
        return self._state(), float(reward), done, info


def train_and_evaluate(
    train_episodes: int,
    eval_episodes: int,
    seed: int,
    gui: bool,
) -> PyBulletRLResult:
    rng = np.random.default_rng(seed)
    env = PandaPrimitivePickEnv(gui=gui)

    # State: aligned(2), low(2), grabbed(2), lifted(2) + action(5)
    n_actions = 5
    q = np.zeros((2, 2, 2, 2, n_actions), dtype=np.float64)

    gamma = 0.97
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.992

    train_success: list[int] = []
    curve_window = 50
    training_curve: list[dict[str, float]] = []

    for ep in range(train_episodes):
        state, _ = env.reset(rng)
        alpha = max(0.08, 0.35 * (1.0 - ep / train_episodes))

        ep_success = 0
        for _ in range(env.max_steps):
            if rng.random() < epsilon:
                action = int(rng.integers(0, n_actions))
            else:
                action = int(np.argmax(q[state]))

            next_state, reward, done, info = env.step(action, rng)

            best_next = float(np.max(q[next_state]))
            td_target = reward + (0.0 if done else gamma * best_next)
            q[state + (action,)] += alpha * (td_target - q[state + (action,)])

            state = next_state
            if info["success"]:
                ep_success = 1
                break
            if done:
                break

        train_success.append(ep_success)
        if (ep + 1) % curve_window == 0 or ep == train_episodes - 1:
            window = train_success[-curve_window:]
            training_curve.append(
                {
                    "episode": float(ep + 1),
                    "success_rate": float(np.mean(window)),
                }
            )
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    tail = train_success[-200:] if len(train_success) >= 200 else train_success
    tail_success_rate = float(np.mean(tail)) if tail else 0.0

    eval_success = 0
    eval_returns = []
    for _ in range(eval_episodes):
        state, _ = env.reset(rng)
        ep_return = 0.0
        for _ in range(env.max_steps):
            action = int(np.argmax(q[state]))
            state, reward, done, info = env.step(action, rng)
            ep_return += reward
            if info["success"]:
                eval_success += 1
                break
            if done:
                break
        eval_returns.append(ep_return)

    env.close()

    return PyBulletRLResult(
        task_name="Task2-PyBullet-RL-PickLift-Qlearning",
        train_episodes=train_episodes,
        eval_episodes=eval_episodes,
        max_steps_per_episode=24,
        training_success_rate_last_200=tail_success_rate,
        evaluation_success_rate=float(eval_success / eval_episodes),
        evaluation_avg_return=float(np.mean(eval_returns)),
        training_curve_window=curve_window,
        training_curve=training_curve,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Q-learning pick/lift policy in PyBullet")
    parser.add_argument("--train_episodes", type=int, default=500)
    parser.add_argument("--eval_episodes", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--output", default="results/task2_pybullet_rl_result.json")
    args = parser.parse_args()

    result = train_and_evaluate(
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        gui=args.gui,
    )

    payload = asdict(result)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
