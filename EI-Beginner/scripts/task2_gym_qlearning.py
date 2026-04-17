#!/usr/bin/env python3
"""Task 2: Reinforcement learning training and evaluation in Gymnasium.

This script trains tabular Q-learning agents on two Gymnasium tasks
and reports success rate on held-out evaluation episodes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict

import gymnasium as gym
import numpy as np


@dataclass
class RLTaskResult:
    env_name: str
    train_episodes: int
    eval_episodes: int
    training_success_rate_last_500: float
    evaluation_success_rate: float
    evaluation_avg_return: float
    training_curve_window: int
    training_curve: list[dict[str, float]]


def is_success(env_name: str, reward: float, terminated: bool) -> bool:
    if env_name == "FrozenLake-v1":
        return terminated and reward > 0.0
    if env_name == "Taxi-v3":
        return terminated and reward >= 20.0
    return terminated and reward > 0.0


def train_q_learning(
    env_name: str,
    make_kwargs: dict,
    episodes: int,
    max_steps: int,
    alpha: float,
    gamma: float,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay: float,
    seed: int,
    curve_window: int,
) -> tuple[np.ndarray, float]:
    env = gym.make(env_name, **make_kwargs)

    if not isinstance(env.observation_space, gym.spaces.Discrete):
        raise ValueError(f"{env_name} observation space must be Discrete for tabular Q-learning")
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise ValueError(f"{env_name} action space must be Discrete for tabular Q-learning")

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q_table = np.zeros((n_states, n_actions), dtype=np.float64)

    rng = np.random.default_rng(seed)
    epsilon = epsilon_start
    success_flags: list[int] = []
    training_curve: list[dict[str, float]] = []

    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep)
        ep_success = 0

        for _ in range(max_steps):
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[state]))

            next_state, reward, terminated, truncated, _ = env.step(action)

            best_next = np.max(q_table[next_state])
            td_target = reward + gamma * best_next * (0 if (terminated or truncated) else 1)
            td_error = td_target - q_table[state, action]
            q_table[state, action] += alpha * td_error

            if is_success(env_name, reward, terminated):
                ep_success = 1

            state = next_state
            if terminated or truncated:
                break

        success_flags.append(ep_success)
        if (ep + 1) % curve_window == 0 or ep == episodes - 1:
            window = success_flags[-curve_window:]
            training_curve.append(
                {
                    "episode": float(ep + 1),
                    "success_rate": float(np.mean(window)),
                }
            )
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    env.close()

    tail = success_flags[-500:] if len(success_flags) >= 500 else success_flags
    tail_success = float(np.mean(tail)) if tail else 0.0
    return q_table, tail_success, training_curve


def evaluate_q_policy(
    env_name: str,
    make_kwargs: dict,
    q_table: np.ndarray,
    eval_episodes: int,
    max_steps: int,
    seed: int,
) -> tuple[float, float]:
    env = gym.make(env_name, **make_kwargs)

    successes = 0
    returns = []

    for ep in range(eval_episodes):
        state, _ = env.reset(seed=seed + 10000 + ep)
        total_reward = 0.0

        for _ in range(max_steps):
            action = int(np.argmax(q_table[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if is_success(env_name, reward, terminated):
                successes += 1

            if terminated or truncated:
                break

        returns.append(total_reward)

    env.close()

    success_rate = successes / eval_episodes if eval_episodes > 0 else 0.0
    avg_return = float(np.mean(returns)) if returns else 0.0
    return success_rate, avg_return


def run_single_task(config: dict, seed: int) -> RLTaskResult:
    env_name = config["env_name"]
    make_kwargs = config.get("make_kwargs", {})

    curve_window = config.get("curve_window", 500)
    q_table, train_tail_success, training_curve = train_q_learning(
        env_name=env_name,
        make_kwargs=make_kwargs,
        episodes=config["train_episodes"],
        max_steps=config["max_steps"],
        alpha=config["alpha"],
        gamma=config["gamma"],
        epsilon_start=config["epsilon_start"],
        epsilon_end=config["epsilon_end"],
        epsilon_decay=config["epsilon_decay"],
        seed=seed,
        curve_window=curve_window,
    )

    eval_success, eval_avg_return = evaluate_q_policy(
        env_name=env_name,
        make_kwargs=make_kwargs,
        q_table=q_table,
        eval_episodes=config["eval_episodes"],
        max_steps=config["max_steps"],
        seed=seed,
    )

    return RLTaskResult(
        env_name=env_name,
        train_episodes=config["train_episodes"],
        eval_episodes=config["eval_episodes"],
        training_success_rate_last_500=train_tail_success,
        evaluation_success_rate=eval_success,
        evaluation_avg_return=eval_avg_return,
        training_curve_window=curve_window,
        training_curve=training_curve,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Q-learning on two Gymnasium tasks")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/task2_gym_qlearning_result.json")
    args = parser.parse_args()

    task_configs = [
        {
            "env_name": "FrozenLake-v1",
            "make_kwargs": {"map_name": "4x4", "is_slippery": False},
            "train_episodes": 7000,
            "eval_episodes": 1000,
            "max_steps": 100,
            "alpha": 0.12,
            "gamma": 0.99,
            "epsilon_start": 1.0,
            "epsilon_end": 0.02,
            "epsilon_decay": 0.9992,
            "curve_window": 500,
        },
        {
            "env_name": "Taxi-v3",
            "make_kwargs": {},
            "train_episodes": 22000,
            "eval_episodes": 1000,
            "max_steps": 200,
            "alpha": 0.35,
            "gamma": 0.99,
            "epsilon_start": 1.0,
            "epsilon_end": 0.02,
            "epsilon_decay": 0.9997,
            "curve_window": 500,
        },
    ]

    results = [asdict(run_single_task(cfg, seed=args.seed)) for cfg in task_configs]

    payload = {
        "task_name": "Task2-RL-Gymnasium-Qlearning",
        "seed": args.seed,
        "results": results,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
