#!/usr/bin/env python3
"""Task 3: Imitation learning baseline with BC + diffusion-style policy.

- Collect expert demonstrations in PyBullet Panda pick-lift environment.
- Train Behavior Cloning (BC) policy.
- Train a diffusion-style denoising policy for discrete action selection.
- Export demonstrations in a LeRobot-like dataset format.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from task2_pybullet_qlearning_pick import PandaPrimitivePickEnv


@dataclass
class Task3Result:
    task_name: str
    seed: int
    demo_episodes: int
    demo_transitions: int
    expert_success_rate: float
    bc_success_rate: float
    diffusion_success_rate: float
    bc_train_accuracy: float
    diffusion_train_loss: float
    lerobot_dataset_jsonl: str
    lerobot_meta_json: str


class BCPolicy(nn.Module):
    def __init__(self, in_dim: int = 4, n_actions: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiffusionDenoiser(nn.Module):
    def __init__(self, state_dim: int = 4, action_dim: int = 5, t_dim: int = 16):
        super().__init__()
        self.t_dim = t_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + t_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def t_embed(self, t: torch.Tensor) -> torch.Tensor:
        # Sinusoidal embedding
        half = self.t_dim // 2
        freqs = torch.exp(torch.linspace(np.log(1.0), np.log(1000.0), half, device=t.device))
        x = t.float().unsqueeze(1) / freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(x), torch.cos(x)], dim=1)
        return emb

    def forward(self, state: torch.Tensor, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self.t_embed(t)
        x = torch.cat([state, xt, te], dim=1)
        return self.net(x)


def expert_action(state: tuple[int, int, int, int]) -> int:
    aligned, low, grabbed, lifted = state
    if lifted == 1:
        return 3
    if grabbed == 1:
        return 3
    if aligned == 0:
        return 0
    if low == 0:
        return 1
    return 2


def run_policy(
    env: PandaPrimitivePickEnv,
    rng: np.random.Generator,
    policy_fn: Callable[[tuple[int, int, int, int]], int],
    episodes: int,
) -> tuple[float, float]:
    successes = 0
    returns = []

    for _ in range(episodes):
        state, _ = env.reset(rng)
        ep_return = 0.0
        for _ in range(env.max_steps):
            action = int(policy_fn(state))
            state, reward, done, info = env.step(action, rng)
            ep_return += reward
            if info["success"]:
                successes += 1
                break
            if done:
                break
        returns.append(ep_return)

    success_rate = successes / episodes if episodes > 0 else 0.0
    avg_ret = float(np.mean(returns)) if returns else 0.0
    return success_rate, avg_ret


def collect_expert_demos(
    env: PandaPrimitivePickEnv,
    rng: np.random.Generator,
    demo_episodes: int,
) -> tuple[np.ndarray, np.ndarray, float, list[dict]]:
    states: list[list[float]] = []
    actions: list[int] = []
    episodes_success = 0
    trajectory_rows: list[dict] = []

    ep_idx = 0
    while ep_idx < demo_episodes:
        state, _ = env.reset(rng)
        ep_success = False
        frame_idx = 0

        for _ in range(env.max_steps):
            act = expert_action(state)
            next_state, reward, done, info = env.step(act, rng)

            states.append([float(x) for x in state])
            actions.append(int(act))

            trajectory_rows.append(
                {
                    "episode_index": ep_idx,
                    "frame_index": frame_idx,
                    "observation.state": [int(x) for x in state],
                    "action.discrete": int(act),
                    "reward": float(reward),
                    "done": bool(done),
                    "success": bool(info["success"]),
                }
            )

            frame_idx += 1
            state = next_state
            if info["success"]:
                ep_success = True
                break
            if done:
                break

        if ep_success:
            episodes_success += 1
        ep_idx += 1

    x = np.asarray(states, dtype=np.float32)
    y = np.asarray(actions, dtype=np.int64)
    success_rate = episodes_success / demo_episodes if demo_episodes > 0 else 0.0
    return x, y, success_rate, trajectory_rows


def train_bc(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[BCPolicy, float]:
    torch.manual_seed(seed)

    model = BCPolicy(in_dim=4, n_actions=5)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y)

    batch_size = 256
    n = x_t.shape[0]

    for _ in range(40):
        perm = torch.randperm(n)
        x_t = x_t[perm]
        y_t = y_t[perm]

        for i in range(0, n, batch_size):
            xb = x_t[i : i + batch_size]
            yb = y_t[i : i + batch_size]
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        pred = torch.argmax(model(torch.from_numpy(x)), dim=1).numpy()
    train_acc = float((pred == y).mean())
    return model, train_acc


def train_diffusion_style(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    timesteps: int = 20,
) -> tuple[DiffusionDenoiser, dict, float]:
    torch.manual_seed(seed)

    model = DiffusionDenoiser(state_dim=4, action_dim=5, t_dim=16)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    x_t = torch.from_numpy(x)
    y_onehot = F.one_hot(torch.from_numpy(y), num_classes=5).float()

    betas = torch.linspace(1e-4, 0.02, timesteps)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    n = x_t.shape[0]
    batch_size = 256
    last_loss = 0.0

    for _ in range(50):
        perm = torch.randperm(n)
        x_t = x_t[perm]
        y_onehot = y_onehot[perm]

        for i in range(0, n, batch_size):
            s = x_t[i : i + batch_size]
            a0 = y_onehot[i : i + batch_size]

            b = s.shape[0]
            t = torch.randint(0, timesteps, (b,))
            noise = torch.randn_like(a0)

            abar = alpha_bars[t].unsqueeze(1)
            xt = torch.sqrt(abar) * a0 + torch.sqrt(1.0 - abar) * noise

            noise_pred = model(s, xt, t)
            loss = F.mse_loss(noise_pred, noise)

            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = float(loss.item())

    sched = {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "timesteps": timesteps,
    }
    return model, sched, last_loss


def diffusion_action(
    model: DiffusionDenoiser,
    sched: dict,
    state: tuple[int, int, int, int],
) -> int:
    betas = sched["betas"]
    alphas = sched["alphas"]
    alpha_bars = sched["alpha_bars"]
    timesteps = int(sched["timesteps"])

    with torch.no_grad():
        s = torch.tensor([[float(x) for x in state]], dtype=torch.float32)
        x = torch.randn((1, 5), dtype=torch.float32)

        for t_inv in range(timesteps - 1, -1, -1):
            t = torch.tensor([t_inv], dtype=torch.long)
            eps = model(s, x, t)

            alpha_t = alphas[t_inv]
            abar_t = alpha_bars[t_inv]
            beta_t = betas[t_inv]

            x = (1.0 / torch.sqrt(alpha_t)) * (x - ((1.0 - alpha_t) / torch.sqrt(1.0 - abar_t)) * eps)
            if t_inv > 0:
                z = torch.randn_like(x)
                x = x + torch.sqrt(beta_t) * z

        act = int(torch.argmax(x, dim=1).item())
    return act


def export_lerobot_like(
    rows: list[dict],
    out_jsonl: Path,
    out_meta: Path,
) -> None:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta = {
        "format": "lerobot_like_v1",
        "observation_keys": ["observation.state"],
        "action_keys": ["action.discrete"],
        "num_rows": len(rows),
        "note": "Minimal LeRobot-style offline dataset export for this project.",
    }
    with out_meta.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task3 imitation learning baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo_episodes", type=int, default=280)
    parser.add_argument("--eval_episodes", type=int, default=120)
    parser.add_argument("--output", default="results/task3_imitation_result.json")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    env = PandaPrimitivePickEnv(gui=False)
    try:
        x, y, expert_success, rows = collect_expert_demos(env, rng, args.demo_episodes)

        bc_model, bc_acc = train_bc(x, y, seed=args.seed)
        diff_model, sched, diff_loss = train_diffusion_style(x, y, seed=args.seed)

        bc_success, _ = run_policy(
            env,
            rng,
            lambda s: int(torch.argmax(bc_model(torch.tensor([[float(v) for v in s]], dtype=torch.float32)), dim=1).item()),
            args.eval_episodes,
        )

        diffusion_success, _ = run_policy(
            env,
            rng,
            lambda s: diffusion_action(diff_model, sched, s),
            args.eval_episodes,
        )

    finally:
        env.close()

    dataset_jsonl = Path("results/task3_lerobot_dataset.jsonl")
    dataset_meta = Path("results/task3_lerobot_meta.json")
    export_lerobot_like(rows, dataset_jsonl, dataset_meta)

    result = Task3Result(
        task_name="Task3-Imitation-DiffusionPolicy-Like",
        seed=args.seed,
        demo_episodes=args.demo_episodes,
        demo_transitions=int(len(x)),
        expert_success_rate=float(expert_success),
        bc_success_rate=float(bc_success),
        diffusion_success_rate=float(diffusion_success),
        bc_train_accuracy=float(bc_acc),
        diffusion_train_loss=float(diff_loss),
        lerobot_dataset_jsonl=str(dataset_jsonl),
        lerobot_meta_json=str(dataset_meta),
    )

    payload = asdict(result)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
