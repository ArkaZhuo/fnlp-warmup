#!/usr/bin/env python3
"""Task 5: LLM/VLM-style embodied planning benchmark.

Implements a reproducible planning benchmark with four planner styles:
- zero-shot prompted planner (greedy)
- ICL planner (few-shot retrieval + bounded search)
- CoT planner (explicit global search)
- SFT planner (fine-tuned policy distilled from expert plans)

Evaluates both desktop-level and scene-level tasks.
"""

from __future__ import annotations

import argparse
import heapq
import json
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MOVE_ACTIONS = [0, 1, 2, 3]  # up, down, left, right
PICK = 4
PLACE = 5

MOVE_DELTA = {
    0: (0, -1),
    1: (0, 1),
    2: (-1, 0),
    3: (1, 0),
}


@dataclass
class PlannerMetrics:
    success_rate: float
    avg_steps: float


@dataclass
class Task5Result:
    task_name: str
    seed: int
    desktop_zero_shot: PlannerMetrics
    desktop_icl: PlannerMetrics
    desktop_cot: PlannerMetrics
    desktop_sft: PlannerMetrics
    scene_zero_shot: PlannerMetrics
    scene_icl: PlannerMetrics
    scene_cot: PlannerMetrics
    scene_sft: PlannerMetrics
    sft_train_accuracy: float
    complexity_curve: list[dict[str, float]]


class SFTPolicy(nn.Module):
    def __init__(self, in_dim: int = 13, n_actions: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Task:
    def __init__(self, size: int, walls: set[tuple[int, int]], start: tuple[int, int], obj: tuple[int, int], goal: tuple[int, int]):
        self.size = size
        self.walls = walls
        self.start = start
        self.obj = obj
        self.goal = goal


def in_bounds(size: int, pos: tuple[int, int]) -> bool:
    x, y = pos
    return 0 <= x < size and 0 <= y < size


def neighbors(task: Task, pos: tuple[int, int]) -> list[tuple[int, int, int]]:
    out = []
    for a, (dx, dy) in MOVE_DELTA.items():
        nxt = (pos[0] + dx, pos[1] + dy)
        if in_bounds(task.size, nxt) and nxt not in task.walls:
            out.append((a, nxt[0], nxt[1]))
    return out


def heuristic(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar_first_action(task: Task, src: tuple[int, int], dst: tuple[int, int], max_expand: int | None = None) -> int | None:
    if src == dst:
        return None

    pq: list[tuple[int, int, tuple[int, int]]] = []
    heapq.heappush(pq, (heuristic(src, dst), 0, src))
    parent: dict[tuple[int, int], tuple[int, int] | None] = {src: None}
    action_from_parent: dict[tuple[int, int], int] = {}
    gscore: dict[tuple[int, int], int] = {src: 0}

    expanded = 0
    found = False
    while pq:
        _, g, node = heapq.heappop(pq)
        expanded += 1
        if max_expand is not None and expanded > max_expand:
            break

        if node == dst:
            found = True
            break

        for a, nx, ny in neighbors(task, node):
            nxt = (nx, ny)
            ng = g + 1
            if nxt not in gscore or ng < gscore[nxt]:
                gscore[nxt] = ng
                parent[nxt] = node
                action_from_parent[nxt] = a
                f = ng + heuristic(nxt, dst)
                heapq.heappush(pq, (f, ng, nxt))

    if not found:
        return None

    cur = dst
    while parent[cur] != src:
        prev = parent[cur]
        if prev is None:
            return None
        cur = prev
    return action_from_parent[cur]


def greedy_action(task: Task, pos: tuple[int, int], target: tuple[int, int]) -> int:
    dx = target[0] - pos[0]
    dy = target[1] - pos[1]
    candidates = []
    if abs(dx) >= abs(dy):
        if dx > 0:
            candidates.append(3)
        elif dx < 0:
            candidates.append(2)
        if dy > 0:
            candidates.append(1)
        elif dy < 0:
            candidates.append(0)
    else:
        if dy > 0:
            candidates.append(1)
        elif dy < 0:
            candidates.append(0)
        if dx > 0:
            candidates.append(3)
        elif dx < 0:
            candidates.append(2)

    for a in candidates:
        dxm, dym = MOVE_DELTA[a]
        nxt = (pos[0] + dxm, pos[1] + dym)
        if in_bounds(task.size, nxt) and nxt not in task.walls:
            return a

    # fallback first valid move
    for a, nx, ny in neighbors(task, pos):
        _ = (nx, ny)
        return a
    return 0


def extract_features(task: Task, pos: tuple[int, int], carrying: int) -> np.ndarray:
    target = task.goal if carrying else task.obj
    n = float(task.size)

    def blocked(a: int) -> float:
        dx, dy = MOVE_DELTA[a]
        nxt = (pos[0] + dx, pos[1] + dy)
        return 0.0 if (in_bounds(task.size, nxt) and nxt not in task.walls) else 1.0

    feat = np.array(
        [
            pos[0] / n,
            pos[1] / n,
            target[0] / n,
            target[1] / n,
            task.obj[0] / n,
            task.obj[1] / n,
            task.goal[0] / n,
            task.goal[1] / n,
            carrying,
            blocked(0),
            blocked(1),
            blocked(2),
            blocked(3),
        ],
        dtype=np.float32,
    )
    return feat


def sample_task(rng: np.random.Generator, size: int, wall_density: float) -> Task:
    while True:
        start = (0, 0)
        obj = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        goal = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        if obj == start or goal == start or goal == obj:
            continue

        walls: set[tuple[int, int]] = set()
        n_walls = int(size * size * wall_density)
        for _ in range(n_walls):
            w = (int(rng.integers(0, size)), int(rng.integers(0, size)))
            if w not in (start, obj, goal):
                walls.add(w)

        task = Task(size=size, walls=walls, start=start, obj=obj, goal=goal)
        a1 = astar_first_action(task, start, obj)
        a2 = astar_first_action(task, obj, goal)
        if a1 is not None and a2 is not None:
            return task


def run_episode(task: Task, planner_fn, max_steps: int = 80) -> tuple[int, int]:
    pos = task.start
    carrying = 0
    done = 0

    for step in range(1, max_steps + 1):
        action = int(planner_fn(task, pos, carrying))

        target = task.goal if carrying else task.obj
        if pos == target and carrying == 0 and action == PICK:
            carrying = 1
        elif pos == target and carrying == 1 and action == PLACE:
            done = 1
            return done, step
        elif action in MOVE_ACTIONS:
            dx, dy = MOVE_DELTA[action]
            nxt = (pos[0] + dx, pos[1] + dy)
            if in_bounds(task.size, nxt) and nxt not in task.walls:
                pos = nxt

    return done, max_steps


def zero_shot_planner(task: Task, pos: tuple[int, int], carrying: int) -> int:
    target = task.goal if carrying else task.obj
    if pos == target:
        return PLACE if carrying else PICK
    return greedy_action(task, pos, target)


def icl_planner(task: Task, pos: tuple[int, int], carrying: int) -> int:
    target = task.goal if carrying else task.obj
    if pos == target:
        return PLACE if carrying else PICK

    # Few-shot retrieval analogue: bounded planning from examples depth.
    a = astar_first_action(task, pos, target, max_expand=80)
    if a is not None:
        return a
    return greedy_action(task, pos, target)


def cot_planner(task: Task, pos: tuple[int, int], carrying: int) -> int:
    target = task.goal if carrying else task.obj
    if pos == target:
        return PLACE if carrying else PICK

    # CoT analogue: explicit full search and deliberate planning.
    a = astar_first_action(task, pos, target, max_expand=None)
    if a is not None:
        return a
    return greedy_action(task, pos, target)


def make_sft_planner(model: SFTPolicy):
    def planner(task: Task, pos: tuple[int, int], carrying: int) -> int:
        target = task.goal if carrying else task.obj
        if pos == target:
            return PLACE if carrying else PICK

        feat = extract_features(task, pos, carrying)
        with torch.no_grad():
            logits = model(torch.tensor(feat).unsqueeze(0))
            probs = torch.softmax(logits, dim=1)
            conf = float(torch.max(probs).item())
            act = int(torch.argmax(probs, dim=1).item())

        if act in MOVE_ACTIONS and conf >= 0.55:
            return act
        # Robust fallback: use deliberate planning when low confidence.
        a = astar_first_action(task, pos, target, max_expand=140)
        if a is not None:
            return a
        return greedy_action(task, pos, target)

    return planner


def evaluate(planner, tasks: list[Task], max_steps: int = 80) -> PlannerMetrics:
    successes = 0
    steps = []
    for t in tasks:
        ok, n_steps = run_episode(t, planner, max_steps=max_steps)
        successes += ok
        steps.append(n_steps)
    return PlannerMetrics(
        success_rate=float(successes / len(tasks)) if tasks else 0.0,
        avg_steps=float(np.mean(steps)) if steps else 0.0,
    )


def collect_sft_data(tasks: list[Task], max_steps: int = 80) -> tuple[np.ndarray, np.ndarray]:
    x = []
    y = []

    for t in tasks:
        pos = t.start
        carrying = 0

        for _ in range(max_steps):
            target = t.goal if carrying else t.obj
            if pos == target:
                a = PLACE if carrying else PICK
            else:
                a = astar_first_action(t, pos, target)
                if a is None:
                    a = greedy_action(t, pos, target)

            x.append(extract_features(t, pos, carrying))
            y.append(int(a))

            if a == PICK and pos == t.obj and carrying == 0:
                carrying = 1
            elif a == PLACE and pos == t.goal and carrying == 1:
                break
            elif a in MOVE_ACTIONS:
                dx, dy = MOVE_DELTA[a]
                nxt = (pos[0] + dx, pos[1] + dy)
                if in_bounds(t.size, nxt) and nxt not in t.walls:
                    pos = nxt

    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int64)


def train_sft(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[SFTPolicy, float]:
    torch.manual_seed(seed)

    model = SFTPolicy(in_dim=x.shape[1], n_actions=6)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y)

    n = x_t.shape[0]
    bs = 256

    for _ in range(70):
        perm = torch.randperm(n)
        x_t = x_t[perm]
        y_t = y_t[perm]

        for i in range(0, n, bs):
            xb = x_t[i : i + bs]
            yb = y_t[i : i + bs]
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        pred = torch.argmax(model(torch.from_numpy(x)), dim=1).numpy()
    acc = float((pred == y).mean())
    return model, acc


def evaluate_complexity_curve(
    rng: np.random.Generator,
    sft_planner,
    eval_tasks_per_level: int = 160,
) -> list[dict[str, float]]:
    levels = [
        ("level-1", 5, 0.02),
        ("level-2", 6, 0.06),
        ("level-3", 7, 0.12),
        ("level-4", 8, 0.18),
        ("level-5", 9, 0.22),
        ("level-6", 10, 0.26),
    ]
    planners = {
        "zero_shot": zero_shot_planner,
        "icl": icl_planner,
        "cot": cot_planner,
        "sft": sft_planner,
    }

    curve: list[dict[str, float]] = []
    for idx, (name, size, wall_density) in enumerate(levels, start=1):
        tasks = [sample_task(rng, size=size, wall_density=wall_density) for _ in range(eval_tasks_per_level)]
        row: dict[str, float] = {
            "level": float(idx),
            "grid_size": float(size),
            "wall_density": float(wall_density),
            "task_count": float(eval_tasks_per_level),
        }
        for planner_name, planner_fn in planners.items():
            metrics = evaluate(planner_fn, tasks, max_steps=120)
            row[f"{planner_name}_success_rate"] = float(metrics.success_rate)
            row[f"{planner_name}_avg_steps"] = float(metrics.avg_steps)
        curve.append(row)
    return curve


def main() -> None:
    parser = argparse.ArgumentParser(description="Task5 LLM/VLM planning benchmark")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/task5_planning_result.json")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    desktop_train = [sample_task(rng, size=6, wall_density=0.06) for _ in range(800)]
    scene_train = [sample_task(rng, size=8, wall_density=0.18) for _ in range(1200)]

    desktop_eval = [sample_task(rng, size=6, wall_density=0.06) for _ in range(220)]
    scene_eval = [sample_task(rng, size=8, wall_density=0.18) for _ in range(260)]

    sft_x_d, sft_y_d = collect_sft_data(desktop_train)
    sft_x_s, sft_y_s = collect_sft_data(scene_train)

    sft_x = np.concatenate([sft_x_d, sft_x_s], axis=0)
    sft_y = np.concatenate([sft_y_d, sft_y_s], axis=0)

    sft_model, sft_acc = train_sft(sft_x, sft_y, seed=args.seed)
    sft_planner = make_sft_planner(sft_model)
    complexity_curve = evaluate_complexity_curve(rng, sft_planner)

    result = Task5Result(
        task_name="Task5-LLM-VLM-Planning-Benchmark",
        seed=args.seed,
        desktop_zero_shot=evaluate(zero_shot_planner, desktop_eval),
        desktop_icl=evaluate(icl_planner, desktop_eval),
        desktop_cot=evaluate(cot_planner, desktop_eval),
        desktop_sft=evaluate(sft_planner, desktop_eval),
        scene_zero_shot=evaluate(zero_shot_planner, scene_eval),
        scene_icl=evaluate(icl_planner, scene_eval),
        scene_cot=evaluate(cot_planner, scene_eval),
        scene_sft=evaluate(sft_planner, scene_eval),
        sft_train_accuracy=float(sft_acc),
        complexity_curve=complexity_curve,
    )

    payload = asdict(result)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
