#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "report_images"
OUT.mkdir(exist_ok=True)


def load_json(name: str):
    with (RESULTS / name).open("r", encoding="utf-8") as f:
        return json.load(f)


def fig_task1():
    d = load_json("task1_pybullet_result.json")
    labels = ["X error", "Y error", "Z error"]
    final_pos = np.array(d["final_cube_position"])
    target_pos = np.array(d["target_place_position"])
    errors = np.abs(final_pos - target_pos)

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, errors, color=["#4C78A8", "#F58518", "#54A24B"])
    for b, v in zip(bars, errors):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.0002, f"{v:.5f} m", ha="center", va="bottom", fontsize=9)
    plt.title("Task1: Place Position Error by Axis")
    plt.ylabel("Absolute Error (m)")
    plt.ylim(0, max(0.006, float(errors.max() * 1.25)))
    plt.tight_layout()
    plt.savefig(OUT / "task1_error_bar.png", dpi=160)
    plt.close()


def fig_task2():
    # Task2 report now focuses on training dynamics rather than final bar charts.
    # Keep this function as a no-op to preserve the script structure.
    return


def fig_task3():
    d = load_json("task3_demo_scaling_result.json")
    points = d["points"]
    xs = [p["demo_episodes"] for p in points]
    bc = [p["bc_success_rate"] for p in points]
    diff = [p["diffusion_success_rate"] for p in points]
    expert = float(d["expert_demo_success_rate"])

    plt.figure(figsize=(8, 4.8))
    plt.plot(xs, bc, marker="o", linewidth=2.2, color="#4C78A8", label="BC policy")
    plt.plot(xs, diff, marker="s", linewidth=2.2, color="#B279A2", label="Diffusion-style policy")
    plt.axhline(expert, color="#59A14F", linestyle="--", linewidth=1.6, label="Expert success")
    plt.xlabel("Expert Demonstration Episodes")
    plt.ylabel("Evaluation Success Rate")
    plt.title("Task3: Imitation Learning Data Scaling Trend")
    plt.ylim(-0.03, 1.05)
    plt.xlim(min(xs) - 3, max(xs) + 8)
    plt.grid(alpha=0.25)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT / "task3_demo_scaling_curve.png", dpi=170)
    plt.close()


def fig_task4():
    d = load_json("task4_vla_result.json")
    curve = d["training_curve"]
    epochs = [p["epoch"] for p in curve]
    train_mse = [p["train_mse"] for p in curve]
    test_mse = [p["test_mse"] for p in curve]
    success = [p["test_success_rate"] for p in curve]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))

    axes[0].plot(epochs, train_mse, marker="o", linewidth=2.0, color="#4C78A8", label="Train MSE")
    axes[0].plot(epochs, test_mse, marker="s", linewidth=2.0, color="#F58518", label="Test MSE")
    axes[0].set_xlabel("Training Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Prediction Error Trend")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, success, marker="o", linewidth=2.2, color="#54A24B", label="Test Success")
    axes[1].set_xlabel("Training Epoch")
    axes[1].set_ylabel("Success Rate")
    axes[1].set_title("Task Success Trend")
    axes[1].set_ylim(0.25, 0.68)
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="lower right")

    fig.suptitle("Task4: Mini VLA Training Trend", y=1.02)
    fig.tight_layout()
    plt.savefig(OUT / "task4_vla_training_curve.png", dpi=170, bbox_inches="tight")
    plt.close()


def fig_task5():
    d = load_json("task5_planning_result.json")
    curve = d["complexity_curve"]
    levels = [p["level"] for p in curve]
    xticklabels = [f"L{int(p['level'])}\n{int(p['grid_size'])}x{int(p['grid_size'])}\nwall {p['wall_density']:.2f}" for p in curve]
    series = [
        ("Zero-shot", [p["zero_shot_success_rate"] for p in curve], "#E45756", "o"),
        ("ICL", [p["icl_success_rate"] for p in curve], "#4C78A8", "s"),
        ("CoT", [p["cot_success_rate"] for p in curve], "#54A24B", "^"),
        ("SFT", [p["sft_success_rate"] for p in curve], "#B279A2", "D"),
    ]

    plt.figure(figsize=(9.5, 4.8))
    for label, values, color, marker in series:
        plt.plot(levels, values, marker=marker, linewidth=2.2, label=label, color=color)
    plt.xticks(levels, xticklabels)
    plt.xlabel("Planning Complexity Level")
    plt.ylabel("Success Rate")
    plt.title("Task5: Planning Robustness as Task Complexity Increases")
    plt.ylim(0.42, 1.05)
    plt.grid(alpha=0.25)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(OUT / "task5_planning_complexity_curve.png", dpi=170)
    plt.close()


def fig_task6():
    d = load_json("task6_humanoid_result.json")
    curve = d["rl_training_curve"]
    episodes = [p["episode"] for p in curve]
    tracking = [p["tracking_mse"] for p in curve]

    plt.figure(figsize=(8.5, 4.8))
    plt.plot(episodes, tracking, marker="o", linewidth=2.2, color="#E45756", label="RL residual snapshot")
    plt.axhline(d["imitation_tracking_mse"], color="#B279A2", linestyle="--", linewidth=1.8, label="Imitation baseline")
    plt.axhline(d["teleop_tracking_mse"], color="#59A14F", linestyle="--", linewidth=1.8, label="Teleop reference")
    plt.axhline(d["rl_tracking_mse"], color="#4C78A8", linestyle=":", linewidth=2.0, label="Final RL eval")
    plt.xlabel("RL Residual Training Episode")
    plt.ylabel("Tracking MSE")
    plt.title("Task6: Humanoid RL Residual Tracking Trend")
    plt.ylim(0.0, max(0.16, float(max(tracking) * 1.12)))
    plt.grid(alpha=0.25)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(OUT / "task6_rl_training_curve.png", dpi=170)
    plt.close()


def fig_task2_training_curve():
    d_gym = load_json("task2_gym_qlearning_result.json")
    d_pb = load_json("task2_pybullet_rl_result.json")

    plt.figure(figsize=(8, 5))

    for item, label, color in zip(
        d_gym["results"],
        ["FrozenLake-v1", "Taxi-v3"],
        ["#4C78A8", "#F58518"],
    ):
        xs = [p["episode"] for p in item["training_curve"]]
        ys = [p["success_rate"] for p in item["training_curve"]]
        plt.plot(xs, ys, marker="o", linewidth=2, label=label, color=color)

    xs = [p["episode"] for p in d_pb["training_curve"]]
    ys = [p["success_rate"] for p in d_pb["training_curve"]]
    plt.plot(xs, ys, marker="o", linewidth=2, label="PyBullet-PickLift", color="#54A24B")

    plt.xlabel("Training Episode")
    plt.ylabel("Window Success Rate")
    plt.title("Task2: RL Training Success Rate Trend")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "task2_training_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    for item, label, color in zip(
        d_gym["results"],
        ["FrozenLake-v1", "Taxi-v3"],
        ["#4C78A8", "#F58518"],
    ):
        xs = [p["episode"] for p in item["training_curve"]]
        ys = [p["success_rate"] for p in item["training_curve"]]
        plt.plot(xs, ys, marker="o", linewidth=2, label=label, color=color)
    plt.xlabel("Training Episode")
    plt.ylabel("Window Success Rate")
    plt.title("Task2 Gym: Training Success Rate Trend")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "task2_gym_training_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    xs = [p["episode"] for p in d_pb["training_curve"]]
    ys = [p["success_rate"] for p in d_pb["training_curve"]]
    plt.plot(xs, ys, marker="o", linewidth=2, label="PyBullet-PickLift", color="#54A24B")
    plt.xlabel("Training Episode")
    plt.ylabel("Window Success Rate")
    plt.title("Task2 PyBullet: Training Success Rate Trend")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "task2_pybullet_training_curve.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    fig_task1()
    fig_task2()
    fig_task3()
    fig_task4()
    fig_task5()
    fig_task6()
    fig_task2_training_curve()
    print("Saved figures to", OUT)
