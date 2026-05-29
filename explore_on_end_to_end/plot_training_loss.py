"""
Training Loss Plotter

从 HuggingFace Trainer 输出的 trainer_state.json 中读取 loss 数据，
用 matplotlib 绘制训练损失曲线（原始值 + EMA 平滑值）。

用法:
    python plot_training_loss.py <output_dir>

    <output_dir> 是 Trainer 的 output_dir，必须包含 trainer_state.json。

输出:
    <output_dir>/training_loss.png

依赖:
    - matplotlib
    - numpy (可选，仅用于计算平滑权重)

参考:
    LLaMA-Factory extras/ploting.py — plot_loss
"""

import json
import math
import os
import sys
from typing import Optional


# ── 平滑算法 ──────────────────────────────────────────────────────


def smooth(scalars: list[float], weight: Optional[float] = None) -> list[float]:
    """
    EMA 平滑（指数移动平均）。

    Args:
        scalars: 原始 loss 序列。
        weight: EMA 权重（0~1），越大越平滑。
                不传则根据数据点数用 sigmoid 动态计算。

    Returns:
        平滑后的 loss 序列（长度与输入相同）。
    """
    if len(scalars) == 0:
        return []

    if weight is None:
        # sigmoid 动态权重：数据点越多 weight 越大
        weight = 1.8 * (1 / (1 + math.exp(-0.05 * len(scalars))) - 0.5)

    last = scalars[0]
    smoothed = []
    for next_val in scalars:
        smoothed_val = last * weight + (1 - weight) * next_val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


# ── 核心绘图函数 ──────────────────────────────────────────────────


def plot_loss(
    output_dir: str,
    keys: list[str] = None,
    save_name: str = "training_loss.png",
    dpi: int = 100,
    show_original: bool = True,
    show_smoothed: bool = True,
):
    """
    从 trainer_state.json 中读取 loss，绘制并保存为图片。

    Args:
        output_dir: 训练输出目录（需包含 trainer_state.json）。
        keys: 要绘制的指标名列表，默认 ["loss"]。
              可以是 "eval_loss" 等其他 Trainer 记录的指标。
        save_name: 输出图片文件名，保存在 output_dir 下。
        dpi: 图片分辨率。
        show_original: 是否绘制原始曲线（半透明）。
        show_smoothed: 是否绘制平滑曲线（实线）。
    """
    if keys is None:
        keys = ["loss"]

    state_path = os.path.join(output_dir, "trainer_state.json")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"未找到 trainer_state.json: {state_path}")

    with open(state_path, encoding="utf-8") as f:
        data = json.load(f)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("需要 matplotlib，请先安装: pip install matplotlib")

    for key in keys:
        steps = []
        metrics = []
        for entry in data.get("log_history", []):
            if key in entry:
                steps.append(entry["step"])
                metrics.append(entry[key])

        if len(metrics) == 0:
            print(f"[WARNING] 未找到指标 '{key}'，跳过。")
            continue

        plt.figure(figsize=(10, 5))

        if show_original:
            plt.plot(
                steps, metrics,
                color="#1f77b4", alpha=0.4, linewidth=1,
                label=f"{key} (original)",
            )

        if show_smoothed:
            plt.plot(
                steps, smooth(metrics),
                color="#1f77b4", linewidth=2,
                label=f"{key} (smoothed)",
            )

        plt.title(f"Training {key}")
        plt.xlabel("Step")
        plt.ylabel(key)
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_path = os.path.join(output_dir, save_name)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()

        print(f"[OK] 已保存: {save_path}")


def plot_loss_comparison(
    run_dirs: list[str],
    labels: list[str],
    key: str = "loss",
    save_path: str = "loss_comparison.png",
    dpi: int = 100,
):
    """
    对比多次训练的 loss 曲线（仅平滑后）。

    Args:
        run_dirs: 各次训练的 output_dir 列表。
        labels: 各次训练的标签。
        key: 要比较的指标，默认 "loss"。
        save_path: 输出图片路径。
        dpi: 图片分辨率。
    """
    if len(run_dirs) != len(labels):
        raise ValueError("run_dirs 和 labels 长度必须一致")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("需要 matplotlib，请先安装: pip install matplotlib")

    plt.figure(figsize=(10, 5))

    for run_dir, label in zip(run_dirs, labels):
        state_path = os.path.join(run_dir, "trainer_state.json")
        if not os.path.exists(state_path):
            print(f"[WARNING] 跳过 {label}: 未找到 trainer_state.json")
            continue

        with open(state_path, encoding="utf-8") as f:
            data = json.load(f)

        steps = []
        metrics = []
        for entry in data.get("log_history", []):
            if key in entry:
                steps.append(entry["step"])
                metrics.append(entry[key])

        if len(metrics) == 0:
            continue

        plt.plot(
            steps, smooth(metrics),
            linewidth=2, label=label,
        )

    plt.title(f"Training {key} Comparison")
    plt.xlabel("Step")
    plt.ylabel(key)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    print(f"[OK] 已保存对比图: {save_path}")


# ── CLI ────────────────────────────────────────────────────────────


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    output_dir = sys.argv[1]
    plot_loss(output_dir)


if __name__ == "__main__":
    main()
