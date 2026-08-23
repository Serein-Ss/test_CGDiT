"""Draw a temporary conceptual explainer for CrystalPIRL verification."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "assets" / "tmp" / "crystalpirl_paired_validation_explainer.png"

COLORS = {
    "base": "#72767E",
    "current": "#2B6F92",
    "candidate": "#D98C10",
    "accept": "#278C77",
    "reject": "#B94A48",
    "purple": "#8167A9",
    "ink": "#25313C",
    "muted": "#66727D",
    "panel": "#F7F8FA",
    "line": "#C8CED4",
}


def add_box(ax, x, y, w, h, text, color, *, fill_alpha=0.10, fontsize=9):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.25,
        edgecolor=color,
        facecolor=color,
        alpha=fill_alpha,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["ink"],
        transform=ax.transAxes,
        linespacing=1.35,
    )


def add_arrow(ax, start, end, color=None, connectionstyle="arc3"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.15,
            color=color or COLORS["muted"],
            connectionstyle=connectionstyle,
            transform=ax.transAxes,
        )
    )


def prepare_ax(ax, label):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.01, 0.01),
            0.98,
            0.98,
            boxstyle="round,pad=0.008,rounding_size=0.018",
            linewidth=0.8,
            edgecolor=COLORS["line"],
            facecolor=COLORS["panel"],
            transform=ax.transAxes,
        )
    )
    ax.text(0.035, 0.955, label, fontsize=12, fontweight="bold", va="top", transform=ax.transAxes)


def panel_policies(ax):
    prepare_ax(ax, "a")
    add_box(ax, 0.08, 0.68, 0.25, 0.16, "pi_0  冻结基础策略\n初始预训练模型", COLORS["base"])
    add_box(ax, 0.40, 0.68, 0.25, 0.16, "pi_k  当前可信策略\n最近一次通过验收", COLORS["current"])
    add_box(ax, 0.72, 0.68, 0.21, 0.16, "π′  候选策略\n一次 PPO/GRPO 更新", COLORS["candidate"])
    add_arrow(ax, (0.33, 0.76), (0.40, 0.76))
    add_arrow(ax, (0.65, 0.76), (0.72, 0.76))
    ax.text(0.365, 0.79, "初始化", fontsize=8, ha="center", color=COLORS["muted"], transform=ax.transAxes)
    ax.text(0.685, 0.79, "提议", fontsize=8, ha="center", color=COLORS["muted"], transform=ax.transAxes)
    add_box(ax, 0.29, 0.35, 0.42, 0.16, "验收门：局部锚点 + 绝对锚点\n+ 生成质量非退化", COLORS["purple"])
    add_arrow(ax, (0.82, 0.68), (0.67, 0.51), COLORS["candidate"])
    add_box(ax, 0.13, 0.10, 0.28, 0.13, "通过：pi_(k+1) ← pi′\n提交新 checkpoint", COLORS["accept"])
    add_box(ax, 0.59, 0.10, 0.28, 0.13, "失败：保持 pi_k\n拒绝或回滚", COLORS["reject"])
    add_arrow(ax, (0.42, 0.35), (0.29, 0.23), COLORS["accept"])
    add_arrow(ax, (0.58, 0.35), (0.73, 0.23), COLORS["reject"])


def panel_pairing(ax):
    prepare_ax(ax, "b")
    add_box(ax, 0.06, 0.70, 0.28, 0.16, "独立 probe 样本 j\n条件 c_j + 随机流 xi_j", COLORS["purple"])
    ax.text(0.20, 0.62, "同一个 j 内共享 xi_j；不同 j 使用不同随机流", ha="center", fontsize=8, color=COLORS["muted"], transform=ax.transAxes)
    policies = [(0.45, "pi_0", COLORS["base"]), (0.65, "pi_k", COLORS["current"]), (0.85, "pi′", COLORS["candidate"])]
    for x, label, color in policies:
        add_box(ax, x - 0.07, 0.72, 0.14, 0.12, label, color, fontsize=10)
        add_arrow(ax, (0.34, 0.78), (x - 0.07, 0.78), color)
        add_box(ax, x - 0.075, 0.43, 0.15, 0.13, f"配对结构\n奖励 R({label}, j)", color, fontsize=8.5)
        add_arrow(ax, (x, 0.72), (x, 0.56), color)
    add_box(ax, 0.13, 0.15, 0.34, 0.15, "局部配对差：D_local(j) = R′(j) − R_k(j)\n回答：这一步是否进步？", COLORS["current"], fontsize=8.5)
    add_box(ax, 0.55, 0.15, 0.34, 0.15, "绝对配对差：D_base(j) = R′(j) − R_0(j)\n回答：是否仍高于起点？", COLORS["base"], fontsize=8.5)
    ax.text(0.50, 0.055, "独立：probe 不参与这次梯度更新；配对：三种策略逐样本比较", ha="center", fontsize=8.5, color=COLORS["ink"], transform=ax.transAxes)


def panel_gate(ax):
    prepare_ax(ax, "c")
    rows = [
        (0.73, "局部收益", "LCB95(D_local) > 0", COLORS["current"]),
        (0.55, "绝对收益", "LCB95(D_base) > 0", COLORS["base"]),
        (0.37, "质量安全", "LCB95(D_V/S/U/D) ≥ −delta", COLORS["purple"]),
    ]
    for y, left, right, color in rows:
        add_box(ax, 0.07, y, 0.22, 0.11, left, color, fontsize=9)
        add_arrow(ax, (0.29, y + 0.055), (0.37, y + 0.055), color)
        add_box(ax, 0.37, y, 0.40, 0.11, right, color, fontsize=9)
    ax.text(0.84, 0.64, "全部满足", ha="center", fontsize=8.5, color=COLORS["accept"], transform=ax.transAxes)
    add_arrow(ax, (0.77, 0.60), (0.88, 0.49), COLORS["accept"])
    add_box(ax, 0.76, 0.31, 0.20, 0.15, "接受\n再经 holdout 复核", COLORS["accept"])
    add_box(ax, 0.08, 0.10, 0.25, 0.12, "均值为正但证据不足\n缩小步长并重新验收", COLORS["candidate"], fontsize=8)
    add_box(ax, 0.39, 0.10, 0.23, 0.12, "任一主门失败\n拒绝候选", COLORS["reject"], fontsize=8)
    add_box(ax, 0.68, 0.10, 0.23, 0.12, "holdout 失败\n回滚到 pi_k", COLORS["reject"], fontsize=8)
    ax.text(0.08, 0.29, "V 有效性  S 稳定性  U 唯一性  D 多样性；δ 为预注册容忍界", fontsize=8, color=COLORS["muted"], transform=ax.transAxes)


def panel_orbitpo(ax):
    prepare_ax(ax, "d")
    add_box(ax, 0.05, 0.72, 0.25, 0.15, "元素 A\n每个独立轨道计数一次\nD3PM 离散概率", COLORS["candidate"], fontsize=8.5)
    add_box(ax, 0.375, 0.72, 0.25, 0.15, "坐标 X\n轨道代表点的周期概率\n环面 wrapped normal", COLORS["current"], fontsize=8.5)
    add_box(ax, 0.70, 0.72, 0.25, 0.15, "晶格 L\n仅对称允许子空间\n高斯概率", COLORS["purple"], fontsize=8.5)
    add_box(ax, 0.25, 0.43, 0.50, 0.14, "联合对数概率\nlog π = log pA + log pX + log pL", COLORS["ink"], fontsize=9)
    for x in (0.175, 0.50, 0.825):
        add_arrow(ax, (x, 0.72), (0.50, 0.57))
    add_box(ax, 0.14, 0.15, 0.30, 0.13, "可重算 old/current log-prob\nimportance ratio、KL", COLORS["base"], fontsize=8.5)
    add_box(ax, 0.57, 0.15, 0.30, 0.13, "送入 PPO 或 GRPO\n产生候选更新 π′", COLORS["accept"], fontsize=8.5)
    add_arrow(ax, (0.39, 0.43), (0.29, 0.28))
    add_arrow(ax, (0.61, 0.43), (0.72, 0.28))
    ax.text(0.50, 0.055, "OrbitPO 是对称约化的概率基础；CrystalPIRL 是候选更新的验证闭环", ha="center", fontsize=8.5, color=COLORS["ink"], transform=ax.transAxes)


def main():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["WenQuanYi Micro Hei", "DejaVu Sans"],
            "font.size": 9,
            "axes.unicode_minus": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    panel_policies(axes[0, 0])
    panel_pairing(axes[0, 1])
    panel_gate(axes[1, 0])
    panel_orbitpo(axes[1, 1])
    fig.text(0.5, 0.012, "概念示意图；不包含实验结果", ha="center", fontsize=8, color=COLORS["muted"])
    fig.subplots_adjust(left=0.025, right=0.975, top=0.985, bottom=0.04, wspace=0.025, hspace=0.035)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
