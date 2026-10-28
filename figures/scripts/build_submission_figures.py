"""Deterministic journal figures, rendered at their final 160 mm width.

Usage: python build_submission_figures.py --output-dir PATH --method-only
       python build_submission_figures.py --output-dir PATH --data-dir PATH

The plotter reads aggregate source tables. It does not alter source results,
train a model, invent uncertainty bars, or infer statistical significance.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd
from PIL import Image

WIDTH = 160 / 25.4
BLUE = "#0072B2"
ORANGE = "#D55E00"
GRAY = "#7C858C"
INK = "#20272D"
LIGHT = "#E2E7EB"
BLUE_PALE = "#EAF2F8"
ORANGE_PALE = "#FAEFE8"
REPORTS = []


def setup():
    available = {font.name for font in font_manager.fontManager.ttflist}
    chinese = next((f for f in ["SimSun", "Noto Serif SC", "Source Han Serif SC", "Noto Sans SC", "Microsoft YaHei"] if f in available), None)
    if not chinese:
        raise RuntimeError("Install a Chinese font before rendering: SimSun or Noto Serif SC is preferred")
    latin = "Times New Roman" if "Times New Roman" in available else "DejaVu Serif"
    plt.rcParams.update({
        "font.family": [latin, chinese],
        "font.size": 8.5,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "axes.linewidth": 0.65,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "path",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "mathtext.fontset": "custom",
        "mathtext.rm": latin,
        "mathtext.it": f"{latin}:italic",
        "mathtext.bf": f"{latin}:bold",
    })
    return {"latin": latin, "chinese": chinese}


def clean(ax, grid=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5, width=0.6, pad=3)
    if grid:
        ax.set_axisbelow(True)
        ax.grid(axis=grid, color=LIGHT, lw=0.55, linestyle=(0, (2, 3)))


def panel(fig, x, y, label, title):
    fig.text(x, y, label, weight="bold", fontsize=9.5, va="bottom")
    fig.text(x + 0.045, y, title, fontsize=9.0, va="bottom")


def audit(fig, name):
    """Check missing glyphs, text bounds, tick overlaps and minimum font size."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    issues = [str(w.message) for w in caught if "missing from font" in str(w.message)]
    bounds = fig.bbox
    from matplotlib.text import Text
    sizes = []
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text():
            continue
        sizes.append(text.get_fontsize())
        bbox = text.get_window_extent(renderer)
        if bbox.x0 < bounds.x0 - 1 or bbox.y0 < bounds.y0 - 1 or bbox.x1 > bounds.x1 + 1 or bbox.y1 > bounds.y1 + 1:
            issues.append(f"Text outside canvas: {text.get_text()!r}")
    for ax in fig.axes:
        for axis in [ax.xaxis, ax.yaxis]:
            labels = [t for t in axis.get_ticklabels() if t.get_visible() and t.get_text()]
            boxes = [t.get_window_extent(renderer) for t in labels]
            for left, right in zip(boxes, boxes[1:]):
                if left.overlaps(right):
                    issues.append("Adjacent tick labels overlap")
    REPORTS.append({"figure": name, "width_mm": fig.get_figwidth() * 25.4, "height_mm": fig.get_figheight() * 25.4, "minimum_font_pt": min(sizes), "issues": sorted(set(issues))})
    if issues:
        raise RuntimeError(f"{name}: " + "; ".join(sorted(set(issues))))


def save(fig, out, name, previews_only):
    audit(fig, name)
    preview = out / "previews"
    preview.mkdir(parents=True, exist_ok=True)
    fig.savefig(preview / f"{name}.png", dpi=180)
    with Image.open(preview / f"{name}.png") as image:
        image.convert("L").save(preview / f"{name}_grayscale.png", dpi=(180, 180))
    if not previews_only:
        fig.savefig(out / f"{name}.pdf")
        fig.savefig(out / f"{name}.svg")
        fig.savefig(out / f"{name}.png", dpi=300)
    plt.close(fig)


def card(ax, x, y, w, h, title, detail="", fill="white", border=GRAY, fontsize=8.5):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=border, lw=0.75))
    if detail:
        ax.text(x + w / 2, y + h * 0.73, title, ha="center", va="center", fontsize=fontsize)
        ax.text(x + w / 2, y + h * 0.36, detail, ha="center", va="center", fontsize=8.0, linespacing=1.45)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fontsize, linespacing=1.4)


def arrow(ax, x1, y1, x2, y2, color=INK, style="-", conn="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8, color=color, lw=0.8, linestyle=style, connectionstyle=conn))


def method_figure(out, previews_only):
    fig = plt.figure(figsize=(WIDTH, 4.28))
    ax = fig.add_axes([0.025, 0.03, 0.95, 0.95])
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8.6)
    ax.axis("off")
    ax.text(0.05, 8.32, "a  双路候选生成", fontsize=9.5, va="center")
    card(ax, 0.1, 5.4, 1.15, 1.1, "缩略形式\n输入 x", fill="#F4F6F7")
    # The rule branch and neural branch form two complete segment candidates.
    card(ax, 1.8, 6.2, 4.15, 1.48, "上下文感知字符转写", "三元上下文 → 左邻 → 右邻 → 字符映射\n训练集频次统计；支持度与频率阈值回退", BLUE_PALE, BLUE)
    card(ax, 1.8, 4.23, 4.15, 1.48, "字符级 Transformer 编辑标注", "复制／删除／输出字符串标签\n低置信度字符回退至统计规则", ORANGE_PALE, ORANGE)
    card(ax, 6.4, 6.2, 3.12, 1.48, "分支后处理", "句末符号修正\n训练词表启发式重切分", BLUE_PALE, BLUE)
    card(ax, 6.4, 4.23, 3.12, 1.48, "分支后处理", "句末符号修正\n训练词表启发式重切分", ORANGE_PALE, ORANGE)
    card(ax, 10.0, 6.2, 1.75, 1.48, "完整片段\n候选 yC", border=BLUE)
    card(ax, 10.0, 4.23, 1.75, 1.48, "完整片段\n候选 yE", border=ORANGE)
    ax.plot([1.25, 1.52, 1.52], [5.95, 5.95, 6.94], color=INK, lw=0.8)
    ax.plot([1.52, 1.52], [5.95, 4.97], color=INK, lw=0.8)
    arrow(ax, 1.52, 6.94, 1.8, 6.94)
    arrow(ax, 1.52, 4.97, 1.8, 4.97)
    for y in [6.94, 4.97]:
        arrow(ax, 5.95, y, 6.4, y)
        arrow(ax, 9.52, y, 10.0, y)
    ax.plot([0.08, 11.9], [3.82, 3.82], lw=0.65, color=LIGHT)
    ax.text(0.05, 3.46, "b  成对特征与整段候选选择", fontsize=9.5, va="center")
    card(ax, 0.1, 1.2, 1.55, 1.55, "输入与候选\nx，yC，yE", fill="#F4F6F7")
    card(ax, 2.05, 1.2, 4.1, 1.55, "28 维成对特征", "字符 n-gram 与差值（12 维）\n距离／长度／覆盖／候选差异（10 维）\n置信／回退／后处理／源长（6 维）", fill="#F4F6F7")
    card(ax, 6.55, 1.2, 2.18, 1.55, "标准化＋逻辑回归", "p：编辑候选\n相对优于转写候选", border=INK)
    card(ax, 9.12, 1.2, 2.63, 1.55, "输出一个完整候选", "p ≥ τ：选择 yE\np < τ：选择 yC", border=INK)
    arrow(ax, 1.65, 1.98, 2.05, 1.98)
    arrow(ax, 6.15, 1.98, 6.55, 1.98)
    arrow(ax, 8.73, 1.98, 9.12, 1.98)
    ax.text(7.65, 0.67, "独立校准集：分组五折 OOF 搜索 C 与 τ\n选定后在全校准集拟合选择器", ha="center", va="center", fontsize=8.0, linespacing=1.4)
    arrow(ax, 7.65, 0.99, 7.65, 1.18, style="--")
    ax.text(0.1, 0.12, "统计规则、音节词表与字符语言模型均由训练集构建；后处理发生在候选选择之前。", fontsize=8.0, va="bottom")
    save(fig, out, "fig1_method", previews_only)


def protocol_note(summary):
    return f"{summary['protocol_label']}；测试集 n = {summary['n_test']}；点为预定义主运行的结果。"


def protocol_figure(out, manifest_path, previews_only):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fig = plt.figure(figsize=(WIDTH, 3.35))
    ax = fig.add_axes([.025, .03, .95, .95])
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.3)
    ax.axis("off")
    ax.text(.1, 6.95, "分组划分与训练、开发、校准、测试职责", fontsize=9.5, va="center")
    card(ax, .1, 4.92, 2.7, 1.35, f"平行行对 {manifest['raw_line_pairs']:,}", "缩略形式／规范还原\n保持原始字符编码", fill="#F4F6F7")
    card(ax, 3.3, 4.92, 3.5, 1.35, f"保留片段 {manifest['trusted_segments']:,}", f"按句末符号切分并匹配\n排除片段数不匹配行对 {manifest['rejected_line_pairs']:,}", fill="#F4F6F7")
    card(ax, 7.3, 4.92, 4.5, 1.35, f"联结分组 {manifest['connected_groups']:,}", "同一原始行或相同 NFC 源文本\n通过连通关系联结；整组划分", BLUE_PALE, BLUE)
    arrow(ax, 2.8, 5.6, 3.3, 5.6)
    arrow(ax, 6.8, 5.6, 7.3, 5.6)
    positions = [.1, 3.1, 6.1, 9.1]
    roles = [
        ("train", "训练集", "统计映射、词表与语言模型\n神经编辑标注器权重"),
        ("dev", "开发集", "候选生成参数与后处理参数\n神经早停与回退阈值"),
        ("calibration", "校准集", "分组五折 OOF 搜索\n拟合整段候选选择器"),
        ("test", "测试集", "所有参数冻结后评估\n不参与训练或参数选择"),
    ]
    ax.plot([1.45, 10.45], [4.1, 4.1], color=INK, lw=.75)
    ax.plot([9.55, 9.55], [4.92, 4.1], color=INK, lw=.75)
    for x, (key, title, role) in zip(positions, roles):
        spec = manifest['splits'][key]
        fill = BLUE_PALE if key == "train" else "#F4F6F7" if key != "test" else ORANGE_PALE
        edge = BLUE if key == "train" else GRAY if key != "test" else ORANGE
        card(ax, x, 1.55, 2.7, 2.0, title, f"{spec['samples']:,} 片段／{spec['groups']:,} 组\n{role}", fill, edge)
        arrow(ax, x + 1.35, 4.1, x + 1.35, 3.55)
    ax.text(.1, .81, "跨划分原始行与 NFC 源文本重叠均为 0；NFC 仅用于分组，评估文本保持原样。", fontsize=8.0)
    ax.text(.1, .24, "划分种子固定为 42；片段数约为 70%／10%／10%／10%，保持联结组完整。", fontsize=8.0)
    save(fig, out, "fig0_data_protocol", previews_only)


def main_comparison(out, data, summary, previews_only):
    values = pd.read_csv(data / "main_comparison.csv")
    fig = plt.figure(figsize=(WIDTH, 3.95 if len(values) > 7 else 3.28))
    ax1 = fig.add_axes([0.28, 0.22, 0.27, 0.64])
    ax2 = fig.add_axes([0.675, 0.22, 0.28, 0.64])
    panel(fig, 0.04, 0.90, "a", "字符错误率 CER（%） ↓")
    panel(fig, 0.58, 0.90, "b", "藏文单元错误率（%） ↓")
    y = np.arange(len(values))[::-1]
    for ax, field, max_x, numeric_pad in [(ax1, "micro_cer", 70 if len(values) > 7 else 40, 1.2), (ax2, "micro_tibetan_unit_error_rate", 110 if values.micro_tibetan_unit_error_rate.max() > .75 else 75, 1.8)]:
        for i, row in enumerate(values.itertuples(index=False)):
            color = ORANGE if row.method_id == "selector" else BLUE if row.method_id == "contextual" else GRAY
            marker = "D" if row.method_id == "selector" else "s" if row.method_id == "contextual" else "^" if row.method_id == "tibnorm" else "o"
            val = getattr(row, field) * 100
            ax.scatter(val, y[i], c=color, marker=marker, s=23, zorder=3)
            ax.text(val + numeric_pad, y[i], f"{val:.2f}", va="center", ha="left", fontsize=8.0)
        ax.set_xlim(0, max_x)
        ax.set_ylim(-0.45, len(values) - 0.55)
        ax.set_yticks(y)
        ax.tick_params(axis="y", length=0)
        clean(ax, "x")
    ax1.set_yticklabels(values.label)
    ax2.set_yticklabels([])
    ax1.set_xticks([0, 20, 40, 60] if len(values) > 7 else [0, 10, 20, 30, 40])
    ax2.set_xticks([0, 25, 50, 75, 100] if ax2.get_xlim()[1] > 100 else [0, 25, 50, 75])
    fig.text(0.04, 0.100, protocol_note(summary), fontsize=8.0, va="bottom")
    if "tibnorm" in values.method_id.tolist():
        fig.text(0.04, 0.057, "† TibNorm 使用外部发布规则库；其完整规范化任务范围更广，资源条件另列。", fontsize=8.0, va="bottom")
        fig.text(0.04, 0.014, "字符 Transformer 未预训练、采用贪心解码；藏文单元含分隔符，不等同于词错误率。", fontsize=8.0, va="bottom")
    save(fig, out, "fig2_test_comparison", previews_only)


def parameter_figure(out, data, summary, previews_only):
    context = pd.read_csv(data / "context_tuning.csv")
    edit = pd.read_csv(data / "edit_tuning.csv")
    context = context[context.max_edit_distance == summary["context_edit_distance"]]
    matrix = context.pivot(index="min_count", columns="min_confidence", values="valid_micro_cer") * 100
    fig = plt.figure(figsize=(WIDTH, 3.2))
    ax1 = fig.add_axes([0.10, 0.24, 0.325, 0.56])
    ax2 = fig.add_axes([0.62, 0.24, 0.33, 0.56])
    panel(fig, 0.04, 0.89, "a", "上下文转写：支持度与频率阈值")
    panel(fig, 0.55, 0.89, "b", "编辑标注：神经标签回退阈值")
    image = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=matrix.values.min(), vmax=matrix.values.max()), cmap="cividis")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax1.add_patch(Rectangle((col - .5, row - .5), 1, 1, facecolor=image.to_rgba(matrix.values[row, col]), edgecolor="none"))
    ax1.set_xlim(-.5, matrix.shape[1] - .5)
    ax1.set_ylim(matrix.shape[0] - .5, -.5)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            val = matrix.values[row, col]
            color = "white" if image.norm(val) < 0.42 else INK
            ax1.text(col, row, f"{val:.2f}", ha="center", va="center", fontsize=8.5, color=color)
    row_index = matrix.index.get_loc(summary["context_min_count"])
    col_index = matrix.columns.get_loc(summary["context_min_confidence"])
    ax1.add_patch(Rectangle((col_index - .49, row_index - .49), .98, .98, fill=False, lw=1.2, edgecolor=ORANGE))
    ax1.set_xticks(range(len(matrix.columns)), [f"{val:.1f}" for val in matrix.columns])
    ax1.set_yticks(range(len(matrix.index)), [str(val) for val in matrix.index])
    ax1.set_xlabel("最小输出频率占比")
    ax1.set_ylabel("最小上下文支持度")
    ax1.tick_params(length=0)
    cax = fig.add_axes([0.445, 0.24, 0.018, 0.56])
    cbar = fig.colorbar(image, cax=cax)
    cbar.solids.set_rasterized(False)
    cbar.set_label("开发 CER（%）", fontsize=8.0, labelpad=5)
    cbar.ax.tick_params(labelsize=8.0, length=2)
    for distance, marker, linestyle, color in [(0, "o", "--", GRAY), (1, "s", "-", BLUE)]:
        subset = edit[edit.max_edit_distance == distance].sort_values("confidence_threshold")
        ax2.plot(subset.confidence_threshold, subset.micro_cer * 100, marker=marker, ms=3.5, lw=0.9, linestyle=linestyle, color=color, label=f"音节校正 d = {distance}")
    chosen = edit[(edit.confidence_threshold == summary["edit_threshold"]) & (edit.max_edit_distance == summary["edit_distance"])].iloc[0]
    ax2.scatter([chosen.confidence_threshold], [chosen.micro_cer * 100], s=50, marker="s", facecolors="none", edgecolors=ORANGE, lw=1.2, zorder=5)
    ax2.set_xlim(-0.03, 0.85)
    ax2.set_ylim(0, 30)
    ax2.set_xticks([0, .2, .4, .6, .8])
    ax2.set_yticks([0, 10, 20, 30])
    ax2.set_xlabel("神经标签回退阈值")
    ax2.set_ylabel("开发 CER（%）")
    ax2.legend(loc="lower left", frameon=False, handlelength=2.1, borderaxespad=0.2)
    clean(ax2, "y")
    fig.text(0.04, 0.045, f"开发集 n = {summary['n_valid']}；橙色边框标示选定设置；d 为词表重切分允许的最大编辑距离。", fontsize=8.0)
    fig.text(0.04, 0.010, summary["protocol_label"], fontsize=8.0)
    save(fig, out, "fig3_dev_parameters", previews_only)


def component_figure(out, data, summary, previews_only):
    values = pd.read_csv(data / "context_components.csv")
    fig = plt.figure(figsize=(WIDTH, 2.18))
    ax = fig.add_axes([0.29, 0.28, 0.655, 0.52])
    panel(fig, 0.04, 0.89, "a", "上下文转写分支的后处理贡献（字符错误率，%） ↓")
    ys = np.arange(len(values))[::-1]
    for y, row in zip(ys, values.itertuples(index=False)):
        val = row.micro_cer * 100
        ax.scatter(val, y, s=26, color=BLUE, marker="s")
        ax.text(val + .55, y, f"{val:.2f}", fontsize=8.5, va="center")
    ax.set_yticks(ys, values.label)
    ax.set_xlim(0, max(30, values.micro_cer.max() * 100 + 3))
    ax.set_xticks(np.arange(0, ax.get_xlim()[1] + .1, 5))
    ax.set_ylim(-.5, 2.5)
    ax.tick_params(axis="y", length=0)
    clean(ax, "x")
    fig.text(0.04, 0.075, "三种输出来自同一已训练转写模型；后处理按实际执行顺序逐步启用。", fontsize=8.0)
    fig.text(0.04, 0.020, f"{summary['protocol_label']}；测试集 n = {summary['n_test']}。", fontsize=8.0)
    save(fig, out, "fig4_context_postprocessing", previews_only)


def selector_figure(out, data, summary, previews_only):
    table = pd.read_csv(data / "selection_contingency.csv")
    outcomes = pd.read_csv(data / "selection_outcomes.csv")
    rows = ["编辑候选更优", "两候选等误差", "上下文候选更优"]
    columns = ["编辑候选", "上下文候选"]
    fig = plt.figure(figsize=(WIDTH, 2.95))
    ax1 = fig.add_axes([0.04, 0.245, 0.455, 0.56])
    ax1.axis("off")
    panel(fig, 0.04, 0.89, "a", "候选优劣与最终选择（片段数）")
    panel(fig, 0.555, 0.89, "b", "相对上下文分支的结局")
    matrix = [[int(table[(table.candidate_relation == row) & (table.selected_model == col)].iloc[0]["count"]) for col in columns] for row in rows]
    cell_rows = [[row] + counts for row, counts in zip(rows, matrix)]
    cells = ax1.table(cellText=cell_rows, colLabels=["编辑距离比较", "选编辑", "选转写"], cellLoc="center", colLoc="center", colWidths=[.5, .25, .25], bbox=[0, 0, 1, 1])
    cells.auto_set_font_size(False)
    cells.set_fontsize(8.5)
    for (row, col), cell in cells.get_celld().items():
        cell.set_edgecolor(LIGHT)
        cell.set_linewidth(.6)
        if row == 0:
            cell.set_facecolor("#F1F4F6")
        else:
            cell.set_facecolor("white")
    # Correct/wrong choices are outlined, with numerical annotation preserving meaning in grayscale.
    cells[1, 1].set_edgecolor(BLUE)
    cells[1, 1].set_linewidth(1.2)
    cells[3, 1].set_edgecolor(ORANGE)
    cells[3, 1].set_linewidth(1.2)
    ax2 = fig.add_axes([0.635, 0.245, 0.315, 0.56])
    counts = outcomes["count"].tolist()
    ys = np.arange(3)[::-1]
    colors = [BLUE, ORANGE, "#C7CDD2"]
    hatches = ["", "///", "..."]
    for y, value, color, hatch in zip(ys, counts, colors, hatches):
        ax2.barh(y, value, height=.50, color=color, edgecolor=INK, lw=.45, hatch=hatch)
        ax2.text(value + summary["n_test"] * .025, y, str(value), va="center", fontsize=8.5)
    ax2.set_yticks(ys, outcomes.outcome)
    ax2.set_xlim(0, summary["n_test"] * 1.075)
    ax2.set_ylim(-.5, 2.5)
    ax2.set_xlabel("测试片段数")
    ax2.set_xticks(np.linspace(0, summary["n_test"], 3).astype(int))
    ax2.tick_params(axis="y", length=0)
    clean(ax2, "x")
    delta_pp = summary["absolute_cer_reduction_vs_contextual"] * 100
    fig.text(0.04, 0.11, f"选择编辑候选 {summary['selector_selected_edit']} / {summary['n_test']} 段；总体 CER 相对上下文分支降低 {delta_pp:.2f} 个百分点。", fontsize=8.0)
    fig.text(0.04, 0.045, summary["protocol_label"] + "；候选优劣仅用于离线分析，参考答案不参与测试时选择。", fontsize=8.0)
    save(fig, out, "fig5_selector_analysis", previews_only)


def neural_role_figure(out, data, summary, previews_only):
    values = pd.read_csv(data / "neural_role.csv")
    full = values[values.role == "selector"]
    has_full = len(full) > 0
    fig = plt.figure(figsize=(WIDTH, 3.95 if has_full else 2.30))
    ax = fig.add_axes([.31, .61, .63, .25] if has_full else [.31, .30, .63, .48])
    if has_full:
        panel(fig, .04, .90, "a", "后验诊断：编辑分支固定端点（CER，%） ↓")
    else:
        fig.text(.04, .90, "后验诊断：神经标签与统计规则的作用（CER，%） ↓", fontsize=9.0)
    def draw(ax, part, precision):
        ys = np.arange(len(part))[::-1]
        for y, row in zip(ys, part.itertuples(index=False)):
            rule = row.method_id in ["pure_rules", "no_neural_selector"]
            marker = "s" if rule else "D" if row.method_id == "main_neural_selector" else "o"
            color = BLUE if rule else ORANGE if row.method_id == "main_neural_selector" else GRAY
            ax.scatter(row.micro_cer * 100, y, s=28, marker=marker, color=color)
            ax.text(row.micro_cer * 100 + .65, y, f"{row.micro_cer * 100:.{precision}f}", va="center", fontsize=8.5)
        ax.set_yticks(ys, part.label)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-.5, len(part) - .5)
        ax.set_xlim(0, 35)
        ax.set_xticks([0, 10, 20, 30])
        clean(ax, "x")
    draw(ax, values[values.role == "edit_branch"], 2)
    if has_full:
        panel(fig, .04, .49, "b", "后验完整消融：候选选择系统（CER，%） ↓")
        draw(fig.add_axes([.31, .23, .63, .20]), full, 4)
    detail = summary["neural_diagnostics"]
    fig.text(.04, .105, f"控制共享训练资源，纯神经复用固定检查点；预定义编辑分支有 {detail['fallback_fraction'] * 100:.2f}% 源字符回退到规则。", fontsize=8.0)
    if has_full:
        difference = abs(summary["no_neural_selector"]["edit_distance_difference"])
        fig.text(.04, .045, f"完整系统相差 {difference} 个字符编辑；该后验消融不用于更换主系统，未证明神经分支的独立贡献。", fontsize=8.0)
    else:
        fig.text(.04, .035, "纯神经与纯规则为固定端点的后验控制；不据此更换主系统、调整测试参数或重训模型。", fontsize=8.0)
    save(fig, out, "fig6_neural_role", previews_only)


def seed_figure(out, data, summary, previews_only):
    values = pd.read_csv(data / "seed_replications.csv")
    seeds = sorted(values.model_seed.unique())
    fig = plt.figure(figsize=(WIDTH, 2.75))
    axes = [fig.add_axes([.16, .29, .34, .52]), fig.add_axes([.61, .29, .34, .52])]
    panel(fig, .04, .90, "a", "编辑标注分支 CER（%） ↓")
    panel(fig, .55, .90, "b", "双路候选选择 CER（%） ↓")
    for ax, method, color in zip(axes, ["edit_tagger", "hybrid"], [GRAY, ORANGE]):
        subset = values[values.method_id == method].sort_values("model_seed")
        for y, row, marker in zip(np.arange(len(seeds))[::-1], subset.itertuples(index=False), ["o", "s", "^"]):
            ax.scatter(row.micro_cer * 100, y, marker=marker, s=27, color=color)
            ax.text(row.micro_cer * 100 + .7, y, f"{row.micro_cer * 100:.2f}", fontsize=8.5, va="center")
        ax.set_yticks(np.arange(len(seeds))[::-1], [f"种子 {seed}" for seed in seeds])
        ax.set_ylim(-.5, len(seeds) - .5)
        ax.set_xlim(0, 35)
        ax.set_xticks([0, 10, 20, 30])
        ax.tick_params(axis="y", length=0)
        clean(ax, "x")
    axes[1].set_yticklabels([])
    fig.text(.04, .14, "数据划分、上下文转写状态与校准折分固定；神经种子 42、43、44 的每次运行均单独报告。", fontsize=8.0)
    fig.text(.04, .055, "每次独立训练编辑标注器并拟合选择器；主运行固定为 42，不按测试结果挑选种子。", fontsize=8.0)
    save(fig, out, "figS1_training_seeds", previews_only)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method-only", action="store_true")
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--previews-only", action="store_true")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    fonts = setup()
    method_figure(out, args.previews_only)
    if args.split_manifest:
        protocol_figure(out, args.split_manifest.resolve(), args.previews_only)
    if not args.method_only:
        if not args.data_dir:
            parser.error("--data-dir is required for empirical figures")
        data = args.data_dir.resolve()
        summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
        main_comparison(out, data, summary, args.previews_only)
        parameter_figure(out, data, summary, args.previews_only)
        component_figure(out, data, summary, args.previews_only)
        selector_figure(out, data, summary, args.previews_only)
        if (data / "neural_role.csv").exists():
            neural_role_figure(out, data, summary, args.previews_only)
        if (data / "seed_replications.csv").exists():
            seed_figure(out, data, summary, args.previews_only)
    report = {"font_configuration": fonts, "figure_checks": REPORTS, "raster_dpi": 300, "vector_formats": ["PDF", "SVG"], "visual_review_required": True}
    (out / "figure_quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
