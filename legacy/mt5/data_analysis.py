"""
数据分析与预处理脚本
运行: python data_analysis.py
"""

import os
import re
from collections import Counter
import matplotlib
matplotlib.use("Agg")  # 无GUI环境
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def analyze_data(source_file="data/Source.txt", restore_file="data/Restore.txt"):
    with open(source_file, "r", encoding="utf-8") as f:
        sources = [l.strip() for l in f if l.strip()]
    with open(restore_file, "r", encoding="utf-8") as f:
        targets = [l.strip() for l in f if l.strip()]

    assert len(sources) == len(targets), "行数不匹配！"

    print(f"{'='*50}")
    print(f"数据集分析报告")
    print(f"{'='*50}")
    print(f"总样本数: {len(sources)}")

    # 长度分析
    src_lens = [len(s) for s in sources]
    tgt_lens = [len(t) for t in targets]

    print(f"\n缩写文本（Source）:")
    print(f"  平均长度: {sum(src_lens)/len(src_lens):.1f} 字符")
    print(f"  最短: {min(src_lens)}  最长: {max(src_lens)}")
    print(f"  95%分位数: {sorted(src_lens)[int(len(src_lens)*0.95)]}")

    print(f"\n还原文本（Restore）:")
    print(f"  平均长度: {sum(tgt_lens)/len(tgt_lens):.1f} 字符")
    print(f"  最短: {min(tgt_lens)}  最长: {max(tgt_lens)}")
    print(f"  95%分位数: {sorted(tgt_lens)[int(len(tgt_lens)*0.95)]}")

    # 还原比例分析
    ratios = [tl/sl if sl > 0 else 1 for sl, tl in zip(src_lens, tgt_lens)]
    print(f"\n还原比例（还原长度/缩写长度）:")
    print(f"  平均: {sum(ratios)/len(ratios):.2f}x")
    print(f"  最大: {max(ratios):.2f}x")

    # 重复样本检查
    src_counter = Counter(sources)
    duplicates = {k: v for k, v in src_counter.items() if v > 1}
    print(f"\n重复缩写文本: {len(duplicates)} 条")
    if duplicates:
        for k, v in list(duplicates.items())[:5]:
            print(f"  '{k[:30]}' 出现 {v} 次")

    # 空样本检查
    empty_src = sum(1 for s in sources if not s)
    empty_tgt = sum(1 for t in targets if not t)
    print(f"\n空行数: Source={empty_src}, Restore={empty_tgt}")

    # ── 推荐 max_length 设置 ──
    p95_src = sorted(src_lens)[int(len(src_lens)*0.95)]
    p95_tgt = sorted(tgt_lens)[int(len(tgt_lens)*0.95)]
    print(f"\n{'─'*40}")
    print(f"推荐训练参数（覆盖95%样本）:")
    print(f"  max_source_length = {min(512, p95_src + 32)}")
    print(f"  max_target_length = {min(512, p95_tgt + 32)}")
    print(f"{'─'*40}")

    # ── 绘制长度分布图 ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(src_lens, bins=50, color="steelblue", alpha=0.7)
    axes[0].set_title("Source Length Distribution")
    axes[0].set_xlabel("Character Count")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(tgt_lens, bins=50, color="darkorange", alpha=0.7)
    axes[1].set_title("Restore Length Distribution")
    axes[1].set_xlabel("Character Count")

    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/data_analysis.png", dpi=150)
    print(f"\n长度分布图已保存至 output/data_analysis.png")

    # 输出样本示例
    print(f"\n{'─'*40}")
    print("数据样本示例（前5条）:")
    for i in range(min(5, len(sources))):
        print(f"  [{i+1}] 缩写: {sources[i]}")
        print(f"      还原: {targets[i]}")
        print()


def clean_data(source_file="data/Source.txt", restore_file="data/Restore.txt",
               out_src="data/Source_clean.txt", out_tgt="data/Restore_clean.txt"):
    """
    数据清洗：去除空行、去除首尾空格、去除明显错误样本
    """
    with open(source_file, "r", encoding="utf-8") as f:
        sources = [l.strip() for l in f]
    with open(restore_file, "r", encoding="utf-8") as f:
        targets = [l.strip() for l in f]

    clean_pairs = []
    skipped = 0
    for s, t in zip(sources, targets):
        # 过滤条件
        if not s or not t:              # 空行
            skipped += 1
            continue
        if len(s) > 512 or len(t) > 512:   # 过长
            skipped += 1
            continue
        clean_pairs.append((s, t))

    print(f"清洗完成: {len(clean_pairs)} 有效 | {skipped} 过滤")

    os.makedirs("data", exist_ok=True)
    with open(out_src, "w", encoding="utf-8") as f:
        f.write("\n".join(s for s, _ in clean_pairs))
    with open(out_tgt, "w", encoding="utf-8") as f:
        f.write("\n".join(t for _, t in clean_pairs))

    print(f"清洗后数据已保存至 {out_src} 和 {out_tgt}")


if __name__ == "__main__":
    analyze_data()
