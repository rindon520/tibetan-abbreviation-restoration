# 藏文缩略词还原：扩展核查图件

当前正文沿原 V21 六图顺序，题目为《上下文感知转写与候选重排序的藏文缩略词还原方法》。正文图2—6的复现入口、聚合数据和图号映射见 [v21/README.md](v21/README.md)。本目录原绘图入口保留为扩展核查材料，其旧编号不用于当前正文。

本目录绘图脚本按跨栏宽度 **160 mm**、正文标签与刻度最小 **8 pt** 导出纯矢量 PDF、SVG 及 **300 dpi PNG**，并生成彩色与灰度预览。此 GitHub 包只保留绘图代码与聚合源数据，不包含 PDF、PNG、SVG 或原始文本。向 Word 插入重生成的主图时使用 160 mm 宽度，避免再次缩小。

## 图件与图注

文件名作为稳定的复现标识。下表记录上一轮跨栏组合的编号，供历史结果与扩展核查使用；当前正文编号统一采用 V21 六图映射。

| 历史组合编号 | 脚本输出文件名 | 内容 | 尺寸（mm） |
|---|---|---|---|
| 图 1 | `fig1_method` | 双路候选生成与整段选择流程 | 160 × 108.712 |
| 图 2 | `fig2_test_comparison` | 种子 42 主方法与基线比较 | 160 × 100.33 |
| 图 3 | `fig4_context_postprocessing` | 同一上下文转写模型的逐步后处理 | 160 × 55.372 |
| 图 4 | `fig6_neural_role` | 后验神经角色与完整去神经控制 | 160 × 100.33 |
| 补充或审稿核查 | `fig0_data_protocol` | 数据过滤、联结分组与四份数据职责 | 160 × 85.09 |
| 补充或审稿核查 | `fig3_dev_parameters` | 开发集参数敏感性 | 160 × 81.28 |
| 补充或审稿核查 | `fig5_selector_analysis` | 候选互补及选择改善、损害和持平计数 | 160 × 74.93 |
| 补充图 S1 | `figS1_training_seeds` | 42、43、44 神经种子的独立运行点 | 160 × 69.85 |

## 数据依据

`source_data/` 仅包含聚合指标与分组计数，不包含原始藏文内容。`summary.json` 列出每一输入证据文件的相对路径与 SHA-256。所有最终经验图使用修正后的联结分组划分：训练 6,730、开发 962、独立校准 962、测试 961。生成分支的参数由开发集选择，选择器由校准集选择；最终测试仅用于冻结后的评估。

主比较图列出未预训练、贪心解码的字符 Transformer。TibNorm 使用其外部发布规则库并执行更广泛的完整规范化，资源条件不同，因此以 † 单独注明；该比较不能证明相同训练资源下的优势。TibNorm 原始来源为 [公开仓库](https://github.com/Divergent-Discourses/TibNorm)，固定提交 `d296f6566da1308f7c14180080acc43ace8dd68a`，[存档 DOI](https://doi.org/10.5281/zenodo.10806456)。外部源码、规则表与原始语料不在此图件包中。

点表示主运行的汇总错误率，没有通过误差棒伪装训练随机性的估计。藏文单元错误率按评估代码包含分隔符单元，不等同于词错误率。`fig5_selector_analysis` 中候选优劣来自事后参考答案对比，测试推理时没有使用参考答案。

扩展图 `fig6_neural_role`单列后验诊断：纯神经标签输出弱于纯编辑对齐规则输出；完整去神经双路系统与主系统仅差 2 个字符编辑。该探索性分析不替换预定义主模型，当前证据不能支持神经分支具有独立贡献。补充图 S1 在固定数据划分、固定上下文分支、固定校准折分下报告神经种子 42、43、44；每次重新拟合选择器，没有按测试表现挑选种子。

## 重新生成

以下命令均从仓库根目录执行。绘图不会训练模型，也不需要原始文本；安装绘图依赖后可直接使用包内聚合表：

```powershell
python -m pip install -r requirements-figures.txt
python figures/scripts/build_submission_figures.py --data-dir figures/source_data --output-dir figures/generated --split-manifest figures/source_data/split_summary.json
```

仅生成方法图：

```powershell
python figures/scripts/build_submission_figures.py --method-only --output-dir figures/generated/method_only
```

安装至少一种脚本可识别的中文系统字体：SimSun、Noto Serif SC、Source Han Serif SC、Noto Sans SC 或 Microsoft YaHei。Windows 优先使用宋体与 Times New Roman；无 Times New Roman 时回退至 DejaVu Serif。字体未包含在代码包中，缺少中文字体时脚本会明确报错。

有权访问冻结实验输出时，可另外重新导出聚合数据。生成到独立目录，不覆盖公开源表：

```powershell
python figures/scripts/prepare_corrected_figure_data.py --repo-root . --run-dir experiments/grouped_seed42 --out-dir figures/generated/aggregate
python figures/scripts/prepare_auxiliary_figure_data.py --repo-root . --run-dir experiments/grouped_seed42 --replications-root experiments --out-dir figures/generated/aggregate
python figures/scripts/build_submission_figures.py --data-dir figures/generated/aggregate --output-dir figures/generated/recomputed --split-manifest figures/source_data/split_summary.json
```

补充种子目录默认匹配本仓库的 `grouped_model_seed43` 和 `grouped_model_seed44`；如使用旧目录名，可显式传 `--replication-pattern 'runs_corrected_seed{seed}'`。数据准备脚本只读实验文本来汇总计数，不向图件源表复制逐条文本。输入文件路径按相对或逻辑路径记录，并保存 SHA256。

分组协议图接受真实实验的 `split_manifest.json`；只有公开聚合信息时使用包内 `figures/source_data/split_summary.json`。重新执行整个数据准备流程时，以新实验的固定划分摘要为准。

## 质量核验

重新绘图后，输出目录中的 `figure_quality_report.json` 记录尺寸、最小字号、缺字、越界与刻度重叠检查。程序检查通过后仍需检查彩色和灰度预览；更换字体或绘图版本时不能跳过视觉复核。投稿稿件的视觉审查和矢量核验记录保留在独立文稿目录，不打包为这里的已渲染图件。

历史随机句段划分存在同原始行跨训练与测试，不作为当前经验图的数据依据。`figures/generated/` 不纳入 Git 或发布 ZIP。
