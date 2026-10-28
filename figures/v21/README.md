# V21 六图稿的复现入口

对应论文《上下文感知转写与候选重排序的藏文缩略词还原方法》。此版本延续原 V21 的六图顺序和单栏宽度，使用修正后的冻结实验。代码算法、数据划分和训练结果未因稿件叙事调整而改变。

| 正文图号 | 稳定输出名 | 内容 | 尺寸（mm） |
|---|---|---|---|
| 1 | 不在代码包中再分发 | 2026-10-27批注修订的三种机制及规范展开示例 | 80 × 90 |
| 2 | `fig2_data_protocol` | 数量筛选、联结分组及四份划分职责 | 80 × 70.75 |
| 3 | `fig3_method_framework` | 上下文转写、编辑标注和独立校准重排序 | 80 × 86.58 |
| 4 | `fig4_main_results` | 主比较、配对结局、来源和 CER 降幅区间 | 80 × 117.32 |
| 5 | `fig5_calibration_sensitivity` | 固定 C=0.3 的校准集折外阈值曲线 | 80 × 99.82 |
| 6 | `fig6_error_analysis` | 编辑操作比较与三种缩写机制的定性结构关联 | 80 × 82.56 |

2026-10-27批注修订重新绘制了图1和图6。图1的三个藏文示例参照安世兴《藏文缩写字》（民族语文，1984(2):25—31），以HarfBuzz塑形后导出矢量字形，随作者稿单独提供；本代码ZIP不复制该图、具体藏文案例或字体文件。公开图6的复现无需HarfBuzz。图2—6可仅从此目录的聚合表复现，不需要原始语料、私有实验路径、模型权重或藏文案例文本。

从仓库根目录执行：

```powershell
python -m pip install -r requirements-figures.txt
python figures/v21/build_v21_figures.py --output-dir figures/generated/v21
```

默认读 `figures/v21/source_data/v21_aggregate.json`。使用独立聚合输入时可传 `--data-dir`；仅生成指定图可传 `--figures 3 4`。无输入数据的方法图检查：

```powershell
python figures/v21/build_v21_figures.py --method-only --output-dir figures/generated/v21_method
```

中文优先使用 Noto Serif SC、Noto Sans SC，回退至宋体／微软雅黑；数字优先使用 Times New Roman，缺少时使用 DejaVu Serif。字体由用户系统安装，字体文件不再分发。可变 Noto 字体会在生成目录固定到 400 字重，避免系统默认实例过细。

生成目录含纯矢量 PDF／SVG、600 dpi PNG、彩色及灰度预览、缺字／越界／刻度重叠检查，以及按输入数值导出的聚合 CSV。宽度为 80 mm，图2、3、5最小字号7.5 pt，图4、6局部为7 pt。生成内容和工作字体位于 `figures/generated/`，不纳入 Git 或发布 ZIP。

`source_data/v21_aggregate.json` 保存绘图所需的聚合计数、指标、校准操作点和证据 SHA256，省略逐条文本、记录索引、源文件绝对路径及模型参数。并列 CSV 便于手工核对；JSON 是默认入口的数值来源。参考→预测和预测→参考的插入／删除含义不同，图6统一采用预测→参考；矩阵由`qualitative_fig6`描述，仅为定性结构关联，不代表实测类型频数；双音节压缩、同字合并和字母替代是三种说明性机制，并非互斥标注类别。

图4的区间是固定种子42与固定划分下的联结组配对 Bootstrap，不能表示训练随机性或跨来源泛化。图5的参数由独立校准集选择，没有使用测试集调参。外部 TibNorm 使用额外公开规则库，资源条件不同。完整去神经控制和三种子结果仍见 [结果说明](../../docs/results.md) 与 [扩展图件](../README.md)。

统计图4—6按期刊作图规范统一对比编码：上下文为青绿圆点、编辑分支为蓝色三角、重排序为橙色菱形；条带增加斜线纹理，采用深色文字，提高灰度打印和小字号对比度。数值、坐标范围、阈值及区间含义保持不变。

图4(a)同时显示底轴CER与顶轴CA；比例定义CA=1−micro_CER，百分数刻度满足CA/%=100−CER/%，两轴是同一测量的严格互补换算，点旁数值为CER。图5(b)显示校准集折外整段还原准确率EM。CA与EM定义不同，详细公式见 [指标定义](../../docs/reproducibility.md)。更新图注在 `caption_updates_zh.json`；图4聚合CSV增加 `character_accuracy=1−cer`，既有CER和计数未改。

2026-10-27同步只改图6定性关系，S/I/D、CA/CER/EM、选择统计和模型算法未变。单独复现：`python figures/v21/build_v21_figures.py --figures 6 --output-dir figures/generated/v21_fig6`。三种机制的原始论述参见安世兴前述文献第25、26、28、29页；这里的方块表示还原时需检查的结构关系，不是对测试样本进行的新类型标注。
