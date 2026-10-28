# 可复现性协议

## 任务与版本

任务是配对文本上的藏文缩略词还原。数据准备依赖参考文本句末符来过滤可配对句段；实验没有测量 OCR、自动识别原始图像、或无参考文本的端到端句段检测性能。

正式入口是 `scripts/run_grouped_experiment.py`。根目录研究脚本保留原接口与历史默认值，不能以这些默认随机划分替代正式入口。source_manifest.json 记录上传包每个研究脚本的源文件哈希、包内哈希和修改目的；每次正式运行另存环境、代码哈希、语料哈希、分组及样本索引。

## 固定四个集合

每个原始行是最低分组单位。相同源文本经 NFC 规范化构成连接边，用并查集把相关原始行连接起来，得到连通分组。同一组完整地放入一个集合。NFC 只用于分组，训练和评分保留原始码点。

按 seed 42 打乱组，近似按句段数分配 70% train、10% dev、10% calibration、10% test。组不能切开，因此精确样本数可能略偏离比例。当前原始文件审计得到 10,820 个配对原始行，经句段数量一致过滤得到 9,615 个句段；seed 42 修正划分为 train 6,730、dev 962、calibration 962、test 961。具体样本清单以当前实验 manifest 为准，不包含在公开代码中。

`corpus/split_manifest.json` 保存源/参考文件 SHA256、records.jsonl SHA256、各集合索引与组清单。所有阶段读取同一文件并校验完整性、互斥性和候选逐行对齐。

| 集合 | 可以用于 |
| --- | --- |
| train | 字符转写、上下文计数、音节词表、规则回退、语言模型和神经权重 |
| dev | 候选生成器超参数、神经早停、规则回退置信度及音节编辑距离 |
| calibration | 选择器拟合及分组 OOF 的正则化/阈值选择 |
| test | 所有参数固定后的最终评估 |

候选模型的旧训练函数内部接受三份数据。正式驱动显式把其名为 test 的第三份参数绑定到 calibration，并将产物改名为 calibration_predictions；那一步没有访问最终 test。正式 test 只在 evaluate 阶段预测，已生成最终结果后拒绝覆盖重复评估。

## 模型与超参数

- 全局字符映射：训练集对齐与计数迭代 5 次，单字符最大输出长度 8。
- 上下文：按双侧三字符上下文、左侧上下文、右侧上下文、全局映射的顺序回退。dev 搜索最小支持数 1/2/3/5、置信度 0.5/0.6/0.7、词表最大编辑距离 0/1，最大音节字符数 8。优先最小 micro CER，持平时偏好更强证据，再用完全匹配率打破并列。
- 编辑标注器：4 层 Transformer 编码器，隐藏维度 256，8 个注意力头，前馈维度 1024，dropout 0.15；AdamW，学习率 3e-4、权重衰减 0.01、batch 128、最多 60 轮、早停 patience 12、梯度裁剪 1.0、标签平滑 0.03。具体 CLI 以保存的环境和训练摘要为准。
- 神经候选后处理：dev 搜索置信度阈值 0.0/0.4/0.5/0.6/0.7/0.8 和音节编辑距离 0/1。所有规则与词表来自 train。
- 候选选择器：28 维特征，包括训练参考文本的 2–5 阶字符语言模型、候选与源的编辑距离、长度比例、词表覆盖率、候选差异、编辑置信度、规则回退比例与校正幅度。特征不含 reference。
- 二分类监督：仅当辅助候选对参考文本的编辑距离严格低于主候选时标为 1，持平标 0。标准化在每个训练折内拟合；分组 5 折 OOF 搜索 C=0.01/0.03/0.1/0.3/1/3/10 与阈值 0.35–0.70（步长 0.05），以 OOF micro CER 选择。最终在 calibration 全集重新拟合，再评估 test。

选择器中“calibration”指独立校准数据的拟合协议，逻辑回归输出不是已证明校准可靠的语言置信度；本实现没有额外的可靠性曲线、ECE 或独立概率校准实验。OOF 得分用于调参，其自身不是最终泛化估计。

补充神经初始化变异时，保持 `--seed 42` 固定划分及校准交叉验证，用 `--model-seed` 改变神经训练种子，并写到独立目录。例如：

```bash
python scripts/run_grouped_experiment.py --code-root . --raw-data-dir data --output-dir experiments/grouped_model_seed43 --stage all --seed 42 --model-seed 43
```

重复同一既定方案的 42/43/44 种子，报告各自结果及均值/标准差，不根据 test 选择最佳种子作为主结果。这个补充只测训练随机性，不能代替新的数据来源划分。

已经完成主运行后，可直接复用其固定语料和上下文模型，顺序运行两个补充种子，再汇总：

```bash
python scripts/run_seed_replications.py --primary-run experiments/grouped_seed42 --output-parent experiments --raw-data-dir data
python scripts/summarize_seed_runs.py --primary-run experiments/grouped_seed42 --replication-parent experiments
```

## 指标定义

micro CER = 所有句段的 Levenshtein 编辑距离之和 / 所有参考文本 Unicode 码点数之和。藏文组合字符可能含多个码点；指标不是字形簇或语言学字单位的错误率。空白、藏文分隔符、标点和非藏文字符参与评分，不进行隐式 NFC/空白/标点归一化。

字符编辑准确率 CA：`micro_character_accuracy = 1 - micro_cer`，CA与EM在表格中均乘100展示为百分数，满足CA/%=100−CER/%；它是 CER 的互补指标，可为负，不是字符分类准确率。整段还原准确率 EM：`exact_match`，定义为还原结果与参考逐码点相等的句段比例。片段错误率：`line_error_rate = 1 - exact_match`。CA使用参考字符数归一化的编辑距离，EM使用测试片段数作为分母。`evaluate_predictions.py` 终端摘要同时显示 CA、EM、CER、片段错误率与TibER，数值为比例值；完整评分JSON保留既有全部指标，计算公式未改。micro Tibetan unit error rate 按实现中的音节和标点单位计算；不能称为经过词级边界金标准验证的 WER。

`edit_operation_counts` 记录把 prediction 编辑成 reference 的操作数，其中 insertion 表示补上预测缺失的参考字符，deletion 表示删去预测多出的字符。若论文按“参考文本生成预测”的常见方向描述插入/删除，应交换这两个标签；总距离和 CER 不受影响。

字符多重集重叠 precision/recall/F1 与 normalized edit similarity 是描述性辅助量，不替代主要指标。旧 mT5 的名为 rouge_l 的分数也不是标准 ROUGE-L，不与标准 ROUGE-L 文献结果比较。

Bootstrap 对同一组的主候选和最终选择进行成对比较，以连通分组而非每个句段作为重采样单位。正式驱动使用 5,000 次重采样、seed 42、百分位 95% 区间。显著性、区间与对比方向以实际产物为准。

## 产物与限制

正式实验目录包含 corpus、contextual、edit_tagger、selector 与 evaluation 子目录。`model_full.json` 保存完整上下文状态；`selector_model.json` 保存标准化参数、逻辑回归系数、训练语言模型与音节词表，可直接用于候选选择推理。神经权重 `best.pt` 需结合 train 的规则和词表，正式驱动负责重建。

原始行分组降低同源句段泄漏。据作者说明，数据整理自《藏文缩略语词典》（谢拉编著，民族出版社），并参考《缩略语词典》（项布旺杰，四川民族出版社）。两书的出版年份、版次、词条页码及采集复核标识仍待补充；各样本未有完整的词典来源标签，因此不能证明跨词典或古籍正文泛化。修正划分来自已用于开发的数据，不能替代完全未接触的新外部测试集。建议后续补充来源级划分、外部数据、现代预训练基线与专家盲评；这些未实施项目不能写成已完成实验。多神经种子的补充实验只检验固定划分下的训练变异，不检验跨来源泛化。

随机种子固定不保证不同硬件、PyTorch/BLAS 版本和并行后端的神经训练逐位一致。上传包不会自动公开真实数据、权重或论文稿件；所有共享先明确数据来源与权利。
