# 藏文缩略词还原 / Tibetan Abbreviation Restoration

用于藏文缩略词还原的研究代码：上下文字符转写、音节词表校正、Transformer 字符编辑标注，以及在独立校准集上学习的候选选择器。

据作者说明，当前配对数据整理自谢拉编著的《藏文缩略语词典》（民族出版社），并参考项布旺杰的《缩略语词典》（四川民族出版社）。两书的出版年份、版次、具体词条来源对应和整理复核流程尚待补齐；书名、作者和出版社按作者提供记录。当前实证范围为词典来源的配对文本，不构成 OCR 或独立古籍正文基准。

本目录是独立的 GitHub 仓库，包含研究代码，以及 `data/` 中的藏文缩写—还原对照语料和藏文字符集。数据文件用途、记录数与读取方式见 [data/README.md](data/README.md)。论文、训练日志、模型权重和第三方分词器不包含在本仓库。现有代码和新增复现入口的来源见 [docs/source_manifest.json](docs/source_manifest.json)。

## 安装与最小验证

使用 Python 3.11 或更新版本，从本仓库根目录执行：

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-core.txt
python -m unittest discover -s tests -v
python scripts/smoke_test.py
```

最小验证使用 [examples/toy](examples/toy) 中人工构造的字符替换数据，不表示藏文语言规律，也不产生论文实验结论。它检查训练、完整模型保存、重新加载及批量推理。

神经编辑标注器和字符级 Seq2Seq 基线另外需要 PyTorch：

```bash
python -m pip install -r requirements-training.txt
```

选择与设备匹配的 PyTorch 安装包；仓库不绑定某种显卡或 CUDA 版本。Windows 上旧 Conda/MKL 环境可能出现重复 OpenMP 运行库问题，建议新建干净 venv 并安装 pip 版依赖，不以忽略运行库冲突作为正式训练方案。

## 正式实验：保留原始行分组

仓库已提供原始行级配对数据 `data/Source.txt` 和 `data/Restore.txt`；两文件按相同行号对应，均为带 BOM 的 UTF-8 文本。文件说明见 [data/README.md](data/README.md)，格式与划分约定见 [docs/data_format.md](docs/data_format.md)。正式实验使用以下入口：

```bash
python scripts/run_grouped_experiment.py --code-root . --raw-data-dir data --output-dir experiments/grouped_seed42 --stage prepare --seed 42
python scripts/run_grouped_experiment.py --code-root . --raw-data-dir data --output-dir experiments/grouped_seed42 --stage train-candidates --seed 42
python scripts/run_grouped_experiment.py --code-root . --raw-data-dir data --output-dir experiments/grouped_seed42 --stage train-selector --seed 42
python scripts/run_grouped_experiment.py --code-root . --raw-data-dir data --output-dir experiments/grouped_seed42 --stage evaluate --seed 42
```

`prepare` 保存原始行与分句的对应关系，按原始行及重复源文本关联组划分 train/dev/calibration/test，写入固定划分与文件哈希。候选生成器仅在 train 学习、在 dev 调参；选择器在独立 calibration 学习并使用分组交叉验证；全部参数固定后执行 test 评估。完整参数及产物见 [docs/reproducibility.md](docs/reproducibility.md)。

同一划分的字符 Transformer Seq2Seq 比较：

```bash
python scripts/train_grouped_seq2seq.py --repo-root . --corpus-dir experiments/grouped_seed42/corpus --output-dir experiments/grouped_seed42/seq2seq --epochs 60 --patience 12 --seed 42
```

词表只来自 train，dev loss 选择 checkpoint，最终采用关闭 TF32 的批量贪心解码。它是从头训练的字符模型，不等同于现代预训练模型基线。

**历史复现限制：**根目录 `train_*.py` 保留既有研究接口，其独立运行的默认划分仍是句段随机划分。历史的 481 条测试句段中有 91 条与训练句段来自同一原始行；这些结果不能用来确认独立原始行上的泛化性能。不要混用历史预测文件与修正后的分组划分。正式论文以修正实验为准。

## 推理

完整双路方法对新文本推理（需保留本地固定划分、训练参考文本、神经权重与选择器，重新构建 train 的规则/词表）：

```bash
python infer_pipeline.py --experiment-dir experiments/grouped_seed42 --input-file input.txt --output restored.jsonl --device cpu
```

该入口只从输入文件读取源文本，不需要待还原文本的参考答案。输出记录包括还原文本、选择器得分及选中的候选模型。

新训练的上下文模型保存完整统计状态，可直接推理：

```bash
python infer_contextual.py --model experiments/grouped_seed42/contextual/model_full.json --text "待还原的藏文文本"
python infer_contextual.py --model experiments/grouped_seed42/contextual/model_full.json --input-file input.txt --output-file predictions.txt
```

旧 `model.json` 仅保存部分词表和统计量，不能保证重载后复现原预测；新入口会明确拒绝这种旧产物。

候选选择器支持对两个**同顺序、同源文本**的候选 JSONL 文件推理，使用正式训练导出的 `selector_model.json`（或独立旧接口新导出的 `selector.json`）：

```bash
python infer_selector.py --model experiments/grouped_seed42/selector/selector_model.json --primary primary.jsonl --auxiliary auxiliary.jsonl --output selected.jsonl
```

每行至少包含 `source` 和 `prediction`，可附带编辑标注器的 `decisions` 与校正前预测。推理不使用 `reference`。神经编辑标注器的旧权重不包含完整规则回退和音节校正状态；仅凭旧 `best.pt` 不能完整恢复整个候选流水线。正式分组入口重新从 train 构建这些统计量。

字符级 Seq2Seq 基线的推理：

```bash
python infer_restore.py --checkpoint runs/transformer/best.pt --input-file input.txt --output-file predictions.txt --device cpu
```

任意模型的评估 JSONL（包含 `reference`、`prediction`）均可统一计分：

```bash
python evaluate_predictions.py --predictions predictions.jsonl --output metrics.json
```

## 绘图复现

当前论文延续原 V21 的六图顺序及单栏叙事；新增 `figures/v21/` 的绘图代码与聚合 CSV/JSON，可从公开聚合数据直接生成图2—6，不依赖私有实验目录。图件目录不含已渲染的 PDF、PNG、SVG 或逐条原始文本；原始配对语料位于仓库的 `data/` 目录。

```bash
python -m pip install -r requirements-figures.txt
python figures/v21/build_v21_figures.py --output-dir figures/generated/v21
```

正文图1为原文藏文案例图，保留在作者本地稿件，代码包不再分发；图2—6依次对应 `fig2_data_protocol`、`fig3_method_framework`、`fig4_main_results`、`fig5_calibration_sensitivity`、`fig6_error_analysis`。尺寸、字体及单图复现见 [figures/v21/README.md](figures/v21/README.md)。旧跨栏图件入口保留为扩展或审稿核查材料，见 [figures/README.md](figures/README.md)；旧四图映射不适用于当前正文。生成文件不进入 Git/ZIP。

## 代码地图

| 路径 | 用途 |
| --- | --- |
| `scripts/run_grouped_experiment.py` | 正式分组划分、训练、校准和最终评估 |
| `scripts/train_grouped_seq2seq.py` | 同一固定划分下的字符 Transformer Seq2Seq 基线 |
| `train_contextual_corrector.py`, `train_transducer.py` | 上下文统计转写与全局转写基线 |
| `tibetan_syllable.py`, `train_syllable_corrector.py` | 音节切分、训练词表与动态规划校正 |
| `train_edit_tagger.py` | Transformer 编码器字符编辑标注候选 |
| `train_hybrid_selector.py` | 28 维候选特征、标准化与逻辑回归选择 |
| `infer_pipeline.py`, `infer_contextual.py`, `infer_selector.py`, `model_artifacts.py` | 双路完整推理、完整 JSON 状态与单组件推理 |
| `train_restore.py`, `infer_restore.py`, `evaluate_restore.py` | 自回归字符 Transformer 基线及指标 |
| `prepare_segment_data.py` | 历史句段抽取工具；不保存原始行分组 |
| `abbrev_restore.py`, `train_abbrev_restore.py`, `ablate_abbrev_restore.py` | 独立的两阶段检测与还原实验、消融 |
| `segment_tibetan.py` | 可选外部分词器接口与音节边界回退 |
| `summarize_sci_metrics.py` | 历史指标汇总与成对 Bootstrap 工具 |
| `scripts/evaluate_tibnorm_external.py` | 固定版本 TibNorm 外部归一化比较，不复制上游资源 |
| `scripts/diagnose_frozen_candidates.py`, `scripts/ablate_neural_candidate.py` | 后验候选诊断与去神经控制，保留探索性身份 |
| `legacy/mt5/` | 早期 mT5 训练、推理和数据分析脚本 |
| `data/` | 藏文缩写—还原对照语料、藏文字符集及文件说明 |
| `tests/`, `examples/`, `docs/` | 核心测试、合成示例与复现文档 |
| `figures/v21/` | 当前 V21 六图稿的图2—6绘图入口与聚合输入 |
| `figures/scripts/`, `figures/source_data/` | 扩展核查图件代码与聚合指标源表 |

## 指标、数据与许可

同时报告字符编辑准确率 CA（`micro_character_accuracy = 1 - micro_cer`）、整段还原准确率 EM（`exact_match`）、字符错误率 micro CER（总编辑距离 / 总参考字符数）、片段错误率（`line_error_rate = 1 - exact_match`）和音节单位错误率。CA 是编辑距离的互补指标，可为负值，不能解释为字符分类准确率；EM 是整段逐码点相等的比例，分母为测试片段数。评估入口终端摘要和完整 JSON 均保留这两种准确率与错误率；终端各率以比例值显示，表格将相应值乘100展示为百分数；百分数满足CA/%=100−CER/%。主方法 CA 为82.71%、EM 为14.26%，两者定义不同。字符多重集重叠分数仅作辅助描述。Unicode 字符计数、标点处理及统计限制见 [docs/reproducibility.md](docs/reproducibility.md)。

原始配对语料与藏文字符集已收录于 [data/](data/README.md)；收录不自动授予公开再分发或开源使用许可。第三方 `segment-ti` 程序与模型仍未打包。默认可使用本仓库音节边界回退；它与外部分词的结果不等价。接口和许可状态见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 与 [LICENSE_STATUS.md](LICENSE_STATUS.md)。仓库没有替作者选择开源许可证；在权利人选择许可证之前，公开上传不等于授予开源使用权。

固定版本 TibNorm 的外部比较使用 6,263 个有效缩略词条目及完整正常化规则，资源条件不同于仅在本文训练集学习的方法。结果和限制见 [docs/external_baseline.md](docs/external_baseline.md)；本文不声称首次混合方法或首次藏文缩略词还原。

固定 seed 42、961 条测试句段的结果见 [docs/results.md](docs/results.md)，可核对的聚合数值与文件哈希见 [docs/results_summary.json](docs/results_summary.json)。结果汇总文件不包含逐条文本或模型权重；原始配对文本见 `data/`。

后验完整去神经控制与主方法只相差 2 次字符编辑，现有实证不支持神经候选的必要性。该诊断不替换预先固定的 seed42 主结果，详情与探索性限制见结果说明。

上传步骤见 [docs/github_upload.md](docs/github_upload.md)。本地验证记录见 [docs/validation.md](docs/validation.md)。
