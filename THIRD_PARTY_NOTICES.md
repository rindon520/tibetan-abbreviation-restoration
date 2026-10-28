# 第三方组件

| 组件 | 本仓库的处理 |
| --- | --- |
| NumPy、scikit-learn、PyTorch | 通过依赖管理安装，不复制其源码或运行库；遵守各项目许可 |
| Hugging Face Transformers、Datasets、SentencePiece | 仅早期 mT5 基线需要，按各自许可安装 |
| 预训练 mT5 权重 | 本仓库不包含权重；需自行确认模型卡和使用条件 |
| `segment-ti` | 按作者提供的压缩包收录源码、Linux 程序、字符规则及历史产物；包中无 LICENSE，不按本项目代码重新许可；模型资源缺失情况见目录说明 |
| TibNorm | 外部对比固定提交，无已确认的 LICENSE；不复制源码和归一化词典，仅提供调用与计分脚本 |

外部 TibNorm 的固定版本、资源条件与复现命令见 [docs/external_baseline.md](docs/external_baseline.md)。

`segment_tibetan.py` 是分词器的命令行调用接口，默认目录为仓库的 `segment-ti/`；两阶段工具也可显式传入 `--segmenter-dir /absolute/path/to/segment-ti`。程序在该目录内运行，接口约定为 `tiSegmenter -s INPUT -t OUTPUT`。所提供的 `tiSegmenter` 与 `tiseg.so` 是 Linux x86-64 ELF 文件，不能在原生 Windows 上直接运行。现有接口调用 CLI，不依赖包内面向 Python 2.7 的历史扩展。

本次压缩包缺少源码引用的 `data/model`、`data/data.txt` 和 `data_bpe.txt`，因此当前收录不等于已提供完整模型或已验证分词可运行。原始文件清单、校验与构建/使用说明见 [segment-ti/README.md](segment-ti/README.md)。

未安装外部分词器时可显式关闭外部分词，使用音节边界回退。回退与词级分词具有不同的实验语义，复现记录必须保存实际分词模式，不能把两者视为相同设置。

后续公开发布第三方组件时，应保留其上游来源、版本及原始许可文本，并确认相应使用与再分发条件。
