# 第三方组件

| 组件 | 本仓库的处理 |
| --- | --- |
| NumPy、scikit-learn、PyTorch | 通过依赖管理安装，不复制其源码或运行库；遵守各项目许可 |
| Hugging Face Transformers、Datasets、SentencePiece | 仅早期 mT5 基线需要，按各自许可安装 |
| 预训练 mT5 权重 | 本仓库不包含权重；需自行确认模型卡和使用条件 |
| `segment-ti` | 本地副本无 LICENSE；不复制源码、模型、词典、语料、二进制或包装器 |
| TibNorm | 外部对比固定提交，无已确认的 LICENSE；不复制源码和归一化词典，仅提供调用与计分脚本 |

外部 TibNorm 的固定版本、资源条件与复现命令见 [docs/external_baseline.md](docs/external_baseline.md)。

`segment_tibetan.py` 只是已有项目的调用接口。需要外部分词时，将经授权获取的可执行程序和它所需的资源放在本地目录，向两阶段工具显式传入 `--segmenter-dir /absolute/path/to/segment-ti`。程序在该目录内运行，接口约定为 `tiSegmenter -s INPUT -t OUTPUT`。现有 ELF 二进制不能在原生 Windows 上直接运行；需适合平台的程序。

未安装外部分词器时可显式关闭外部分词，使用音节边界回退。回退与词级分词具有不同的实验语义，复现记录必须保存实际分词模式，不能把两者视为相同设置。

如以后获得明确授权再发布第三方组件，应同时保留上游来源、版本及其原始许可文本。
