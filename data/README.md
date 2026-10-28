# 数据说明

本目录包含藏文缩写与还原对照语料，以及藏文字符及组合形式清单。三个 TXT 文件均按作者提供的原始字节保存，未去重、重排、分句或进行 Unicode 规范化。

| 文件 | 内容与用途 | 记录数 | 编码 |
| --- | --- | ---: | --- |
| [Source.txt](Source.txt) | 藏文缩写原文，作为模型输入 | 10,820 行 | UTF-8，带 BOM |
| [Restore.txt](Restore.txt) | 与缩写原文对应的还原文本，作为训练或评估的参考文本 | 10,820 行 | UTF-8，带 BOM |
| [tibetan_VCC_7387.txt](tibetan_VCC_7387.txt) | 藏文字符及组合形式清单，供字符处理与词表构建参考 | 7,387 行 | UTF-8，无 BOM |

## 对照关系与读取

`Source.txt` 第 n 行与 `Restore.txt` 第 n 行对应，形成一条“缩写原文 → 还原文本”的配对记录。两个文件均无空行，行数一致。这里的 10,820 对是原始行级记录数，不是分句后的样本数；行数核对只确认结构对应，不代表已逐条完成语义复核。

三个文件均使用 CRLF 换行，末行没有额外换行符。读取时使用 `splitlines()`，不要仅以换行符数量作为记录数。以下示例可兼容语料文件开头的 BOM，并保留条目中的空格和组合字符：

```python
from pathlib import Path

data_dir = Path("data")
sources = (data_dir / "Source.txt").read_text(encoding="utf-8-sig").splitlines()
restores = (data_dir / "Restore.txt").read_text(encoding="utf-8-sig").splitlines()
assert len(sources) == len(restores)
pairs = list(zip(sources, restores))

character_entries = (
    (data_dir / "tibetan_VCC_7387.txt").read_text(encoding="utf-8").splitlines()
)
```

字符集文件没有编号或标签列。其 7,387 行中有 7,199 个不同条目，包含重复条目以及由多个 Unicode 码点组成的形式，不能将文件名中的 `7387` 理解为 7,387 个不同 Unicode 字符或模型词表大小。`train_restore.py` 的词表构建函数会进一步将每行内容拆为码点；本次保留原始清单，不预先去重或删除空格。

## 实验使用

正式分组实验可直接读取本目录的配对文件：

```bash
python scripts/run_grouped_experiment.py --code-root . --raw-data-dir data --output-dir experiments/grouped_seed42 --stage prepare --seed 42
```

该入口在抽取句段后，按原始行及重复源文本关联组划分 train/dev/calibration/test。训练、调参与评估继续遵循既有分组协议；加入字符集文件不改变既有实验结果或词表构建约定。完整格式与流程见 [数据格式](../docs/data_format.md) 和 [复现说明](../docs/reproducibility.md)。

## 来源与使用条件

配对语料的整理来源按作者提供的信息记录，见 [数据来源说明](../docs/data_format.md)。字符集文件在本仓库保留作者提供的文件名与内容，不据此推断其编制来源、标准版本或覆盖全部藏文字符。

本次将作者指定的三个文件加入当前私有仓库，不改变 [许可状态](../LICENSE_STATUS.md)，也不另行授予数据公开再分发许可。

## 文件校验

以下 SHA-256 用于确认文件与本次提供的版本逐字节一致：

| 文件 | 字节数 | SHA-256 |
| --- | ---: | --- |
| `Source.txt` | 814,658 | `1bff6b35dec9ccebdfad0c1fe6a0e1f543b359746bf890178c517144be9b6a13` |
| `Restore.txt` | 1,012,080 | `4a168f52013f596fc35df3c416c560457377d1a27f1ee4d400324e9798a91d90` |
| `tibetan_VCC_7387.txt` | 91,741 | `2ccd65a16e3fb659beaba5ad4912edd13f6684c92aea5ad8238e759641c27cbd` |
