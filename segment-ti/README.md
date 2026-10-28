# 藏文分词器 segment-ti

本目录收录作者提供的 `segment-ti.zip`：42 个原始文件，合计 37,533,358 字节。源码、预编译程序、字符规则和随包文本均按原始字节保留；本说明与 `archive_manifest.json` 是本仓库新增的说明文件。

**当前压缩包缺少源码引用的模型与词典资源，不能直接视为完整可运行的分词器模型包。** 原始 `readme` 保留不改，其 `./test` 示例与实际 makefile 目标不同；本目录命令行程序名为 `tiSegmenter`。

## 文件组成

| 文件或目录 | 用途 |
| --- | --- |
| `PerSeg.cpp`、`Segmentor.cpp/.h`、`Perceptron.cpp/.h` | 命令行入口、分词逻辑与感知机相关源码 |
| `splitutf.cpp/.h`、`Utf8Stream.cpp/.h`、`ChType.cpp/.h`、`DataStruct.h` | UTF-8 处理、字符类型与辅助结构 |
| `makefile`、`tiSegmenter` | 编译配置与 Linux x86-64 ELF 可执行程序 |
| `utf/` | 随包提供的字符类型、数字和分隔规则资源 |
| `cpp2py.sh`、`cpp2python.py`、`tiseg.cc`、`tiseg.i`、`tiseg.so`、`test.py`、`named_tuple.py` | 历史 Python 扩展相关文件；构建脚本面向 Python 2.7/Boost.Python 2.7，不是本仓库 Python 3 接口 |
| `duanju.txt`、`ti`、`a.txt`、`test.txt`、`out.txt` | 随包文本及示例材料，不将其解释为训练模型 |
| `*.o`、`tiseg.cc~`、`exposed_decl.pypp.txt` | 随包保留的历史编译、备份或生成产物 |
| `archive_manifest.json` | 原压缩包 SHA-256、完整文件清单、逐文件大小与 SHA-256 |

## 缺失资源与运行环境

`Segmentor.cpp` 引用以下相对路径，但本次提供的压缩包中没有对应文件：

```text
segment-ti/data/model
segment-ti/data/data.txt
segment-ti/data_bpe.txt
```

这些路径是分词器自身的资源，不能用仓库根目录 `data/Source.txt`、`data/Restore.txt` 或藏文字符集替代。补齐匹配的资源版本后，仍需验证实际分词输出。

`tiSegmenter` 与 `tiseg.so` 为 Linux x86-64 ELF 文件，不能直接在原生 Windows 运行。Linux 或 WSL 中也需要匹配的系统依赖与上述资源；本次未执行原包二进制或验证完整分词功能。Git 已为 `tiSegmenter` 设置可执行标记，文件内容保持不变。

原 makefile 使用 `g++` 并链接 zlib（`-lz`），其预期的编译与调用方式如下。以下命令不代表当前缺失资源的压缩包已能完成分词：

```bash
cd segment-ti
make
./tiSegmenter -s /absolute/path/input.txt -t /absolute/path/output.seg
```

编译会更新对象文件和可执行程序；如需保持原包校验一致，请先将本目录复制到单独的工作目录中编译。历史 Python 扩展构建脚本包含旧机器的绝对路径，不能视为当前 Python 3 环境的安装脚本。

## 本仓库的调用接口

仓库根目录 [segment_tibetan.py](../segment_tibetan.py) 通过命令行调用本目录的 `./tiSegmenter`，并将工作目录设为本目录，以便程序解析相对资源路径；不依赖历史 `tiseg.so` 扩展。

```python
from pathlib import Path
from segment_tibetan import segment_lines

segmented, status = segment_lines(
    ["待分词的藏文文本"],
    segmenter_dir=Path("segment-ti"),
    prefer_external=True,
    allow_fallback=False,
)
```

上述严格模式需要可运行的程序与完整资源。允许回退时，接口会在外部分词调用失败后使用音节边界回退；音节边界回退与词级分词的结果不等价，实验必须记录实际 `status["mode"]`。当前资源不完整时，可显式设置 `prefer_external=False` 使用回退。

原命令行程序在部分初始化失败分支仍返回退出码 0，因此不能只凭退出码认定分词成功。本仓库接口会检查输出行数与输入是否一致；空结果或缺行会触发严格报错或已允许的回退，不作为成功的外部分词输出。行数一致仍不代表分词内容已通过人工复核。

## 来源、许可与校验

本目录来源为作者本次提供的 `segment-ti.zip`，未据文件名推断上游作者、机构、版本或许可证。原包没有 LICENSE 文件；本次收录不将其重新许可为本项目原创代码。现有记录见 [第三方说明](../THIRD_PARTY_NOTICES.md) 和 [许可状态](../LICENSE_STATUS.md)。

原压缩包 SHA-256：

```text
050d87bc9981507d19496910f35d00cebd02ada941b089cd2b0390552d776be7
```

`scripts/audit_release.py` 依据 `archive_manifest.json` 核对原包文件，允许其中原有的二进制和较大文本进入发布包；本目录中新生成的其他文件仍由忽略规则排除。
