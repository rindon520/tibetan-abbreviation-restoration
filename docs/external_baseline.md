# TibNorm 外部基线

公开比较采用 [TibNorm](https://github.com/Divergent-Discourses/TibNorm) 的完整默认归一化流水线，[固定提交](https://github.com/Divergent-Discourses/TibNorm/tree/d296f6566da1308f7c14180080acc43ace8dd68a) 为 `d296f6566da1308f7c14180080acc43ace8dd68a`。它不是本仓库的训练模型，也没有针对测试集修改规则。

获取经授权的上游项目后，安装 `requirements-external.txt`，保持固定提交且工作区无修改，运行：

```bash
python scripts/evaluate_tibnorm_external.py --repo-root . --upstream-dir /absolute/path/to/TibNorm --corpus-dir experiments/grouped_seed42/corpus --output-dir experiments/grouped_seed42/tibnorm_external
```

脚本检查提交、工作区、默认 flag=1 和固定数据文件哈希；仅把源文本交给上游正常化函数。规则顺序为 abbreviations、table1、table2、table3。该提交加载 6,522 行缩略词表，按 transcription 键合并后有效条目为 6,263。

961 条最终测试句段的实测聚合结果：micro CER 28.0764%，Tibetan unit error rate 52.8257%，整句完全匹配率 3.4339%。总参考码点 25,249，总字符编辑距离 7,089。结果与相关文件哈希保存在 results_summary.json，不包含逐条文本。

TibNorm 使用外部公布词典，资源范围与本文仅使用 train 学习的候选方法不同；词典是否与本文语料重叠无法核查。其完整流水线还改变数字、标点和拼写，任务范围广于参考配对上的缩略词还原。本文不能据此声称首次藏文缩略词还原、首次混合方法或全面优于现有研究。

上游当前副本没有找到 LICENSE 文件。外部源码、词典和运行产物不复制进本上传包，不另行套用本项目许可证。该结果属于固定默认外部比较，不能用测试结果改写上游规则后仍称为同一基线。
