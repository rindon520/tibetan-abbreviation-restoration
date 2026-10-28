# 早期 mT5 实验

这里保留旧版 `train.py`、`predict.py` 和 `data_analysis.py`，与本文的统计转写/编辑候选/校准选择器分开。旧脚本未按正式分组协议训练，也不提供公平对比的独立泛化证据。

从仓库根目录运行 `python legacy/mt5/train.py`。先编辑 Config 中的数据路径和硬件设置，安装本目录 requirements.txt。训练默认写到 `checkpoints/` 和 `final_model/`；推理的默认路径及任务前缀已与训练对齐：

```bash
python legacy/mt5/predict.py --model_dir final_model --text "待还原文本"
```

原始旧脚本的 `rouge_l` 实际按字符多重集重叠计算 F1，并非最长公共子序列的标准 ROUGE-L。它仅是历史实现保留项，不用于正式论文评分；报告结果应统一调用 `evaluate_restore.compute_metrics`。
