# 贡献与研究变更

变更需说明影响的实验设置、数据划分与指标，运行 `python -m unittest discover -s tests -v` 和 `python scripts/smoke_test.py`。更改训练流程、特征、归一化或调参规则时，应重新生成独立的实验目录，保留旧结果，不覆盖已用于论文的产物。

保持 train/dev/calibration/test 的分组互斥。不要根据 test 选择候选模型、阈值、词表或归一化规则；不要把测试文本写入训练规则、词表或语言模型。真实语料和模型产物不在普通提交中发布。

本目录采取平铺模块以保留既有导入和 CLI。新增复现驱动放在 scripts，协议说明放在 docs，合成样例放在 examples，论文在仓库外单独维护。
