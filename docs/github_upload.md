# GitHub 上传

只上传本仓库目录内容。`data/` 收录指定的 `Source.txt`、`Restore.txt`、`tibetan_VCC_7387.txt` 及说明文件，见 [data/README.md](../data/README.md)；`segment-ti/` 收录作者提供的分词器压缩包内容及整合说明，见 [segment-ti/README.md](../segment-ti/README.md)。上层工作区中的论文、其余数据、训练输出和其他第三方程序不属于仓库。

在仓库根目录完成本地检查：

```bash
python -m unittest discover -s tests -v
python scripts/smoke_test.py
python scripts/audit_release.py --check
```

在 GitHub 创建一个空仓库，然后在本目录执行（替换 OWNER 和仓库 URL）：

```bash
git init
git add .
git diff --cached --stat
git status --short
git commit -m "Prepare restoration research code and grouped reproducibility workflow"
git branch -M main
git remote add origin https://github.com/OWNER/tibetan-abbreviation-restoration.git
git push -u origin main
```

提交前检查暂存文件清单，不要将父目录作为 Git 仓库根目录。指定的三份 `data/` 文件，以及 `segment-ti/` 中有清单记录的分词器原始文件与说明，已在忽略规则与发布审计中单独列入。分词器包内的历史编译产物、共享库及大文本作为指定包内容保留；其他语料、训练输出、二进制和还原模型权重仍不应强制加入版本控制。CI 运行 CPU 核心检查，不自动启动真实语料训练，也不将缺少配套模型的分词器执行作为通过条件。

需要公开完整实验时，将经授权的模型和大型数据放在独立的发布资产或研究数据平台，并在 README 记录权限、版本、下载地址与 SHA256。作者决定 LICENSE 后再加入该文件。以上命令用于首次建立仓库；已有仓库应先同步远程，再提交并推送新增文件。
