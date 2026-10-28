# GitHub 上传

只上传本目录内容；上层工作区中的论文、真实数据、训练输出和第三方程序不属于仓库。

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

提交前检查暂存文件清单，不要将父目录作为 Git 仓库根目录，不要使用强制添加把语料或权重加入版本控制。CI 运行 CPU 核心检查，不自动启动真实语料训练。

需要公开完整实验时，将经授权的模型和大型数据放在独立的发布资产或研究数据平台，并在 README 记录权限、版本、下载地址与 SHA256。作者决定 LICENSE 后再加入该文件。仓库创建、提交和推送命令仅供作者操作，本整理过程没有向远程推送。
