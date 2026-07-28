# 项目维护

`正式代码/manage_model_assets.py` 用于扫描全部模型权重、计算 SHA256、生成可跟踪的
资产清单，并按用途制作私有 GitHub Release 包。

仅更新清单：

```bash
python 程序/项目维护/正式代码/manage_model_assets.py
```

验证本地权重：

```bash
python 程序/项目维护/正式代码/manage_model_assets.py --verify
```

生成全部 Release 包：

```bash
python 程序/项目维护/正式代码/manage_model_assets.py --package all
```

生成的归档位于 `发布资产/模型权重/`，该目录被 Git 忽略。归档只包含权重及不含
患者记录的配置、阈值、训练历史和指标摘要。
