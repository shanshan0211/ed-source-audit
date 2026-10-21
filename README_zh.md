# 急诊预测模型信息来源审计：代码发布包

本文件夹用于整理后上传 GitHub。原研究目录没有被移动或删除；本次仅准备本地发布包，没有创建仓库或上传文件。

对应工作稿：*Information-Source Dependence in Emergency Department Prediction Models: A Retrospective Audit*。

## 包含什么

- `src/`：数据构建、基线模型、TLQ-T、E1–E5 审计与评价代码。
- `configs/`：合成测试配置与历史真实数据配置模板。
- `scripts/`：跨平台运行入口、患者分组重分析、既有聚合结果的补算/绘图与发布检查。
- `tests/`：人工数据测试，不包含真实患者记录。
- `pyproject.toml`、`.gitignore`、文档及文件校验清单。

不包含任何研究数据、临床结果文件、逐例预测、嵌入缓存、模型权重、日志、论文草稿或本地依赖环境。现有代码并不等于现有论文的全部结果可在无数据条件下一键复现。

## 如何运行

建议 Python 3.11/3.12。在本文件夹打开终端并激活自己的虚拟环境：

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/check_release.py
python scripts/run_pipeline.py --config configs/ci.yaml --dry-run
```

若要运行完整合成模型流程，再安装模型依赖：

```bash
python -m pip install -e ".[models,text,dev]"
python scripts/run_pipeline.py --config configs/ci.yaml
```

合成配置使用本地 hashing 嵌入，不下载 Bio_ClinicalBERT。它与论文历史嵌入不同，只用于工程测试。运行入口拒绝覆盖已存在的缓存/结果目录；测试数字不能写成临床结果。

## 科学与数据边界

本工作是回顾性信息来源依赖审计，不是因果分解或已经完成的严格分诊时点增量价值研究。24 小时输入/结局窗口重叠、模型输入不匹配、住院关联字段、历史划分/预处理及快照问题仍然存在，详见[复现范围与限制](docs/REPRODUCIBILITY.md)。

`scripts/reanalyse_triage.py` 是待在有授权真实数据的电脑上运行的补分析工具，目前只有人工数据流程测试；不能写成新增真实实验已经完成。既有图表补算脚本所需的聚合 CSV 也未随本包发布。

真实数据运行须自行取得数据授权、核实机构伦理要求，并遵守数据使用协议。不能将数据、患者 ID、主诉、逐例预测、模型文件或未经审查的结果上传 GitHub。不要把“去标识化”自动写成“机构已批准豁免”。

## 发布前最后确认

只上传本文件夹，不上传父目录。账号密码、数据授权文件、真实数据和本地环境均不应进入仓库。

作者尚未选择代码许可证，本包没有擅自添加 MIT/Apache 授权；公开可见不等于允许他人任意再利用。正式公开前请确认代码权属、第三方许可并选择许可证。仓库链接及软件 DOI 也尚未产生，不能先写入投稿材料。

更完整的安装说明见[英文 README](README.md)，发布核对事项见[清单](docs/PUBLICATION_CHECKLIST.md)。
