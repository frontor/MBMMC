# MBMMC 中文使用说明

## 三个模型训练入口

```text
train_rf.py
train_crossnn.py
train_mpcnet.py
```

查看帮助：

```bash
python train_rf.py --help
python train_crossnn.py --help
python train_mpcnet.py --help
```

## 安装

Conda：

```bash
conda env create -f environment.yml
conda activate mbmmc
pip install -e .
```

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

macOS/Linux：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## 模型训练最小示例

```bash
python examples/make_toy_data.py
bash examples/run_rf.sh
bash examples/run_crossnn.sh
bash examples/run_mpcnet.sh
```

示例数据为合成数据，只用于检查软件流程，不能作为论文结果。

## 跨平台模拟数据生成

独立模拟器位于：

```text
scripts/simulation/generate_cross_platform_in_silico_beta.py
```

查看帮助：

```bash
python scripts/simulation/generate_cross_platform_in_silico_beta.py --help
```

安装仓库后：

```bash
mbmmc-generate-simulation --help
```

生成并运行最小示例：

```bash
python examples/make_simulation_reference.py
bash examples/run_simulation.sh
```

模拟器不属于 RF、crossNN 或 MPCNet 的模型训练代码。输入、输出、来源治理和正式
使用建议见 `docs/SIMULATION.md`。

## 正式训练前确认

- 甲基化矩阵和 metadata 中样本名完全一致；
- `Sample` 和 `Types` 列存在；
- beta 值及缺失值编码符合 `docs/INPUT_OUTPUT.md`；
- 患者、研究、平台或批次分组变量设置正确；
- 独立测试集未参与参数选择；
- 参数候选集和选择指标已经冻结。

参数组合见 `configs/publication_candidates.yaml`，复现清单见
`docs/REPRODUCIBILITY.md`。

## crossNN 方法来源

MBMMC 的 crossNN 是参考 Yuan 等人在 2025 年发表的方法后独立实现的版本，不是
原 crossNN 作者的官方发布版本。学术来源和代码层面对比见：

```text
THIRD_PARTY_NOTICES.md
docs/CROSSNN_METHOD_PROVENANCE.md
```

## 许可证与非商业使用

MBMMC 使用：

```text
PolyForm Noncommercial License 1.0.0
SPDX: PolyForm-Noncommercial-1.0.0
```


有权利的版权主体可以为未来版本选择其他许可证，但已经发布的历史版本继续适用其
发布时的许可。详细规则见 `docs/LICENSE_POLICY.md`。
