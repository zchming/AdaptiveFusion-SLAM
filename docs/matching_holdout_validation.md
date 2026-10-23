# V22：冻结候选的同序列新图像验证

MAGSAC 候选在预先选定的48对新输入上得到 **46/48准确位姿**，LightGlue为45/48，
原始估计为45/48；相对LightGlue新增一对准确结果，没有损失其原有准确结果。三个
等量控制为45/43/43。支持继续保留MAGSAC候选，仍不自动启用为在线默认方案。

## 预先固定的样本与参数

四条既有序列各12对，共96张图像。按原关键帧ID选相邻关键帧，原始估计端点有效；
贪心构造互不共用图像的候选对，再按候选索引等间隔选12对。每个端点与V19任何调试
图像至少相隔1秒。选择在推理和GT评价前封存，不看匹配质量、真值或候选结果。

实际96个RGB文件哈希均唯一，与V19调试图像的哈希交集为零。选择清单和协议保留在
结果中。这里验证的是同序列、不同图像的泛化；不是新序列泛化，也不覆盖lost帧恢复。

RootSIFT2048、SIFT-LightGlue权重/配置、原始深度、相机、F/H参数和原生PnP全部保持
V19/V21设置。不叠加时间补偿，不调参，不补充选择表现更好的样本。

## 全部结果

准确位姿必须在nearest和interpolated两种GT对齐下同时满足3cm/2°；所有48对保留。

| 方案 | 应用 /48 | 准确 /48 | 损坏原始准确位姿 | 修复原始错误 |
| --- | ---: | ---: | ---: | ---: |
| 原始估计 | — | 45 | 0 | 0 |
| LightGlue | 48 | 45 | 1 | 1 |
| LightGlue + F-MAGSAC++ | 48 | 46 | 1 | 2 |
| 等量子集17 | 48 | 45 | 2 | 2 |
| 等量子集23 | 48 | 43 | 2 | 0 |
| 等量子集41 | 48 | 43 | 3 | 1 |

F-MAGSAC相对LightGlue新增准确的是desk65-66，没有新增准确性损失。分场景准确数为
xyz12/12、desk12/12、room11/12、teddy11/12；LightGlue的desk为11/12，其余相同。

| Interpolated GT | 平移中位数 cm | 平移P95 cm | 旋转中位数 ° | 旋转P95 ° |
| --- | ---: | ---: | ---: | ---: |
| LightGlue | 1.097 | 2.889 | 0.561 | 1.450 |
| F-MAGSAC++ | 1.204 | 2.420 | 0.536 | 1.360 |
| 等量子集17 | 1.278 | 2.923 | 0.537 | 1.326 |
| 等量子集23 | 1.156 | 4.311 | 0.611 | 1.400 |
| 等量子集41 | 1.157 | 3.340 | 0.597 | 1.374 |

F的平移尾部与准确数改善，但平移中位数略高于LightGlue；并非所有维度都最佳。开发集
表现最好的随机子集23，在新样本上只有43/48，说明不能按开发集最优随机子集部署。
F在本次新样本中优于所有三个控制的准确数，进一步支持几何验证这一路线。

## 两个未达标结果

- room20-21：F返回最大误差3.413cm/1.965°，原始位姿也未达标，未被修复。
- teddy170-171：F返回5.002cm/1.317°，原始位姿准确。这一对仍构成新增退化，
  LightGlue也未解决它，不能隐藏或用总体准确率抵消它。

没有F/H模型失败、匹配推理失败、进程失败、原生拒绝、超时或OOM；全部240次新的原生
调用均应用了结果。准确性未达标与估计器拒绝分别记录。

## 实施与成本

新增 `python/prepare_matching_holdout.py`，只读取原估计轨迹、关键帧元数据及开发集
图像清单，输出封存的样本清单，不读取GT。它生成明确标记为“仅pair manifest、没有
拟合”的兼容输入容器，不把未拟合数据称作实验结果。

原RootSIFT和MAGSAC入口增加显式 `--expected-cases`，默认仍39；RootSIFT另支持选择
匹配器，本次仅执行LightGlue。旧39对协议和默认算法设置未改变。全部推理与240次拟合
完成并封存后，才依次评价LightGlue和MAGSAC，不在基线GT结果出来后再构造候选输入。

96张图像的SIFT提取2.514s，匹配阶段6.766s，模型载入0.262s，验证阶段1.050s；
GPU Torch峰值138.29MiB allocated/206MiB reserved。它们是阶段测时，顶层解释器启动
和GT评价另记于日志。没有训练、新权重下载、BA或在线跟踪接入。

原LightGlue拟合日志有固定文字“2 pose arms retained”；本次实际结构化记录每对只有
一个LightGlue arm，共48次。已将后续日志改为实际数量，不改变数值结果或重跑实验。
执行时源码快照仍保留。

## 复现与证据

先用只含各scene `frames/features/dataset` 的推理清单固定输入：

```sh
python python/prepare_matching_holdout.py \
  --sources "$INFERENCE_SOURCES" --development-study "$V19_STUDY" \
  --native-binary "$NATIVE_PNP" --pairs-per-scene 12 --exclusion-s 1 \
  --output "$NEW_PAIR_MANIFEST"
```

随后按原显式输入路径运行RootSIFT匹配，指定 `--expected-cases 48 --matchers
rootsift_lightglue`；原始LightGlue拟合完成后，MAGSAC验证指定 `--expected-cases 48`
和[冻结的新样本协议](../results/matching_holdout_protocol.json)，继续执行四组原生拟合。
两个study都封存后才调用原独立评价器，完整参数在外部`commands.json`中保留。

- [48对清单、时间戳及RGB哈希](../results/matching_holdout_pairs.csv)
- [完整480行评价](../results/matching_holdout_results.csv)
- [总体与场景统计](../results/matching_holdout_summary.csv)
- [逐对基线、筛选与等量控制](../results/matching_holdout_diagnostics.csv)
- [完整摘要与原始证据位置](../results/matching_holdout_summary.json)

开发集与新样本结果分别报告，不混合成一个可能误导的总体准确率。候选状态继续为
“离线候选，生产默认关闭”；完整序列和在线稳定收益尚需独立授权验证。
