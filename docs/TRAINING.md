# 神经训练、输入与模型使用

公开训练后权重在`models/pretrained/`，新建筑试用和续训见[使用公开模型](USE_PRETRAINED.md)。

先准备实景视频观测、配准和经核验的监督。公开代码包含训练/推理；默认运行不替用户长训练。支持单建筑适配，不宣称已有跨校园建筑泛化模型。

## 六角点局部头

输入NPZ含`points_CAD_m`或`points_world_m`、归一化0..1的`rgb`、`frame_refs`或`source_refs`。目标NPZ含`target_world_m`，形状6×3，按ABCDEF排列，另可给`weights`形状6。

```bash
python -m zip2cad train-corners --inputs data/train/corner_inputs.npz --targets data/train/corner_targets.npz --output outputs/corner_training --steps 1500
python -m zip2cad infer-corners --inputs data/train/corner_inputs.npz --checkpoint outputs/corner_training/checkpoint.pt --contract outputs/corner_training/input_contract.json --output outputs/corner_prediction.json
```

可用`--resume checkpoint.pt`继续训练。使用训练时的归一化契约，否则输入范围变化会影响结果。代码为44,440参数PointNet attention局部头，米制角点/边向量/边长约束。六角点不能决定真实梁宽、梁高和全部设备。

## 冻结Sonata语义头

特征NPZ含`features`（N×1088）。标签NPZ含`semantic`（N，0..8，未知为−1）；建议同时给`train_mask`，只在训练子集计算归一化统计。参考CAD可用于生成训练标签，但必须与推理输入分开。

```bash
python -m zip2cad train-semantic --inputs data/train/frozen_features.npz --targets data/train/semantic_labels.npz --output outputs/semantic_training --steps 1500
```

默认训练小MLP任务头，不训练Sonata骨干。要训练当前图语义边界头，追加`--graph --points data/train/points.npz`；点NPZ包含与特征逐行对应的`points_CAD_m`，或`coord`及`CAD_translation_m`。图边监督只使用训练掩码内的已知类别；可在标签NPZ追加`facade_weak_mask`作训练弱约束，此掩码不用于推理。

标签不自动从通用SKP名称推断，需人工类别映射或已核验语义面标注。

## 房体和两条风管参数头

输入文件夹：`input_contract.json`与`all.npz`；`all.npz`含15×96×96的`features`。用`release_neural.build_joint_inputs()`从CAD坐标观测、九类语义概率和原CAD外轮廓框架构建。目标NPZ含`parameters`（25），来自核验参考结构，不属于推理输入。

25参数顺序：房体cx/cy/长/深/底高/高度/角度；两条三节点风管的6个xy节点；两宽、两高、两底高。xy按原外轮廓框架归一化，z相对支承面除6m，角度为弧度。定义与解码见`roof_joint_geometry.py`。两路线可以不等长，结构分支共享截面是假设，应在具体建筑核验。

```bash
python -m zip2cad train-joint --inputs data/train/joint_inputs --targets data/train/object_targets.npz --output outputs/joint_training --steps 3000
python -m zip2cad infer-joint --inputs data/train/joint_inputs --checkpoint outputs/joint_training/checkpoint.pt --output outputs/joint_prediction.json
```

要保持已有房体分支不被后续风管训练破坏，可追加`--resume 已有checkpoint.pt --freeze-room`。此时创建/恢复独立房体与风管分支，只优化未冻结参数，并在保存前校验房体权重摘要完全不变。两分支代码与案例组合权重使用同一结构。

训练输出步数不足100时标记smoke-only，完整流程拒绝把此类权重导出为正式模型结果。本公共训练器是参数化后的研究基线，提供增强和米制损失；不声称与历史多轮A/B、续训及分支冻结逐位复现。已发布案例以冻结权重及哈希为复现依据。

## 在新视频流程中启用

配置`neural.enabled=true`，提供`sonata_repository`、`sonata_weights`、`semantic_checkpoint`和`geometry_checkpoint`。若需要单独Sonata环境，先在该环境运行同一项目；官方ZipMap可通过`reconstruction.python`在另一环境运行。

禁止把训练目标坐标塞入输入NPZ，再称为模型预测。报告训练/开发划分、尺度来源与训练参考是否参与评价；新建筑应另做独立验证。
