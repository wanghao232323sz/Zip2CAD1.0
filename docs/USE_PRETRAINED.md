# 用公开模型处理自己的建筑

仓库已附带`models/pretrained/semantic.pt`、`geometry.pt`和`corners.pt`，无需自己的训练即可调用推理。先运行`python tools/verify_pretrained_models.py`检查哈希、加载与前向推理；这是软件兼容性检查，不是新建筑精度评价。

## 完整视频到SKP

1. 按INSTALL安装依赖、匹配GPU的PyTorch、官方SketchUp SDK及Sonata所需依赖。
2. 下载官方骨干：

```bash
python tools/setup_backbones.py zipmap --weights
python tools/setup_backbones.py sonata --weights
```

3. 复制`configs/project.pretrained.example.json`为`configs/my_building.json`。保留`project_root: ".."`，填写自己的视频、原始SKP、目标建筑PID及配准锚点文件。锚点格式见DATA。这里用当前公开语义头和房体/风管头，已设置`neural.enabled=true`。
4. 运行：

```bash
python -m zip2cad run --config configs/my_building.json --output outputs/my_building
```

原CAD保持原样，候选另存。分布不匹配的输入可能预测出过短管段并被几何校验拒绝；此时应核对坐标、结构适用性并微调，不能关闭校验强行导出。缺依赖、模型或配准时需先解决报错，不能直接跳过解释为完成修复。六角点分支不是14步默认推理的一部分；它只用于具备相应ABCDEF角色的局部结构。

## 适用范围与新建筑微调

当前模型来自单图书馆适配，没有独立建筑验证。语义头要求相同1088维Sonata特征契约；几何头固定输出一间屋顶房体和两条三节点风管。匹配这一结构的新建筑可先直接试用，再与观测核对。其它拓扑不能靠开启模型自动变成通用CAD修复；需要修改参数解码/训练目标或保留几何基线。

同拓扑新建筑可准备自己的训练输入与核验目标，然后微调：

```bash
python -m zip2cad train-joint --inputs data/train/joint_inputs --targets data/train/object_targets.npz --resume models/pretrained/geometry.pt --output outputs/my_joint_training --steps 1500
```

若房体已可靠、只希望调整风管，追加`--freeze-room`。完整输入格式见TRAINING。语义训练入口目前重新训练任务头，不提供语义头续训参数。

六角点模型强烈依赖角色及归一化方式；不要对新建筑使用图书馆的绝对坐标契约。新案例应先用自己的标注与契约微调：

```bash
python -m zip2cad train-corners --inputs data/train/corner_inputs.npz --targets data/train/corner_targets.npz --resume models/pretrained/corners.pt --output outputs/my_corners --steps 1500
```

原图书馆的最终混合细节包含照片约束和几何后处理；三个任务头不能单独保证复现全部细节，也不能把同建筑开发指标用作新建筑效果承诺。
