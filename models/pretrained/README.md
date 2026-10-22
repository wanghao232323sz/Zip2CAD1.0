# Zip2CAD公开任务模型 v1

作者将本目录的自有任务头权重按MIT许可公开（见LICENSE）。不包含或重新授权ZipMap、Sonata或MobileSAM骨干；其权重由官方获取。

| 文件 | 结构与用途 | 训练记录 |
|---|---|---|
| semantic.pt | 冻结Sonata 1088维特征上的GraphRoofHead，9类语义，包含特征均值/标准差 | graph_boundary_facade，1500步 |
| geometry.pt | 房体与风管独立分支，25参数，保护房体权重 | 组合已训练分支，房体/风管各有效7500步；组合本身无新训练 |
| corners.pt | 44,440参数PointNet attention，六个固定角色ABCDEF | 累计3000步，metric_edges_v2 |

模型权重与本机训练版本逐张量相同。semantic仅移除优化器、随机状态及训练日志，保留推理所需归一化。SHA256和来源checkpoint哈希见manifest.json；不会加载原始参考SKP进行推理。

训练来源为单图书馆真实ZipMap观测、核验监督及部分程序化增强；不存在独立建筑泛化评测。几何头仅适用于一间房体和两条三节点风管；角点头依赖六个角色及坐标契约。模型可能保留训练建筑形状先验，无法保证未知建筑效果。不得将合成前向冒充精度验证。

使用见`docs/USE_PRETRAINED.md`；校园CAD、原始训练数据和最终混合SKP不随仓库发布，因此不提供原数据完整复现承诺。
