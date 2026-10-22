# 数据与本机案例保留

源码仓库包含自有训练任务头`models/pretrained/`；不含校园SKP、原始训练数据、官方大骨干、完整特征缓存或原生SDK。

## 从头运行所需

| 输入 | 谁提供 | 使用位置 |
|---|---|---|
| 实景视频 | 用户自行采集 | Step01–04 |
| 原始SKP | 用户已有图纸 | Step05、原模型保护 |
| 尺度与配准对应 | 用户核对稳定外部结构 | Step06 |
| ZipMap官方权重 | setup_backbones下载并核验 | Step03 |
| 可选Sonata、语义头、几何头 | 官方骨干＋仓库公开任务头，可选自己微调 | Step09–10 |
| 可选完整参考SKP | 训练/评价单独准备 | 监督或Step12 |

## 本机图书馆1.0

`artifacts/library1.0/release`保留之前92文件交付成果。`joint_inputs`、`joint_supervision`、`corner_supervision`保存训练/推理所需的小型结构化资料。`original_source_snapshot`保留原项目全部Python源码、配置和测试快照，以免整理时丢失旧算法细节；这些属于私有历史档案，不是推荐公共入口。

案例中的参考SKP与预测SKP分别存储。公开提交时不要通过`git add -f artifacts`绕过忽略规则。当前有效自有权重已单独公开到models/pretrained，校园案例数据仍可保持本地；部署这些权重不需要原始训练数据。

本机大视频和官方权重可用相同卷的硬链接保留到新目录；这节省重复磁盘占用。训练与下载程序不得原地改写这些只读输入。新案例运行输出始终另存。

## 配准锚点格式

参考`examples/registration_anchors.example.json`。`observed_points`是融合点云三维坐标，`CAD_points`是同物理位置的CAD米制坐标。至少4点，分布应避免退化。图像像素不可直接填入这两个三维字段。
