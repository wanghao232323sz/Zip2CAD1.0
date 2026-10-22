# 当前图书馆梁宽、端架与立柱的完整程序

除通用视频→SKP14步主线外，当前最后一版梁宽、图片分割、多视图直线、连通立柱算法也已整理成可执行公开源码：

- `research/library_details/segment_beams.py`：冻结MobileSAM，神经梁种子投影提示与深度扫掠。
- `triangulate_sections.py`：LSD平行线、跨视图截面边、三边矩形与一对一支撑。
- `fit_front_frames.py`：多视图掩膜共识，前梁面/立柱候选，线约束校正。
- `refine_posts.py`：立柱专用提示框和保持连通的向下延伸。
- `tools/run_library_details.py`：按依赖顺序串联以上程序，另存原生SKP。

所有输入由`--case`目录的`inputs.json`声明，输出由`--output`指定；没有本机路径或日期实验目录依赖。算法核心在`roof_multiview_lines.py`、`roof_mask_surface.py`、`roof_image_sections.py`。

本机案例数据在`artifacts/library1.0/advanced`，其中有原始照片、相机、已有神经预测种子和前次掩膜证据；不包含训练参考屋顶的读入步骤。相机经过此前同场景校正，仍是开发条件。

```bash
# 用配置好的Sonata/MobileSAM环境，完整重新分割、重建：
python tools/run_library_details.py --case artifacts/library1.0/advanced --output outputs/library_details_new
# 节省时间，可明确重用已生成梁掩膜，后续线与立柱仍重新计算：
python tools/run_library_details.py --case artifacts/library1.0/advanced --output outputs/library_details_cached_masks --reuse-beam-masks
```

需要CUDA、MobileSAM和SDK。此流程仍有一房体/双端架的案例假设、视图选择及不确定侧条；它不是未经配准即可用于任意建筑的全自动端架系统。输入中的前次候选是算法中间产物，不是参考SKP回填。已有完整图三模型继续单独标注为参考。

公开仓库不附校园私有数据；其它用户需提供同一格式的照片/相机/观测种子，或取得可公开的案例资料。主线可独立用自己的视频/CAD运行。
