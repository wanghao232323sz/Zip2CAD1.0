# GitHub发布与数据分离

本文件夹是仓库工作目录。发布时提交Git选择的文件；不要把整个目录拖入网页上传，否则会绕过`.gitignore`。当前未推送远程仓库。

首次上传的逐步操作见[GITHUB_PUBLISH.md](GITHUB_PUBLISH.md)。

## 随仓库公开

源码、训练与推理入口、测试、14步指南、配置、作者指定公开的九张README展示图，以及`models/pretrained/`三个已训练任务头、模型卡、MIT许可和SHA256清单。模型合计约3.25MB，直接进入普通Git，无需Git LFS或另找下载链接。`configs/project.pretrained.example.json`已启用这些模型。

官方ZipMap/Sonata/MobileSAM源码及大权重使用`tools/setup_backbones.py`按固定版本下载，不需要重复上传。SketchUp SDK由使用者安装。

## 可继续保留本地

| 内容 | 不公开的理由及影响 |
|---|---|
| 所有校园SKP/SKB，包括原始、参考和修复版本 | 按作者要求不公开；使用者提供自己的CAD |
| 校园视频、未选入README的原始照片、点云、训练标签与特征缓存 | 部署已训练模型不依赖这些数据；不公开会限制对原图书馆训练和指标的完整复现 |
| 历史实验、优化器状态、日志、本机路径与环境 | 不影响公开模型前向推理；完整原始训练状态仍保留本机 |
| 本机案例`artifacts/`、输出`outputs/` | 用于作者自己的复查；公共新建筑入口无需读取 |
| 第三方SDK和下载后的骨干 | 按官方安装或下载即可 |

原始训练数据不是推理必需品，因此可以保留私有。公开权重可能学习到训练建筑的形状先验；不公开SKP并不代表权重完全不含该形状信息。当前发布的是训练后的研究模型，跨建筑效果尚未验证。

## 发布前检查

```bash
python tools/verify_release.py
python tools/verify_pretrained_models.py
python tools/make_public_archive.py
git add .
git diff --cached --stat
git ls-files "*.skp" "*.skb"
```

最后一条应为空。不要使用`git add -f artifacts data`。`dist/Zip2CAD1.0-source.zip`包含公开代码和三个任务头，可解压后作为全新GitHub仓库内容上传；不要上传本机目录中的全部文件。源码包不含校园CAD，也不含原图书馆完整数据。

别人的使用顺序见`docs/USE_PRETRAINED.md`。克隆后仍须安装依赖、下载官方骨干、提供自己的视频/CAD及配准信息。模型公开不等于任意建筑已达到图书馆效果。
