# 安装与外部依赖

## 基础环境

建议Python 3.11。Windows已作本地原生SDK验证；Linux可运行数据处理、训练及核心测试，原生SKP需本平台支持的官方SDK。

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux: source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m zip2cad doctor
```

Python 3.11以上只针对本项目核心代码。ZipMap要求的PyTorch/torchvision和Sonata要求的spconv/CUDA应分别遵循固定版本上游依赖；必要时采用两个环境，不要盲目覆盖现有CUDA包。

## 官方ZipMap

来源：[Haian-Jin/ZipMap](https://github.com/Haian-Jin/ZipMap)。源码提交、权重URL与SHA256在`models/manifest.json`。

```bash
python tools/setup_backbones.py zipmap --weights
python -m pip install -e '.[neural,reconstruction]'
```

适配器直接从固定源码目录导入ZipMap；无GLB输出时不导入官方交互可视化依赖。默认Windows使用`TORCH_COMPILE_DISABLE=1`。运行官方推理需要CUDA；`reconstruction.python`可以指向另一个环境的Python。权重约5.47GB，默认不会在每次推理时下载，也不会覆盖已存在但哈希不符的文件。

先按GPU型号安装匹配CUDA的PyTorch/torchvision，不要让上游训练项目的固定torch==2.6.0要求覆盖已验证的新GPU环境。本机测试组合为PyTorch2.11.0＋CUDA12.8、RTX5070Ti Laptop（约12GB）。其它硬件按PyTorch官方安装器选对应版本；本项目不把该本机组合声明为所有机器的强制依赖。

## 可选Sonata及任务头

来源：[facebookresearch/sonata](https://github.com/facebookresearch/sonata)。

```bash
python tools/setup_backbones.py sonata --weights
python -m pip install -e '.[neural]'
# 根据third_party/sonata/environment.yml / 官方README安装匹配CUDA的spconv等依赖。
```

推理使用冻结Sonata、关闭flash attention；当缺少torch-scatter时采用项目`segment_reduce`前向适配。该适配不用于宣称端到端骨干反向传播等价。仓库`models/pretrained/`已附语义头/几何头，使用配置见USE_PRETRAINED；新建筑可按需要微调。只有官方Sonata骨干，不会自动获得本项目九类语义和屋顶参数头。

## 官方SketchUp C SDK

SDK二进制不随源码包发布。安装支持平台的官方SDK后设置：

```powershell
$env:ZIP2CAD_SKETCHUP_SDK='D:/your-sdk/SketchUpAPI.dll'
python -m zip2cad doctor
```

Windows也会自动检查已安装SketchUp的SDK库。运行不需要控制SketchUp窗口、Ruby插件或应用UI。

## 可选图像研究依赖

`python tools/setup_backbones.py mobilesam --weights`只准备官方源码/权重。MobileSAM等图像约束研究并不等于默认14步流程会自动为任意屋顶完成细节。当前通用入口的适用范围见README。
