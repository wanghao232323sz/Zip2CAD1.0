# 首次发布到GitHub

当前文件夹已有Git仓库和公开文件选择规则。操作目录必须是包含本README和pyproject.toml的Zip2CAD1.0根目录，不是旧的聊天目录。

## Step1 在GitHub创建空仓库

登录 https://github.com 后打开 https://github.com/new 。建议仓库名`Zip2CAD`，可见性选`Public`。不要勾选初始化README、添加.gitignore或License，因为本地已经提供。点击Create repository，复制Quick setup中的HTTPS仓库地址。

GitHub账户名以网站登录账户为准，不能仅根据本机git user.name推断。

## Step2 打开本地PowerShell

在Zip2CAD1.0文件夹的资源管理器地址栏输入`powershell`后回车。在终端运行：

```powershell
Get-Location
git status --short
git remote -v
```

没有remote输出表示尚未绑定远程。若已有origin，先核对它是否正是目标仓库；不要把代码推送到不相关仓库。

## Step3 检查公开内容

使用已安装项目依赖的Python执行：

```powershell
python tools/verify_release.py
python tools/verify_pretrained_models.py
git add .
git diff --cached --stat
git ls-files -- '*.skp' '*.skb'
git ls-files -- models/pretrained docs/assets/comparisons
```

SKP/SKB一行应没有输出；三个任务头、九张比较图及模型/图片说明应在公开列表。`artifacts/`、`data/`、`outputs/`、`third_party/`及环境目录保持忽略。不要使用`git add -f`，也不要把整个本机文件夹拖到网页上传。

## Step4 创建本地提交

```powershell
git commit -m "Release Zip2CAD 1.0 with pretrained heads and visual comparisons"
git branch -M main
```

若提示Author identity unknown，仅为本仓库设置姓名和邮箱，再执行commit。姓名、邮箱需要填写自己的真实选择；可使用GitHub设置中提供的隐私邮箱。

```powershell
git config user.name "YOUR_NAME"
git config user.email "YOUR_EMAIL"
```

如果提示nothing to commit，运行`git log -1`检查是否已有提交；不用反复创建相同提交。

## Step5 绑定并推送

将下面地址替换成Step1复制的实际HTTPS地址：

```powershell
git remote add origin https://github.com/YOUR_ACCOUNT/Zip2CAD.git
git remote -v
git push -u origin main
```

若origin已经存在且确认地址正确，跳过remote add。若地址错误，用`git remote set-url origin 实际HTTPS地址`修正。首次推送可能弹出Git Credential Manager浏览器登录；按提示登录GitHub。不要把GitHub密码或访问令牌写到项目文件和聊天中。

出现rejected或fetch first时，先检查远程仓库是否创建了额外README/License提交；不要直接强制推送。保留完整报错再处理。

## Step6 检查发布结果

打开GitHub仓库首页，确认三组对比图，每组均按实景、原SKP、新SKP排列；点击可打开原图。图片保留原比例，视角并非完全相同，属于作者提供的定性展示，不代表三个独立模型测试。

确认models/pretrained中有semantic.pt、geometry.pt、corners.pt，docs/USE_PRETRAINED.md可打开。搜索仓库文件应找不到校园SKP/SKB或本机私有数据目录。Actions中的测试结果用于软件检查，不能当作新建筑准确率。

## 后续更新

```powershell
git add .
git diff --cached --stat
git commit -m "Update project documentation and results"
git push
```

大模型骨干仍由官方脚本下载；仓库内公开的三个自有任务头合计约3.25MB，不需要Git LFS。源码ZIP仅作下载备份；GitHub首页展示README要求README位于仓库根目录，而不是只上传一个ZIP。

官方操作说明：https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github
