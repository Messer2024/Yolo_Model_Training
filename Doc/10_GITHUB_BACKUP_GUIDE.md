# YOLO Studio - Git 版本控制与 GitHub 仓库备份指南

**文档版本**：v1.0.0  
**编制日期**：2026-08-20  
**项目名称**：YOLO Studio

---

## 1. 版本控制规范与分支管理策略

本项目遵循 **Git Flow / GitHub Flow** 最佳实践：

- `main`：稳定生产发布分支，仅包含经过严格测试的 Release 版本；
- `develop`：主开发分支，各功能合并集成的主干；
- `feature/*`：新功能或新模块开发分支（例如 `feature/auto-label-sam`）；
- `bugfix/*`：缺陷修复分支。

---

## 2. `.gitignore` 过滤规则

深度学习与 GUI 项目中包含大量大体积二进制权重文件（`.pt`, `.onnx`）、本地训练运行产物（`runs/`）以及虚拟环境，必须被严格忽略：

```gitignore
# Python 运行时与缓存
__pycache__/
*.py[cod]
*$py.class
.pytest_cache/
.coverage
htmlcov/

# 虚拟环境
venv/
env/
.env

# 数据集与图像缓存 (避免误传大文件至 Git)
data/
datasets/
*.jpg
*.jpeg
*.png
*.bmp
*.mp4
*.avi

# 模型权重与训练输出 (使用 Git LFS 或 Release 资产托管)
*.pt
*.pth
*.onnx
*.engine
*.tflite
*.weights
runs/
output/
dist/
build/
*.spec

# IDE 与编辑器临时文件
.idea/
.vscode/
*.swp
.DS_Store
```

---

## 3. GitHub 仓库初始化与全量备份步骤

### 3.1 本地 Git 仓库初始化
在项目根目录（`c:\Users\messe\OneDrive\Documents\Antigravity\YOLO Training Model`）下执行：

```bash
# 1. 初始化 Git 仓库
git init

# 2. 检查当前工作区状态
git status

# 3. 添加所有工程文档与代码
git add Doc/ src/ agents/ tests/ requirements.txt README.md .gitignore

# 4. 提交第一个里程碑版本
git commit -m "feat(init): Initialize YOLO Studio architecture, full documentation, agents & core codebase"
```

### 3.2 关联 GitHub 远程仓库并推推送备份

1. **在 GitHub 上创建新仓库**：
   - 登录 GitHub，点击右上角【New repository】；
   - 仓库名称填写：`yolo-studio` 或 `YOLO-Training-Model`；
   - 设为 `Public` 或 `Private`，**不要勾选** "Initialize this repository with a README"（因为本地已存在）。

2. **关联远程仓库并推送**：
   ```bash
   # 添加远程 GitHub 仓库 (请将 your_username 替换为您的 GitHub 用户名)
   git remote add origin https://github.com/your_username/yolo-studio.git

   # 重命名默认分支为 main
   git branch -M main

   # 推送至 GitHub
   git push -u origin main
   ```

---

## 4. 持续集成与发布 (GitHub Actions CI/CD)

项目在 `.github/workflows/ci.yml` 中配置了自动化测试工作流：
- 每次向 `main` 分支发起 Pull Request 时，自动启动 Ubuntu / Windows 虚拟机运行 `pytest` 自动化测试；
- 打 Release Tag 时，自动执行 PyInstaller 打包并将编译后的 `.exe` 压缩包上传至 GitHub Release Assets。
