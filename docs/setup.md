# 安装配置文档

## 系统要求

- **操作系统**：推荐使用 Ubuntu 22.04
- **显卡**：Nvidia 显卡  
- **驱动版本**：建议使用 550 或更高版本  

---

## 1. 创建虚拟环境

建议在虚拟环境中运行训练或部署程序，推荐使用 uv 创建虚拟环境。如果您的系统中已经安装了 uv，可以跳过步骤 1.1。

### 1.1 下载并安装 uv

UV 是一款轻量、高速的 Python 包与环境管理工具。使用以下命令下载并安装：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

安装完成后，更换一下 uv 的源：

```bash
mkdir -p ~/.config/uv
cat <<EOF > ~/.config/uv/uv.toml
[[index]]
url = "https://pypi.tuna.tsinghua.edu.cn/simple"
default = true
EOF
```
---

## 2. 安装

### 2.1 下载

通过 Git 克隆仓库：

```bash
git clone https://github.com/kkmjj0721/legged_wbc_mjlab.git
```

### 2.2 安装依赖并创建环境

```bash
uv python install 3.11
uv sync
```

## 总结

按照上述步骤完成后，您已经准备好在虚拟环境中运行相关程序。若遇到问题，请参考各组件的官方文档或检查依赖安装是否正确。

