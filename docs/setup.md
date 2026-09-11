# 安装配置文档

## 系统要求

- **操作系统**：推荐使用 Ubuntu 22.04
- **显卡**：Nvidia 显卡  
- **驱动版本**：建议使用 550 或更高版本  

---

## 一条命令完成安装

在仓库根目录执行：

```bash
bash setup.sh
```

这个命令会自动完成 uv 检查/安装、uv 管理的稳定版 Python 3.11 准备、Python 下载源测速、PyPI 镜像测速、项目依赖安装，以及按本机 CUDA/驱动选择 PyTorch。Python 版本会读取项目根目录的 `.python-version`（也可用 `UV_PYTHON_VERSION` 覆盖）。已经存在可用的 uv Python 3.11 时会直接复用，不会重复下载。脚本会自动使用持久化缓存（默认 `~/.cache/uv`）和 Python 安装目录（默认 `~/.local/share/uv/python`）。

镜像会先并行测速真实 wheel 文件并按下载速度排序；实际安装时项目依赖每个镜像默认最多等待 300 秒。如果某个镜像连接成功但下载长时间没有完成，脚本会终止本次尝试并自动切换下一个镜像，不会一直卡在第一个地址。Python standalone 默认使用 4 路分段并行下载，单个分段无数据 20 秒就会判定通道异常并回退单流下载；文件保存到持久化缓存后再交给 uv 校验和安装。下载通道默认最多等待 600 秒。可以通过 `--sync-timeout` 和 `--python-install-timeout` 调整上限，设置为 `0` 表示不限制。

部署仿真（sim2sim）可以直接执行：

```bash
bash setup.sh --extra sim2sim
```

安装脚本的详细选项：

```bash
bash setup.sh --help
```

如果服务器需要代理，可以在执行一键脚本前设置 `UV_PROXY`；脚本会同时传给 uv、curl 和 Python 下载器：

```bash
UV_PROXY=http://user:password@proxy.example.com:8080 bash setup.sh
```

如果服务器完全无法访问外网，可以在有网机器准备 wheelhouse 后传到服务器，再使用缓存和 wheelhouse 离线安装：

```bash
UV_WHEELHOUSE=/data/legged-wheels UV_OFFLINE=1 bash setup.sh
```

离线模式不会测速、不会访问 PyPI，也不会下载 Python；要求对应的 uv Python 和 wheel 文件已经存在。

## 手动分步方式

如果不希望使用一键脚本，可以先安装 uv，再运行自动安装助手：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

然后在仓库根目录执行：

```bash
python3 tools/uv_auto_sync.py
```

脚本会并行测试 PyPI、清华、阿里、USTC、腾讯、华为云和上海交大等镜像，选择当前网络下响应速度最快的地址，然后执行 `uv sync`。如果首选镜像在安装过程中失败，会自动切换到测速结果中的下一个可用镜像。

默认会测试项目内置的常见公共源（PyPI、国内高校镜像和云厂商镜像），并发测速后选最快的一个。互联网不存在一个可靠的“全部源”列表，镜像也会变更或下线；如果你有内部源，可以把它们追加到文件中一起测试：

```text
# mirrors.txt，每行一个 PEP 503 simple 地址
https://pypi.example.com/simple
https://mirror.example.com/repository/pypi/simple
```

```bash
python3 tools/uv_auto_sync.py --index-file mirrors.txt
```

使用 `--index` 或 `--index-file` 时，脚本只测试你指定的地址；不指定时才使用内置公共源列表。测速读取真实 wheel 文件前缀，而不是只比较首页响应时间。

实际安装尝试默认有 300 秒超时；超时会自动终止当前 uv 进程并切换下一个镜像。项目依赖同步成功后只使用该镜像继续安装一次 PyTorch，不会因为 PyTorch backend 失败而重复执行整套项目同步。Python standalone 下载默认有 600 秒超时，例如：

```bash
python3 tools/uv_auto_sync.py --sync-timeout 120
```

如果需要调整 Python 下载超时：

```bash
bash setup.sh --python-install-timeout 1200
```

默认使用 4 路分段并行下载 Python；如果当前网络或代理不支持 Range 请求，可以改成单路顺序下载：

```bash
bash setup.sh --python-download-workers 1
```

下载成功后，压缩包会保存在 `UV_CACHE_DIR/python-standalone-mirror`，后续安装同一版本会直接使用缓存。

也可以用环境变量覆盖一键脚本的默认缓存和网络参数：

```bash
UV_CACHE_DIR=/data/uv-cache \
UV_PYTHON_INSTALL_DIR=/data/uv-python \
UV_HTTP_TIMEOUT=60 \
UV_HTTP_RETRIES=5 \
UV_CONCURRENT_DOWNLOADS=8 \
bash setup.sh
```

PyTorch 默认单独允许最多 30 分钟下载（CUDA wheel 较大），可用 `--torch-timeout` 调整；设为 `0` 表示不限制。

PyTorch 不直接照搬锁文件里的固定 CUDA wheel。脚本会优先读取 `nvidia-smi` 报告的驱动 CUDA 版本；只有系统没有 `nvidia-smi` 时才尝试读取 `nvcc`，自动选择兼容的 PyTorch 后端，例如 CUDA 13.0 对应 `cu130`、CUDA 12.4 对应 `cu124`；没有可用 NVIDIA 驱动时安装 CPU 后端。这样同一份代码可以在不同 CUDA/驱动机器上复用。

部署仿真（sim2sim）还需要额外依赖：

```bash
python3 tools/uv_auto_sync.py --extra sim2sim
```

如需手动覆盖自动判断：

```bash
python3 tools/uv_auto_sync.py --torch-backend cu124
python3 tools/uv_auto_sync.py --torch-backend cpu
```

如果所在网络有内部 Python 发行包镜像，可以覆盖 Python 下载通道（可重复传入多个地址参与测速）：

```bash
python3 tools/uv_auto_sync.py \
  --python-mirror https://mirror.example/python-build-standalone
```

Python standalone 也支持用文件提供多个下载源：

```text
# python-mirrors.txt，每行一个 uv --mirror 地址
https://mirrors.ustc.edu.cn/github-release/astral-sh/python-build-standalone
https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone
```

```bash
python3 tools/uv_auto_sync.py --python-mirror-file python-mirrors.txt
```

也可以通过 uv 的标准环境变量指定 PyPI 镜像或 Python 镜像：

```bash
UV_DEFAULT_INDEX=https://pypi.example.com/simple \
UV_PYTHON_INSTALL_MIRROR=https://mirror.example/python-build-standalone \
bash setup.sh
```

`UV_INDEX_URL` 在当前 uv 中仍可兼容使用，但已经被标记为弃用，因此一键脚本和文档优先使用 `UV_DEFAULT_INDEX`。

只查看测速结果、不安装依赖：

```bash
python3 tools/uv_auto_sync.py --dry-run
```

也可以只测试指定的内部镜像：

```bash
python3 tools/uv_auto_sync.py --index https://pypi.example.com/simple
```

脚本通过 `uv sync --default-index` 选择下载通道；当索引源改变时，uv 可能同步更新 `uv.lock` 中记录的源地址，请按项目协作约定决定是否提交该变更。

## 总结

按照上述步骤完成后，您已经准备好在虚拟环境中运行相关程序。若遇到问题，请参考各组件的官方文档或检查依赖安装是否正确。

