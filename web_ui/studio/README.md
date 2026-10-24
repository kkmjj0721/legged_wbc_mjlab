# 本地 URDF Studio 预览

迁移工作台使用用户指定的 [URDF Studio](https://urdf.enkeebot.com/) 项目提供的 `RobotCanvas`，不是外部网站 iframe。其源项目为 [OpenLegged/URDF-Studio](https://github.com/OpenLegged/URDF-Studio)，当前固定于提交 `23b96e2f3d5b1b68ad9629925d4479e9847d35a5`；上游组件包版本为 `@urdf-studio/react-robot-canvas@0.1.0`。接入时 npm registry 尚未提供该包，因此保留了实际组件、解析器与依赖模块的源码闭包，并在本地打包。

`dist/` 是随项目交付的运行资源。日常启动 Flask 不需要 Node.js、npm、外部网站、CDN 或网络，模型也不会上传外部服务。只加载当前导入快照中明确引用的网格／纹理，切换模型或停止预览后旧 token 立即失效。

## 使用范围

- XML / URDF 均显示最终导入的 MJCF 模型，保留原对象名与资源几何，使用上游 MJCF 解析器和 RobotCanvas 渲染。
- 本地旋转、平移、缩放；外观／碰撞、关节轴、质心、惯量与足端标记开关；“适应视图”重新取景。
- 主页面修改关节角、机体位置与四元数后同步显示，足端坐标仍由 MuJoCo CPU 前向运动学计算。浏览器渲染不推进仿真，也不用于测量动力学。
- 姿态编辑使用主页面控件；该嵌入组件只读，不提供完整 URDF Studio 的编辑、AI 或云端服务。
- 需要浏览器 WebGL2。关闭预览、切换模型和页面退出时释放旧模型与 Three.js 资源。

## 重建

只有修改本目录源码时才需要 Node.js 20.19+（本次使用 Node.js 24.20.0、npm 12.2.0）。从仓库根目录执行，缓存和临时文件均留在本项目：

```bash
mkdir -p .artifacts/robot-migration-build/tmp .artifacts/robot-migration-build/npm-cache
export TMPDIR="$PWD/.artifacts/robot-migration-build/tmp"
export npm_config_cache="$PWD/.artifacts/robot-migration-build/npm-cache"
npm --prefix web_ui/studio ci --ignore-scripts --no-audit --no-fund
npm --prefix web_ui/studio run build
```

`package-lock.json` 固定 npm 依赖；`vendor/UPSTREAM.json` 记录每个未改动上游源文件的 SHA-256，重建先验证完整性。`vite.config.mjs` 仅在构建时将上游 `/fonts/` 和 HDRI 路径改为本地 `/studio/` 前缀，源文件保留原样。上游源码不包含完整工作台或 USD 运行时；`public/wasm` 是 OBJ、Collada 网格解析助手。

## 许可与归属

- 上游 URDF Studio 及本地保留的 OBJ / Collada 网格助手：Apache-2.0，见 [vendor/LICENSE](vendor/LICENSE)。
- 页面集成代码位于 `src/`；上游文件位于 `vendor/src/`，二者分开维护。
- Fira Code 与 JetBrains Mono 字体：SIL Open Font License 1.1，许可在 `public/fonts/`，随构建复制。
- `potsdamer_platz_1k.hdr`：Poly Haven，CC0；仅本地使用。来源与许可见 [Poly Haven](https://polyhaven.com/a/potsdamer_platz)、[许可](https://polyhaven.com/license)。
- 运行资源包含 `/studio/THIRD_PARTY_NOTICES.txt`，由构建脚本收集所打包 npm 运行依赖的许可证和上述归属信息。
