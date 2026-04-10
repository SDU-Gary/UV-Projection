# FaithC OpenGL Previewer

最小 C++ 交互预览器（GLFW + OpenGL + ImGui），支持：
- 导入 `GLB/GLTF/OBJ/PLY` 模型；
- 轨道相机交互（旋转/平移/缩放）；
- ImGui 调参并触发 FaithC 减面（`Apply` 按钮异步执行）；
- 显示网格统计、减面结果与状态信息。

## Build

```bash
cd viewer/opengl_previewer
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Run (direct)

```bash
./build/faithc_viewer \
  --mesh /abs/path/model.glb \
  --repo-root /abs/path/FaithC \
  --python-bin /home/kyrie/miniconda3/envs/faithc/bin/python \
  --bridge-script /abs/path/FaithC/tools/preview/run_faithc_preview.py \
  --work-dir /abs/path/FaithC/experiments/runs/preview_tmp
```

## Run (recommended)

通过项目 CLI 启动（会自动透传桥接脚本和工作目录）：

```bash
faithc-exp preview --mesh assets/examples/pirateship.glb
```

## Controls

- 鼠标左键拖拽：旋转
- 鼠标右键或中键拖拽：平移
- 滚轮：缩放
- `Apply`：按当前 FaithC 参数执行重建减面
- `Cancel`：软取消（不中断进程，只忽略本次回传结果）
