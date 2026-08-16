# rm_autoaim — RoboMaster 自瞄系统（场景 C：前哨站）

基于 RoboMaster 视觉组招新考核试题，实现完整的前哨站（Outpost）自瞄 Pipeline。
从视频帧中检测装甲板 → PnP 位姿解算 → 多目标跟踪预测 → 弹道解算 → 输出瞄准角。

## 技术栈

| 项 | 说明 |
| :--- | :--- |
| 语言标准 | C++23（xmake `set_languages("c++23")`，GCC 11 需降级为 `c++20`） |
| 构建系统 | xmake |
| 依赖 | OpenCV 4、Eigen 3、FFmpeg（libavcodec/avformat/avutil/swscale）、spdlog |
| 线程模型 | `std::jthread` + `std::stop_token`，5 级无锁原子槽 Pipeline |

## 目录结构

```
rm_autoaim/
├── xmake.lua
├── .clang-format
├── include/rm_autoaim/
│   ├── Types.hpp          # 全局数据类型（Armor2D / ArmorPose / AimAngle 等）
│   ├── Reader.hpp         # 模块一：视频解码 + IMU 字幕轨解析
│   ├── Detector.hpp       # 模块二：装甲板检测（颜色分离 → 灯条 → 配对）
│   ├── Tracker.hpp        # 模块三：多目标跟踪（固定沙盘 + 期望 ID 状态机）
│   ├── Predictor.hpp      # 模块四：卡尔曼预测（CV 位置 + Singer 旋转）
│   ├── BallisticSolver.hpp# 模块五：弹道解算（重力 + 可选空气阻力）
│   ├── Pipeline.hpp       # 模块六：多线程编排
│   └── internal/
│       ├── Hungarian.hpp  # 匈牙利算法
│       └── KalmanFilter.hpp
├── src/                   # 对应实现
├── tests/reader/          # Reader 单元测试
└── outpost.mkv            # 测试视频（不入库）
```

## 构建与运行

### 环境准备（Ubuntu 22.04，无 sudo）

1. 编译安装 OpenCV 4.10 到 `~/.local`（见下方提示）。
2. 安装 xmake：`curl -fsSL https://xmake.io/shget.text | bash`
3. 系统需已装：Eigen3、FFmpeg 开发库（`libavcodec-dev` 等）、GCC 11+。

> 若 `pkg-config` 找不到 OpenCV（OpenCV 4 默认不生成 `.pc`），在 `xmake.lua` 中
> 改用 `add_includedirs` / `add_linkdirs` / `add_links` 显式指定路径。
> GCC 11 不支持完整 C++23，请将 `set_languages("c++23")` 改为 `set_languages("c++20")`
> （本项目仅使用 C++20 特性，不受影响）。

### 编译

```bash
xmake f -m release
xmake build rm_autoaim
```

### 运行（双运行模式）

程序支持**两种运行模式**，由是否传入 `--debug-viz` 决定：

| 模式 | 命令 | 输出 | 适用场景 |
| :--- | :--- | :--- | :--- |
| **模式 A：生产模式** | `./rm_autoaim ./outpost.mkv` | 仅向终端/上位机输出瞄准角（pitch/yaw/t_fly），不写视频 | 性能最优，实机部署 |
| **模式 B：调试可视化** | `./rm_autoaim ./outpost.mkv --debug-viz debug_viz.avi` | 额外输出带装甲板标注的视频文件 | 离线调试、结果展示 |

```bash
# 模式 A：生产模式（默认，不输出视频，吞吐量最高）
./build/linux/x86_64/release/rm_autoaim ./outpost.mkv

# 模式 B：调试可视化（输出带标注的视频）
./build/linux/x86_64/release/rm_autoaim ./outpost.mkv --debug-viz debug_viz.avi

# AddressSanitizer 检查
xmake f -m asan && xmake build rm_autoaim
```

## Pipeline 架构

```
Reader → [FrameData] → Detector → [Armor2D[]] → Tracker → [TrackedArmor[]]
      → Predictor → [PredictedState[]] → Ballistic → [AimAngle[]]
```

每个模块独立 `std::jthread` 线程，模块间通过
`std::shared_ptr<std::atomic<std::shared_ptr<T>>>` 原子槽传递，配合版本号去重，
实现无锁单生产者-单消费者数据流。

## 模块要点

### 模块二：Detector（检测）
- HSV 颜色分离 + CLAHE 增强 + 动态形态学核
- 灯条过滤（宽高比 / 面积 / 凸度，阈值放宽以保留可疑候选）
- 软评分配对：多维代价（高度差 / 角度差 / Y 偏移 / X 比例）+ 滑动窗口剪枝 + 冲突消解贪心
- 垂直重叠度 ≥ 50% 硬约束（拒绝跨甲板配对）
- 时序长边校验：EMA 跟踪装甲板长边，突变（>1.5×）拒绝异常配对

### 模块三：Tracker（跟踪）
- 固定沙盘架构：3 个槽位对应 3 面装甲板，ID 循环 0→1→2
- 期望 ID 状态机：视觉证据驱动推进（连续 5 帧未匹配才切换），跳过容忍（8 帧）
- 惯性匹配：上一帧配对在代价矩阵中享 -0.30 奖励，抑制 ID 震荡
- 物理校验：depth [0.5, 15]m、pitch ±30°，变化率超限拒绝
- 中位数锚定 + 偏差限幅（±30%）抗漂移
- 四元数 SLERP 平滑姿态，避免欧拉角 ±180° 奇异性

### 模块四：Predictor（预测）
- 位置：常速度（CV）卡尔曼滤波（x,y,z,vx,vy,vz）
- 旋转：Singer 模型（角度/角速度/角加速度）
- 真实 `dt`：由 `steady_clock` 测量帧间隔，杜绝固定 dt 的系统偏差
- 目标短暂丢失时 predict-only 保持状态（15 帧内不删除滤波器）

### 模块五：Ballistic（弹道）
- 重力模型，迭代求解 pitch 角（可选空气阻力 RK4）
- 三道物理防火墙：深度硬边界、变化率限幅（>40% 复用上帧）、输出角合理性校验

## 测试结果简述（outpost.mkv，~142 FPS）

| 指标 | 结果 |
| :--- | :--- |
| Detector 平均耗时 | ~5.6 ms |
| Tracker 平均耗时 | ~71 μs |
| 最终输出帧率 | ~160 FPS（≥ 视频帧率） |
| ID 顺序正确率 | > 95%（0→1→2 循环） |
| 丢帧率 | < 1% |

## 优化历程

- `docs/Detector_Optimization_Changelog_V1_to_V2.2.txt`：检测模块优化日志
- `docs/Tracker_Optimization_Changelog_V2.2_to_V3.3.txt`：跟踪模块优化日志
- `docs/Predictor_Optimization_Changelog_V3.3_to_V4.txt`：预测模块优化日志
- `docs/Ballistic_Optimization_Changelog_V4_to_V5.txt`：弹道模块优化日志

版本演进：V1 → V2.2（Detector）→ V3.0/3.1/3.2/3.3（Tracker）→ V4（Predictor 真实 dt）→ V5（Ballistic 鲁棒性）。

## Git 提交规范

遵循 Conventional Commits：`feat(scope): description` / `fix(scope): description`。
每个提交完成一个逻辑单元，可独立编译。