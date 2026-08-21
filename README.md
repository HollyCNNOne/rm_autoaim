# rm_autoaim — RoboMaster 自瞄系统（场景 C：前哨站）

基于 RoboMaster 视觉组招新考核试题，实现完整的前哨站（Outpost）自瞄 Pipeline。
从视频帧中检测装甲板 → PnP 位姿解算 → 多目标跟踪预测 → 弹道解算 → 输出瞄准角。

## 技术栈

| 项 | 说明 |
| :--- | :--- |
| 语言标准 | C++23（xmake `set_languages("c++23")`，GCC 11 需降级为 `c++20`） |
| 构建系统 | xmake |
| 依赖 | OpenCV 4、Eigen 3、FFmpeg（libavcodec/avformat/avutil/swscale）、spdlog |
| 线程模型 | `std::jthread` + `std::stop_token`，V7.1 有界队列 + 实时调度 + CPU 亲和性 |

## 目录结构

```
rm_autoaim/
├── xmake.lua
├── .clang-format
├── include/rm_autoaim/
│   ├── Types.hpp          # 全局数据类型（Armor2D / ArmorPose / AimAngle 等）
│   ├── Reader.hpp         # 模块一：视频解码 + IMU 字幕轨解析
│   ├── Detector.hpp       # 模块二：装甲板检测（颜色分离 → 灯条 → 配对）
│   ├── Tracker.hpp        # 模块三：塔台中心 EKF 跟踪 + 预测软匹配 + 四态状态机
│   ├── Predictor.hpp      # 模块四：卡尔曼预测（CV 位置 + Singer 旋转）
│   ├── BallisticSolver.hpp# 模块五：弹道解算（重力 + 可选空气阻力）
│   ├── Pipeline.hpp       # 模块六：多线程编排（V7.1 有界队列 + 实时调度）
│   └── internal/
│       ├── Hungarian.hpp      # 匈牙利算法
│       ├── KalmanFilter.hpp   # 线性卡尔曼滤波器
│       └── TurretEKF.hpp      # 9D 塔台中心扩展卡尔曼滤波器
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

V7.1 线程模型：

| 线程 | 调度策略 | CPU 核心 | 通信机制 |
| :--- | :--- | :--- | :--- |
| Reader | SCHED_OTHER | 默认 | 有界队列（5 帧）→ Detector |
| Detector | **SCHED_FIFO** prio=99 | **核心 0** | 原子槽 → Tracker / Predictor |
| Tracker | SCHED_OTHER | 默认 | 原子槽 → Ballistic |
| Predictor | SCHED_OTHER | 默认 | 原子槽 → Ballistic |
| Ballistic | SCHED_OTHER | 默认 | 原子槽 → 主线程 |
| Render | SCHED_OTHER | **核心 1** | 信号队列（mutex+cv）← Detector |

## 模块要点（V7 最终架构）

### 模块二：Detector（检测）
- HSV 颜色分离 + CLAHE 增强 + 动态形态学核
- 灯条过滤（宽高比 / 面积 / 凸度，阈值放宽以保留可疑候选）
- 软评分配对：多维代价（高度差 / 角度差 / Y 偏移 / X 比例）+ 滑动窗口剪枝 + 冲突消解贪心
- 垂直重叠度 ≥ 50% 硬约束（拒绝跨甲板配对）
- 时序长边校验：EMA 跟踪装甲板长边，突变（>1.5×）拒绝异常配对

### 模块三：Tracker（跟踪）— V7 双循环架构
- **TurretEKF**：9D 扩展卡尔曼滤波器跟踪塔台中心 `[xc, vxc, yc, vyc, za, vza, yaw, vyaw, r]`
  - 跟踪对象从"三个独立装甲板"升级为"塔台中心 + 相位"
  - 丢帧时 EKF::predict() 持续外推，输出永不冻结
  - 相位切换时 handleArmorJump() 只改 yaw，塔台中心保持连续
- **预测驱动软匹配**：三维代价替代硬惯性
  - `cost = 0.50*(1-IoU) + 0.35*pred_error + 0.15*center_dist`
  - 匹配依据从"历史记忆"升级为"物理预测"
  - 匈牙利算法全局最优分配
- **四态状态机**：INACTIVE → TENTATIVE → CONFIRMED ↔ LOST
  - TENTATIVE：连续 3 帧命中才升级（抗单帧误检）
  - LOST：连续 15 帧丢失才回收（抗短暂遮挡，EKF 持续预测）
  - 匹配阈值随状态动态调整
- 后处理：中位数锚定 + 偏差限幅 + 四元数 SLERP 平滑

### 模块四：Predictor（预测）
- 位置：常速度（CV）卡尔曼滤波（x,y,z,vx,vy,vz）
- 旋转：Singer 模型（角度/角速度/角加速度）
- 真实 `dt`：由 `steady_clock` 测量帧间隔，杜绝固定 dt 的系统偏差
- 目标短暂丢失时 predict-only 保持状态（15 帧内不删除滤波器）

### 模块五：Ballistic（弹道）
- 重力模型，迭代求解 pitch 角（可选空气阻力 RK4）
- 三道物理防火墙：深度硬边界、变化率限幅（>40% 复用上帧）、输出角合理性校验

### 模块六：Pipeline（系统调度）— V7.1 三项优化
- **有界帧队列**（5-slot）：单槽覆盖 → 有界缓冲，吸收 Detector 6ms 瞬时抖动
- **实时优先级**：SCHED_FIFO prio=99 → nice(-20) → 默认，渐进降级，永不崩溃
- **CPU 亲和性**：Detector 绑定核心 0，Render 绑定核心 1，物理隔离 L3 缓存

## 设计哲学与思想演进

> 真正的进化不该被外部命名所定义，而应源于问题本身。

### 第一章：原始困境（V1-V3）

**最初面对的是一个纯粹的工程问题：塔台上有 3 个装甲板匀速旋转，如何让算法知道"谁是谁"？**

最初的直觉是："既然塔台是匀速旋转的，那么它的运动是有规律的。我能不能利用这个规律，而不是每一帧都重新认人？"

- V1-V2 引入帧计数器和周期模运算来预测下一个装甲板何时出现（`phase = frame_counter % period`）
- 这是"物理相位锁定"思想的雏形——物理规律应该约束算法行为
- 遇到的问题：固定周期无法应对帧率波动和丢帧；3 个独立槽位互相抢夺 ID，出现"三圈全是 0"
- **核心直觉形成**："我想跟踪的是那个转动的塔台，而不是三个独立的发光板"

> 这时已经有了"跟踪本体而非表象"的直觉——这与后来看到的华南虎方案内核惊人地一致。

### 第二章：匈牙利匹配与惯性（V3-V4）

**发现单纯依靠相位预测还不够，需要解决"检测框与轨迹如何匹配"的问题。**

思考："既然上一帧匹配过的组合，这一帧大概率还是同一对，我能不能给它们一些'惯性奖励'？"

- 引入匈牙利算法进行全局最优匹配
- 加入惯性奖励（`-0.30`）：如果检测框和某个轨迹上一帧配对过，就给予巨大优惠
- 这是"记忆机制"的首次尝试——连续运动应该被连续记住
- 遇到的问题：惯性奖励在强制推进时（旧目标消失，新目标出现）会"锁死"旧 ID，导致新目标被强行拉回旧槽
- **核心矛盾触及**："记忆需要周期性重置"

> 物理 1 号明明出现了，却被强行标成 0 号——惯性奖励过度保护了"历史记忆"。

### 第三章：状态机的诞生（V5）

**意识到"惯性奖励太暴力了，我需要更细腻的生命周期管理"。**

思考："一个目标应该有'刚出现'、'正在跟踪'、'暂时丢失'、'彻底消失'四种状态。不同状态应该有不同的行为规则。"

- 设计 `TENTATIVE → CONFIRMED → LOST → INACTIVE` 四态状态机
- 每个状态独立的命中/丢失计数阈值
- 这是"容错记忆"的体现——短暂遮挡不应该立即判死刑
- 新问题暴露：状态机虽然有了，但匹配代价依然依赖硬惯性（-0.30）；丢帧时数值冻结——因为没有测量值，所有状态都停止更新
- **核心需求浮现**："我需要一个'即使没有测量，也在推演'的底层动力"

### 第四章：EKF 的引入——思想融合（V6）

**此时同时拥有了三条线索：**

1. **物理规律**（帧计数器/相位锁定）→ 想跟踪"塔台本体"
2. **记忆机制**（惯性奖励/匈牙利匹配）→ 需要"连续识别"
3. **状态管理**（四态状态机）→ 需要"容错生命周期"

**最终思考汇聚成：**"我能不能用一个数学模型（EKF）同时承载这三个需求？让它既跟踪塔台中心（物理本体），又为匹配提供预测位置（替代硬惯性），同时在丢帧时持续推演（根治冻结）？"

**最终设计：**

| 原始直觉 | 工程实现 |
| :--- | :--- |
| "我想跟踪的是塔台，不是甲板" | EKF 状态向量 `[xc, yc, yaw, r, za]` — 塔台中心 + 相位 |
| "记忆应该基于物理规律，而不是硬绑定" | 预测代价匹配替代硬惯性 — 用 EKF 预测位置计算偏差 |
| "短暂丢失不应该立即死亡" | 四态状态机 + EKF 持续预测 — LOST 状态真正具备恢复能力 |

> 这是一条独自走过的路——从未见过华南虎或港科大的代码，但思考路径与它们殊途同归。因为解决的问题本身就是它们要解决的问题。后来看到华南虎和港科大的代码时，那不是"学习"，而是"印证"。

### 第五章：调度优化——让思想跑出极限（V7）

**V6 解决了"算法该怎么做"，但 V7 解决了"系统怎么能让它稳"。**

发现的问题：
- Detector 偶尔耗时 1 秒（被系统调度打断）
- 可视化后性能暴跌（Render 冲走缓存）
- 读帧和算帧之间有跳帧（通信机制无缓冲）

调度优化：

| 优化 | 解决的问题 | 实现 |
| :--- | :--- | :--- |
| **CPU 亲和性** | Render 冲走 Detector 的 L3 缓存 | Detector 绑定核心 0，Render 绑定核心 1 |
| **有界帧队列** | 单槽覆盖导致版本跳跃丢帧 | 5-slot 队列 + mutex + cv，吸收瞬时抖动 |
| **实时调度** | kswapd/kworker 抢占导致秒级毛刺 | SCHED_FIFO prio=99 → nice(-20) → 默认 |

> 这些优化体现的是对"物理世界"的深刻理解——算法要对，线程也要对。

### 思想图谱

| 原始直觉 | 对应的工程实现 | 后来发现的印证 |
| :--- | :--- | :--- |
| "我想跟踪的是塔台，不是甲板" | EKF 车体中心跟踪 | 华南虎方案 |
| "记忆应该基于物理规律，而不是硬绑定" | 预测代价匹配替代硬惯性 | 港科大方案 |
| "短暂丢失不应该立即死亡" | 四态状态机 + EKF 持续预测 | 港科大状态机 |
| "调度也应该被设计，而非被默认" | CPU 亲和性 + 有界队列 + SCHED_FIFO | 工业界通用实践 |

## 优化历程

- `docs/Detector_Optimization_Changelog_V1_to_V2.2.txt`：检测模块优化日志
- `docs/Tracker_Optimization_Changelog_V2.2_to_V3.3.txt`：跟踪模块优化日志
- `docs/Predictor_Optimization_Changelog_V3.3_to_V4.txt`：预测模块优化日志
- `docs/Ballistic_Optimization_Changelog_V4_to_V5.txt`：弹道模块优化日志
- `docs/Tracker_Pipeline_Optimization_Changelog_V5_to_V6.txt`：TurretEKF 双循环架构
- `docs/Tracker_Pipeline_Optimization_Changelog_V6_to_V7.txt`：系统级线程调度优化

版本演进：V1 → V2.2（Detector）→ V3.0/3.1/3.2/3.3（Tracker 物理约束）→ V4（Predictor 真实 dt）→ V5（Ballistic 鲁棒性）→ V6（TurretEKF 双循环架构）→ V7（系统级线程调度优化）。

## Git 提交规范

遵循 Conventional Commits：`feat(scope): description` / `fix(scope): description`。
每个提交完成一个逻辑单元，可独立编译。