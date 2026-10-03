# 桌面 RGB-D 前三步交付

2026-10-03。本轮完成独立桌面、RGB-D 标定验证及自动杯子位姿估计的基础版，不推进第 4 步抓取控制适配，也不改旧 Qt、阶段 3 或语言模型。

## 完成范围

| 步骤 | 本轮实现 | 验证 |
| --- | --- | --- |
| 1 场景 | DexMV 中源自 robosuite 的 TableArena；原 025_mug、香蕉、糖盒、芥末瓶、汤罐；固定 RGB-D 相机和 Adroit | 初始穿透 0，物体自由关节重力沉降，原生窗口连续显示 120 帧 |
| 2 RGB-D | 同一物理时刻输出 RGB、光轴米制深度、K 和相机到世界变换；点云与反投影 | 关闭 MSAA 后桌面平面误差约 0.00086 mm，投影/反投影数值误差小于 1e-10 像素 |
| 3 感知 | CUDA Grounding DINO 自动框杯子；深度平面去除/连通域提取；Open3D 多起点 ICP 估计位姿 | 3 个开发场景后冻结，在 20 个新位置/朝向布局做理想与噪声对照，另测无目标和深度失效 |

数字详见 [结果表](presentation/tabletop_rgbd/RESULTS.md)。所有“通过”均指场景、相机或感知准入，**不是视觉抓取成功**。Adroit 在这份感知场景中通过关节等式约束停放；杯子和四件物体自由运动。后续接控制时必须单独解除停放并重新做物理验收，不能直接运行旧策略。

## 如何查看

直接打开真正的 MuJoCo 窗口：

```bash
cd /home/smgbro/mujoconew/GITHUB
bash scripts/137_tabletop_gpu.sh scripts/139_view_tabletop.py --seed 100
```

关闭窗口或 `Ctrl+C` 退出。窗口用于观察桌面，不执行抓取。

查看原 Qt 语言抓杯实验台：

```bash
bash scripts/128_launch_qt.sh
```

旧 Qt 行为未变，仍是原标准场景。新 RGB-D 面板和指令触发视觉抓取属于后续第 4/5 步，尚未接入，不能混为一谈。

本轮三张图：[桌面 RGB](presentation/tabletop_rgbd/evidence/tabletop-rgb.png)、[深度预览](presentation/tabletop_rgbd/evidence/tabletop-depth.png)、[噪声输入的自动位姿](presentation/tabletop_rgbd/evidence/noisy-pose.png)。原始米制深度是 `.npy`，彩色图只是预览，不是可直接计算距离的深度数据。

## 实际算法与方案调整

1. 使用官方 `IDEA-Research/grounding-dino-tiny` 的固定权重，提示词 `a mug.`，图像中没有输入人工框或物体 ID 图。
2. 从深度反投影点云，用 Open3D RANSAC 找水平桌面；在自动框内去掉桌面，保留主要连通前景。没有用仿真 mask，也没有用颜色阈值直接指定红杯子。
3. 沿用训练杯的实际渲染网格，编译后还原到杯子本体坐标，0.8 尺度只应用一次；对全部偏航方向按 15 度初始化，使用 Open3D point-to-plane ICP 细化。
4. 根据点云覆盖率、配准残差、直立程度和桌面高度一致性准入。分数不是经标定的成功概率。输出相机系/世界系 4x4 位姿、检测分数、配准指标和耗时。
5. 每帧重新检测和配准，三帧开发序列已跑通；这是低频逐帧定位基线，不是高帧率运动跟踪器。

原任务书候选是 SAM 2 + FoundationPose。本机没有 `nvcc`，FoundationPose 需要的原生 CUDA 编译链尚未配置，因此采用不改旧环境的 Open3D 几何基线完成当前“已知直立杯”任务；**没有声称安装或运行 FoundationPose/SAM 2**。后续遇到明显遮挡、倾斜或几何配准失败时，再评估升级该模块。

## 数据隔离与验证边界

- 感知脚本只接收 `observations/` 和独立静态 CAD，不接收 `evaluation_only/`、场景 XML 或 MuJoCo 对象。
- 真值只由独立评测脚本在感知进程结束后读取。杯子 CAD 在不同布局间逐数组核验一致，不包含物体在场景中的位置。
- 20 个新布局只改变杯子约 ±2 cm、偏航约 ±15 度及干扰物朝向；相机、物体类别和总体布局固定。不是 20 个真实场景或未知物体。
- 噪声测试为同布局深度加 1 mm 独立高斯噪声、2% 随机无效像素；不代表完整真实相机误差模型。真实反光、透明、遮挡、标定漂移尚未验证。
- 无杯子时检测器仍出现误检框，当前由多目标歧义或几何门槛拒绝。负例零放行仅适用于这 4 项管线测试，不能解释成检测器零误检。
- 接近 1 秒的完整单帧定位目前适合静态任务初始化，不能作为 100 Hz 抓取控制反馈。需要后续异步跟踪与结果过期保护。
- 194 项项目单元测试通过；原冻结语言模型校验保留。未运行新 BC、DAPG 或 LoRA 训练。

## 文件与复现

大资源独立放在 `/media/smgbro/shared/visual_grasp/`：`runtime/`、`models/`、`development/`、`heldout-v1/`、`delivery-v1/`。代码在当前项目的 `tabletop/`、`perception/` 和脚本 134 至 141；旧模型与阶段 6 的 `lora/language/` 不移动。

已安装的视觉扩展包逻辑大小约 1.8 GB，检测权重目录约 659 MB；共享盘 exFAT 的实际分配占用更大。仅以只读方式复用原共享盘 PyTorch/Transformers，不在旧目录安装升级。

```bash
# 新机器/环境缺失时才安装；需要已有阶段 6 Python/PyTorch 基础
bash scripts/134_setup_tabletop_vision.sh
bash scripts/tabletop_python.sh scripts/135_download_tabletop_detector.py

# 用新 output 目录采集，已有目录不覆盖
bash scripts/137_tabletop_gpu.sh scripts/136_capture_tabletop.py \
  --seed 100 --frames 3 --output /media/smgbro/shared/visual_grasp/manual/scene-100
bash scripts/tabletop_python.sh scripts/138_estimate_tabletop.py \
  --observations /media/smgbro/shared/visual_grasp/manual/scene-100/observations \
  --mesh /media/smgbro/shared/visual_grasp/manual/scene-100/mug_model.npz \
  --output /media/smgbro/shared/visual_grasp/manual/scene-100/estimate
```

## 来源与许可记录

- 场景代码：本机 DexMV（Apache-2.0）中的 robosuite 派生 TableArena；[robosuite 原代码](https://github.com/ARISE-Initiative/robosuite)为 MIT。本轮调用既有模块，没有把第三方源码或大资产重复提交。
- YCB 外观/碰撞资产沿用已有 DexMV 安装，路径与 SHA256 见 `scene-assets.json`；[DexYCB 数据页](https://dex-ycb.github.io/)另明确数据集为 CC BY-NC 4.0，不能把代码许可证当成数据许可证。未重新发布第三方 mesh、纹理或原始视频；外部资产包分发需另核对应条款。
- [Grounding DINO 官方权重](https://huggingface.co/IDEA-Research/grounding-dino-tiny)为 Apache-2.0，修订 `a2bb814dd30d776dcf7e30523b00659f4f141c71`，本轮只推理、不训练。
- 点云方法：[Open3D 配准教程](https://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html)、[Open3D ICP](https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html)。
- 深度问题：[MuJoCo RGB/Depth 讨论](https://github.com/google-deepmind/mujoco/discussions/688)、[离屏与采样文档](https://github.com/google-deepmind/mujoco/blob/main/doc/programming/samples.rst)。本机实测通过关闭 MSAA 消除主要偏差，未采用经验平移补偿。

GitHub 默认 `main` 原先停在 `5dea040`；本轮先确认它没有独有提交，再快进到 Qt 版本 `1862317`，未强推、未删除旧分支或历史。
