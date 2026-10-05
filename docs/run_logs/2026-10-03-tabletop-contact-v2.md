# 2026-10-03 接触约束、跟踪与搬运开发记录

用户要求：先稳定对握，再抬杯、搬运；遇到问题优先查开源方法；少量截图和答辩资料；若需要训练则停在训练前。

实际执行：无训练。完成第一视频、种子 0、侧向 RGB-D 的物理与 Qt 开发验证。旧模型、旧标准 Qt 和冻结阶段 3/6 凭据未变更。

## 执行与产物

1. `152_diagnose_contact_transfer.py`：用独立参考数据做 FK，不把直接设置诊断 qpos 当作物理成功。过渡结束后的手部坐标变换一致；识别原杯初态沉降与稳定锚点差异。
2. `153_test_contact_tracking.py`：只通过电机步进的开发试验。缓存初始视觉对照仅适用于完全相同的种子 0，缓存试验的真值用于评测和阶段判定，不能称作实时视觉闭环。正式运行必须有 `--live-vision`，这时动作和杯子成功指标使用视觉输入。
3. 接触规则增加至少两个相反法向、三指载荷与 0.5 秒保持；阶段末端等待，不跨阶段重置。
4. 原前视先后在 642 步点云不足、789 步倾角门槛、829 步遮挡时停止。没有删除失败记录，没有引入物体真值回退。
5. 增加固定侧向 RGB-D 后，开发物理运行完成 1529 步；Qt 加上严格视觉多帧确认后，两次完整运行均为 1542 步并通过。
6. 额外指尖增力增益 40 试验在 276 步发生动作饱和；最终设为 0。手腕反馈单变量对照末端误差降低 40.6%，两组均能对握。
7. `154_run_contact_qt_checks.py`：完整任务、停止、默认锁、第二参考拒绝、越界指令拒绝均通过。停止约 0.536 秒；无 Qt 管理工作进程残留。
8. `155_publish_contact_tracking.py`：确认最终物理运行源码 SHA 与当前一致；218 项完整回归通过，`Guard revision verified`；归档三张图与小型 JSON。

上游 MANO/chumpy 弃用、文件句柄和非可写数组警告仍存在，未为了消除警告改动原依赖。

## 主要命令

工作目录 `/home/smgbro/mujoconew/GITHUB`，所有输出目录必须是新目录。

```bash
bash scripts/137_tabletop_gpu.sh scripts/152_diagnose_contact_transfer.py \
  --output /media/smgbro/shared/visual_grasp/contact-control-v2/diagnosis-initial

bash scripts/137_tabletop_gpu.sh scripts/153_test_contact_tracking.py \
  --output /media/smgbro/shared/visual_grasp/contact-control-v2/live-transport-side \
  --kp 0 --root-gain 1 --goal transport --live-vision --camera rgbd_side

python3 scripts/154_run_contact_qt_checks.py \
  --output /media/smgbro/shared/visual_grasp/contact-control-v2/qt-regression

python3 scripts/155_publish_contact_tracking.py
```

`153` 是开发试验工具，结束后需要检查 `report.json.passed`，不能仅依据进程退出码；`154` 对预期结果汇总检查，任何一项不符会返回非零。

本轮曾误用视觉专用 `tabletop_python.sh` 启动 Qt，得到 `ModuleNotFoundError: PySide6`。随后切换到原有 `language/gui-runtime`，没有安装新依赖或重新配置驱动。脚本 154 固定正确隔离环境，避免再次混用。

本轮为新 153 记录源码哈希时，也发现相对 `__file__` 不能直接 `relative_to(ROOT)`；已改为先 `resolve()`，该次失败目录保留，没有仿真动作。两处都是工具入口错误，非物理抓取结果。

## 修改范围

- 新增：`src/fromrealhand/tabletop/contact_control.py`、`configs/tabletop-contact-v2.json`、脚本 152--155、`tests/test_contact_tracking.py`。
- 修改候选控制：`control_scene.py` 增加固定侧相机和可选固定安装坐标；`vision_client.py` 接受传感器名；`143_stream_visual_grasp.py` 集成 v2、阶段等待、对握法向和真实控制步数。
- 修改候选视觉：`tracking.py` 增加上一视觉位姿约束的点云筛选、独立的空中倾角限制；`150_check_visual_tracker.py` 保存失败诊断且失败返回非零。
- 修改候选界面：`tabletop_window.py` 标识开发验证、相机来源、实际步数和对握状态。旧 `window.py`、128 启动器未改。

## 归档与边界

最终物理运行：`/media/smgbro/shared/visual_grasp/control_runs/20261003T142231129777Z/`。
开发试验：`/media/smgbro/shared/visual_grasp/contact-control-v2/`。
小型答辩材料：`docs/presentation/tabletop_contact_v2/`，共享盘另有 `delivery/` 副本。

源锚点、目标定义、相机及阶段时间均有改变，不能把 v1 与 v2 的结果当作严格单变量成功率对照。只对手腕反馈关闭/开启的两次缓存初态试验作单变量误差比较。

第二视频仍执行前拒绝；没有独立新布局评估，没有实机、放杯或倒水测试。默认自动执行继续锁定。未运行 DAPG、BC 或 LoRA 训练，未提交或推送 GitHub。
