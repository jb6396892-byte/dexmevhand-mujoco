# 2026-10-03 新桌面视觉控制测试记录

> 命令勘误：下方历史 Qt 命令误写为 `tabletop_python.sh`，该视觉环境不含 PySide6，不能直接照抄。当前可复现 Qt 检查由 `154_run_contact_qt_checks.py` 统一使用独立 GUI 环境，见 [v2 记录](2026-10-03-tabletop-contact-v2.md)。历史测试结果和截图保留，不以新结果覆盖。

用户授权开始测试，并要求少量截图、答辩资料。本轮不训练、不修改原模型、不发布未通过候选。

## 主要命令

工作目录：`/home/smgbro/mujoconew/GITHUB`。GPU/X11、共享盘写入和旧 mujoco-py 导入锁使用批准的本机执行权限；不得把沙箱文件锁错误误判为驱动损坏。

```bash
bash scripts/137_tabletop_gpu.sh -m unittest discover -s tests

# 各用例 --output 必须使用新的目录，脚本拒绝覆盖。
bash scripts/tabletop_python.sh scripts/148_check_tabletop_qt.py \
  --scene first --goal transport \
  --output /media/smgbro/shared/visual_grasp/control-tests-v1/qt-first-final

bash scripts/tabletop_python.sh scripts/148_check_tabletop_qt.py \
  --scene first --cancel-step 100 \
  --output /media/smgbro/shared/visual_grasp/control-tests-v1/qt-user-stop

bash scripts/tabletop_python.sh scripts/148_check_tabletop_qt.py \
  --scene second --output /media/smgbro/shared/visual_grasp/control-tests-v1/qt-second-baseline

bash scripts/tabletop_python.sh scripts/148_check_tabletop_qt.py \
  --locked --output /media/smgbro/shared/visual_grasp/control-tests-v1/qt-lock

bash scripts/tabletop_python.sh scripts/148_check_tabletop_qt.py \
  --instruction '将杯子放到我手上' \
  --output /media/smgbro/shared/visual_grasp/control-tests-v1/qt-reject-handoff

bash scripts/tabletop_python.sh scripts/150_check_visual_tracker.py \
  --run /media/smgbro/shared/visual_grasp/control_runs/20261003T063820118456Z \
  --output /media/smgbro/shared/visual_grasp/control-tests-v1/tracking-regression-v2.json

python3 scripts/151_publish_visual_control_tests.py
```

最后一个脚本通过 `physics_environment()` 运行完整测试和 `122_verify_guard_revision.py verify`，复制两张截图及 JSON，并保存 SHA256 清单。测试工具退出码 0 只表示检查脚本完成；实际任务是否成功必须查看 `task_passed` / `end_to_end_grasp_passed`，本轮为 false。

## 中间失败没有删除

| 用例 | 步数 | 停止原因 |
| --- | --- | --- |
| qt-second-baseline | 0 | 近倒置参考超出视觉适配范围 |
| qt-first-v2 | 57 | 根部关节目标越限 |
| qt-first-v3 | 154 | 调整固定安装后仍有 Y 滑轨目标越限 |
| qt-first-v4 | 0 | 整段预检查发现视觉动作修正超过旧 0.25 上限 |
| qt-first-v5 | 432 | 手遮挡导致配准低于质量门槛 |
| qt-first-v6 | 512 | 鲁棒配准通过，但接近阶段没有有效指尖接触 |
| qt-first-final | 512 | 带源码哈希和事后真值诊断重跑，仍是接近技能未通过 |
| qt-user-stop | 101 | user_stop，停止耗时 0.5873 秒 |
| qt-lock / qt-reject-handoff | 0 | 默认锁和指令准入拒绝 |

初次用裸 conda 运行部分测试时，旧 mujoco-py 尝试创建 writable roots 外的构建锁，9 项导入错误；改用隔离 GPU 环境和本机权限后通过。最后回归 **211 项通过**，冻结语言校验输出 `Guard revision verified`。上游 MANO/chumpy 警告保留在 `quality.json`。

## 修改文件与产物

- 控制与场景：`src/fromrealhand/tabletop/{control,control_scene,vision_client}.py`。
- 视觉跟踪与界面：`src/fromrealhand/perception/tracking.py`、`src/fromrealhand/desktop/tabletop_window.py`。
- 候选配置：`configs/tabletop-control-candidate.json`，默认锁定。
- 运行入口：脚本 142--145；原 Qt 入口不变。
- 测试/诊断：脚本 147--151 与 `tests/test_tabletop_control_candidate.py`。146 是上一轮测试前清单工具。
- 报告、两张实测图与小型证据：`docs/presentation/tabletop_control_tests/`。
- 大体积 RGB-D、动作、运行日志：`/media/smgbro/shared/visual_grasp/control_runs/` 和 `control-tests-v1/`。

最终物理运行后的唯一跟踪算法文件变化是检测框合法性/裁边检查；随后缓存 23 帧、3 个负例和完整 211 项测试通过。物理源文件哈希与当前差异单独保存在 `publication-sources.json`，不能把不同源码版本混为一次完整验收。

## 尚需验收

下一步是接触约束的动作迁移与直立杯参考修复，不是增大训练轮次。完成当前开发场景的接近、三指对握、抬杯和搬运后，再冻结候选做独立留出评估。没有执行实机测试；没有修改旧 checkpoint；没有推送 GitHub。详细边界见 [测试报告](../TABLETOP_CONTROL_TEST_RESULTS.md)。
