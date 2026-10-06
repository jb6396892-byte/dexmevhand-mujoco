# 2026-10-06 随机桌面开发记录

## 需求

同一杯子随机位置；干扰物数量、位置不同；目标可随机或 Qt 调整；独立完整任务成功率超过 80%；同步 GitHub，少量截图和答辩说明。

## 开发过程

- 原 v4 模型、输出和旧 Qt 模式保留。新增配置 `tabletop-random-v5.json`；六个随机采样单元测试验证种子重复性、目标范围、物体间距、数量、参数拒绝和开发/测试隔离。
- `dev-initial`：四条开发回放均失败，一条零步关节工作区失败，三条完成抬杯但搬运未到位。
- `dev-hand-integral`：手部积分未解决问题，第一视频还出现支撑丢失；该方案未采用。
- `dev-servo-anchor`：第一视频采用保留手指动作的目标轨迹变换，两条通过。第二视频伺服重锚定后仍有约 26–28 mm 的相对偏差。
- `dev-proprioception`：第二视频使用已知初始杯位姿和三指正运动学位移估计，两条通过；不向策略提供实时物体真值。
- `dev-full`：完整 24 个开发工况全部通过。第一视频仅修改固定安装的关节坐标原点以改善行程余量，初始手部世界位置不随种子变化；未增加关节行程。
- Qt 初次测试因独立 UI 环境没有 NumPy 启动失败。移除界面对采样模块的直接依赖，以 Qt 自动值状态显示随机目标，未安装新依赖。`qt-tests` 保留失败日志。
- `qt-tests-v2` 六项功能通过；截图发现横向坐标输入使侧栏超宽，改为纵向 X/Y/Z。`qt-final` 是最终布局验收，不用旧截图替代。
- 最终 Qt 六项全部通过，停止响应约 0.54 s，心跳最大间隔约 0.055 s，未遗留子进程。
- 手动目标下、上边界分别为 `[-0.04,-0.08,0.14]` 和 `[0.04,0,0.19]` m；固定开发种子 15、四件干扰物，两视频各测两端，4/4 通过。单独存 `target-min-boundary`、`target-max-boundary`，不混入独立随机成功率。

## 复现命令

最终结果：冻结后两视频各 40/40，合计 80/80；最大目标误差 5.8573 mm，穿透峰值 0.761643 mm。40 个独立布局双视频均通过，Wilson 95% 下界 0.912378。所有 119400 个动作未截断，无执行中物体状态写入或额外力。最终 249 项软件测试通过，发布汇总 `delivery_passed=true`。

以下为原运行目录，脚本拒绝覆盖已有实验；重复执行需使用新输出目录。

```bash
bash scripts/137_tabletop_gpu.sh scripts/167_evaluate_random_tabletop.py \
  --output /media/smgbro/shared/visual_grasp/random-v5/dev-full

bash scripts/137_tabletop_gpu.sh scripts/170_freeze_random_tabletop.py \
  --development /media/smgbro/shared/visual_grasp/random-v5/dev-full/evaluation.json \
  --output /media/smgbro/shared/visual_grasp/random-v5/freeze.json

bash scripts/137_tabletop_gpu.sh scripts/167_evaluate_random_tabletop.py \
  --split heldout --freeze /media/smgbro/shared/visual_grasp/random-v5/freeze.json \
  --output /media/smgbro/shared/visual_grasp/random-v5/heldout

/usr/bin/python3 scripts/172_check_random_qt.py \
  --output /media/smgbro/shared/visual_grasp/random-v5/qt-final

bash scripts/137_tabletop_gpu.sh scripts/167_evaluate_random_tabletop.py \
  --seeds 15 --target-world -0.04 -0.08 0.14 --count 4 \
  --output /media/smgbro/shared/visual_grasp/random-v5/target-min-boundary

bash scripts/137_tabletop_gpu.sh scripts/167_evaluate_random_tabletop.py \
  --seeds 15 --target-world 0.04 0 0.19 --count 4 \
  --output /media/smgbro/shared/visual_grasp/random-v5/target-max-boundary

bash scripts/137_tabletop_gpu.sh scripts/171_publish_random_tabletop.py
```

最终发布脚本再次核对冻结哈希、80 条完整分母、源码回归、Qt 操作结果，再归档汇总。不会删除失败案例或把开发结果当作独立成功率。所有新原始实验放共享盘 `visual_grasp/random-v5/`；GitHub 只提交代码、配置、小型结果和两张截图。
