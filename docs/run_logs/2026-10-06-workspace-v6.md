# 2026-10-06 扩大随机桌面工作区

## 改动与过程

保留 v5 配置、模型和报告。新增 v6 配置，采样器、物理执行器、Qt worker、CLI 和冻结脚本增加显式配置选择；初始杯 XY 增加独立覆写和后端范围验证。Qt 通过两个页签控制初始位置与目标，不引入 NumPy 到隔离 UI 环境。

最初高度尝试 110–220 mm：8 个开发案例中 5 个成功。第一视频种子 31 是零步准入失败，诊断到第 1142 帧竖直关节 1 越过 0 下限；第二视频种子 30、33 为搬运后段关节目标越界。`dev-initial` 和 `workspace-diagnosis` 原样保留。

最终高度收敛为 140–200 mm；初始杯区域 15 × 15 cm、目标水平范围 15 × 18 cm 不缩回。第一视频固定安装偏置 `[0.02,0.04,0]` m，第二视频 `[-0.01,0,0]` m；模型滑轨范围、杯子参数和验收阈值不变，初始机械手世界姿态仍固定。

## 命令

```bash
bash scripts/137_tabletop_gpu.sh scripts/167_evaluate_random_tabletop.py \
  --protocol configs/tabletop-random-v6.json \
  --output /media/smgbro/shared/visual_grasp/random-v6/dev-full

bash scripts/137_tabletop_gpu.sh scripts/173_check_workspace_boundaries.py \
  --output /media/smgbro/shared/visual_grasp/random-v6/boundaries

/usr/bin/python3 scripts/172_check_random_qt.py \
  --protocol configs/tabletop-random-v6.json \
  --output /media/smgbro/shared/visual_grasp/random-v6/qt-final

bash scripts/137_tabletop_gpu.sh scripts/170_freeze_random_tabletop.py \
  --protocol configs/tabletop-random-v6.json \
  --development /media/smgbro/shared/visual_grasp/random-v6/dev-full/evaluation.json \
  --output /media/smgbro/shared/visual_grasp/random-v6/freeze.json

bash scripts/137_tabletop_gpu.sh scripts/167_evaluate_random_tabletop.py \
  --protocol configs/tabletop-random-v6.json --split heldout \
  --freeze /media/smgbro/shared/visual_grasp/random-v6/freeze.json \
  --output /media/smgbro/shared/visual_grasp/random-v6/heldout

bash scripts/137_tabletop_gpu.sh scripts/171_publish_random_tabletop.py \
  --run /media/smgbro/shared/visual_grasp/random-v6 \
  --boundary boundaries --history dev-initial workspace-diagnosis \
  --output docs/presentation/random_tabletop_v6/evidence
```

上面是原运行路径，脚本拒绝覆盖。复跑需要新目录。v5 的历史成功率不合并进 v6 的新独立集合；角点、界面和开发测试也不计入独立随机分母。

## 最终结果

- 开发 24/24，边界组合 16/16。边界取初始四角和目标盒两个对角端点，不是全部目标角点。
- 新留出种子 2001–2040，两视频各 40/40，总计 80/80。全部预注册工况均计入，无补抽或删除。
- 布局级双视频成功率 Wilson 95% 区间 91.24%–100%；不将同布局的两视频当作独立布局。
- 新留出最大目标误差 5.759229514 mm，最大穿透 0.763728716 mm。边界最大误差 6.904998866 mm，最大穿透 0.779980247 mm。
- 执行审计：120183 次学习动作调用，0 次动作裁剪，0 次执行中杯子位姿写入，无额外物体助力。
- Qt 两视频手动坐标、抬杯指令、停止、锁定、拒绝越界指令六项通过，界面响应检查通过。保留两张实际结果截图。
- 发布脚本重新验证冻结代码与模型 SHA-256，252 项软件测试通过，`delivery_passed=true`。
- 本轮没有重新训练；大文件仍在共享盘 `visual_grasp/random-v6/`，Git 仅收录代码、配置、量化证据及约 450 KB 截图。

复用模型：`/media/smgbro/shared/visual_grasp/dual-learn-v4/structured-bc/candidate.pt`。
