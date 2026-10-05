# 两视频桌面控制修复与训练前冻结

## 执行记录

1. `156_inspect_dual_reference.py`：只在单独参考仿真做 FK 诊断；不能用该结果当物理成功。
2. `153_test_contact_tracking.py`：保留失败轨迹，依次排查桌面净空、接触等待、检测歧义、遮挡旋转跳变、搬运偏差。缓存初态试验仅作电机筛查，正示范要求 `--live-vision`。
3. 第一轮八工况通过 5/8，其余在预检查或阶段门控安全停止。加入受限姿态映射和足够等待后，再固定参数完整复测，没有运行保留种子。
4. `157_collect_dual_visual_demos.py`：`frozen-development` 同版八工况 8/8；源码 SHA 保存在各回放目录。
5. `158_package_dual_visual_inputs.py`：八条示范、32 个阶段片段；验证前动作对齐、完整阶段、真值后审计和源码一致，训练集单独拟合归一化，139 维共享/八分支网络仅前向检查通过。
6. `159_check_dual_tabletop_qt.py`：实际语言模型、视觉进程和电机仿真；两视频成功，停止/锁定/越界拒绝通过。GPU 渲染正常。
7. `161_prepare_tabletop_bc.py`：共享和分支残差入口分别做默认 dry-run，训练配置锁定，没有执行 Adam、反向传播或权重保存。
8. `160_publish_dual_contact_evidence.py`：源码核对、完整回归、原语言门控核验，归档三张图、小型指标和失败历史。

## 复现命令

以下目录已存在，不要覆盖。复现时换新的输出目录。

```bash
bash scripts/137_tabletop_gpu.sh scripts/157_collect_dual_visual_demos.py \
  --output /media/smgbro/shared/visual_grasp/dual-contact-v3/frozen-development

bash scripts/137_tabletop_gpu.sh scripts/158_package_dual_visual_inputs.py \
  --collection /media/smgbro/shared/visual_grasp/dual-contact-v3/frozen-development \
  --output /media/smgbro/shared/visual_grasp/dual-contact-v3/training-inputs

/usr/bin/python3 scripts/159_check_dual_tabletop_qt.py \
  --output /media/smgbro/shared/visual_grasp/dual-contact-v3/qt-final

bash scripts/137_tabletop_gpu.sh scripts/160_publish_dual_contact_evidence.py
```

## 文件范围

- 新增：`functional_reference.py`、`association.py`、`training_inputs.py`，三个 `tabletop-dual-v3-*.json`，脚本 156–161，功能映射/训练输入测试及本文档、答辩材料。
- 修改当前桌面候选：`control.py`、`contact_control.py`、`control_scene.py`、`tracking.py`、142/143/153、`tabletop_window.py` 及相关测试。
- 保留旧真实视频、示范、checkpoint、v2 资料和其他 dirty worktree 内容。没有改动已冻结的语言模型权重或原标准场景界面。
- 所有视觉抓取大数据继续在共享盘 `visual_grasp`。Qt 调用语言模型产生的语言执行日志仍由原语言系统放在 `lora/language/desktop_runs`，没有移动模型。

## 环境记录

一次直接用系统默认 Python 环境跑全套测试时，旧 `mujoco_py` 导入触发只读编译锁错误；改用项目已有 `137_tabletop_gpu.sh` 隔离环境后回归通过，没有重装环境。一次 `python` 在提升权限后的 PATH 中不可用，Qt 检查改用明确的 `/usr/bin/python3` 启动器。

结果与限制见 [交付报告](../TABLETOP_DUAL_V3_RESULTS.md)。这不是新的训练报告，也不是独立泛化结论。
