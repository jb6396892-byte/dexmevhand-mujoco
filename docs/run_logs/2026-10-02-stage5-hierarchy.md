# 2026-10-02 阶段 5 层次化框架执行日志

范围：用户最新指定 C4 + D1，替代之前“阶段 5 = C3 低层训练”的口头安排。先调用已准入专家参考动作，不启动新的技能训练、语言训练或独立留出评估。正在运行的 200 次任务 DAPG 保持冻结。

## 执行顺序

1. 检查 git 为干净工作区、读取阶段 4 接口，训练当时为 146/200。
2. 检索 BehaviorTree.CPP 官方顺序/重试/超时说明和 SayCan 官方项目，确认不应把物理失败当可无限重试的软件失败。
3. 创建严格技能注册表、JSON Schema、中文精确别名规划器、有限执行器；复用已有物理条件，不改冻结文件。
4. 16 项新单元测试通过；“抓起杯子” dry-run 返回 reach/grasp/lift；固定计划 CLI 实际完成 624 步。
5. 两视频 16 项实际仿真用例通过预定验收，其中普通计划 4 个、故障注入 12 个；保留两张截图。
6. 全量 142 项测试通过；训练 162/200 时发现探索告警，核对到零基 iteration=159 的 1.061 mm 拇指穿透，保留原判定及运行配置。
7. 整理中文操作文档、答辩资料、网页入口与 GitHub 交付；原始数据、模型和动作轨迹不上传。

## 命令

物理命令在 `dexmv` 中运行，使用项目现有 MuJoCo 动态库和 PRIME 环境，不安装新依赖。PyYAML 5.4.1 与 jsonschema 3.2.0 已存在。

```bash
python scripts/16_run_instruction.py "抓起杯子" --dry-run
python scripts/15_run_skill_plan.py --output data/processed/hierarchy_cli_smoke_v1
python scripts/16_run_instruction.py "抓起杯子" --scene second --output data/processed/hierarchy_instruction_cli_v1
python scripts/16_run_instruction.py "停止" --output data/processed/hierarchy_stop_cli_v1
python scripts/100_validate_hierarchy.py
PYTHONPATH=src python -m unittest discover -s tests -q
bash -n scripts/101_view_skill_plan_gpu.sh
python scripts/90_supervise_training.py status
```

产物：`data/processed/stage5_hierarchy_v1/`；CLI 预检独立位于 `data/processed/hierarchy_cli_smoke_v1/`，没有覆盖旧数据。验证脚本会检查阶段 4 的 281 文件冻结清单。语法、Schema 和仿真均通过，旧 MANO 依赖仍有弃用警告，未修改旧环境。

补充核验：第二视频中文入口实际完成 710 步；停止指令不创建 MuJoCo 环境；“不要抓起杯子”在执行前以退出码 2 拒绝。阶段 4 的 281 文件哈希、原训练受保护文件、阶段 5 代码哈希、发布 JSON 和截图哈希全部通过；网页脚本语法、HTML 链接及 `git diff --check` 通过。最新复查为北京时间 17:54 的 173/200，近期探索告警已清除，但历史任务失败仍保留。

## 修改文件

- `src/fromrealhand/hierarchy/{__init__,registry,planner,executor}.py`
- `configs/skill_registry.yaml`、`configs/skill_plan_pickup.json`
- `scripts/15_run_skill_plan.py`、`16_run_instruction.py`、`hierarchy_common.py`
- `scripts/100_validate_hierarchy.py`、`101_view_skill_plan_gpu.sh`
- `tests/test_hierarchy.py`
- `docs/STAGE5_HIERARCHY.md`、`docs/presentation/stage5/`、本日志以及路线图和网页当前状态。

## 未解决及边界

物理滑落后的重抓、任意场景目标、自主动作生成、学习可行性概率和高层语言模型均未实现。当前重规划是边界处根据真实反馈重校验同一目标的剩余前缀，没有假装生成新的恢复轨迹。物理安全失败停止；暂时取不到动作才可有限重试。窗口终态保持不等于持续动力学控制。本轮自动渲染验证是离屏截图，未操作用户原生桌面窗口。

训练仍由原 systemd 用户服务监督。当前框架验证数据和训练采样数据分开统计，不能用框架 16/16 验收宣称学习策略 100% 成功。
