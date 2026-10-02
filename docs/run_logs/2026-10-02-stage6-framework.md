# 2026-10-02：阶段 6 框架与 200 次 DAPG 收尾

## 本轮范围与变更

用户要求 D2 语言规划，在存储确认时选择先只搭框架。本轮新增 `src/fromrealhand/language_planner/`、四个配置文件、`requirements-stage6.txt`、102 至 108 脚本、运行包装脚本和测试。原阶段 4/5 冻结代码、原数据、示范、已有 checkpoint 未修改。网站与路线图将阶段 6 明确为 D2，未完成的 C3 单独保留。

存储排查：沙箱内 `findmnt` 显示 shared 只读；沙箱外相同只读查询显示 exFAT `rw`。前者是权限视图限制，不是硬盘错误证据。未 remount、fsck，未安装包、下载模型或执行 LoRA；后续启动仍需确认。

## 执行与结果

| 命令 / 检查 | 结果 |
|---|---|
| `scripts/90_supervise_training.py status` | 184 → 190 → 199 → 200；最后 completed，退出码 0 |
| `scripts/102_build_language_data.py` | 504/42/56 行及来源哈希生成，拒绝覆盖 |
| `scripts/104_train_language_lora.py check` | 冻结输入通过；模型不存在，训练未启动 |
| `bash scripts/103_setup_language_env.sh --check-only` | 沙箱内拒绝只读视图，无写入；随后沙箱外核实 shared 实际 rw |
| 15 项新测试 | 全部通过 |
| 初次普通沙箱全量测试 | 因旧 mujoco_py 编译锁在只读环境目录失败，不是新增功能回归 |
| 正确权限与原 MuJoCo 环境下全量重测 | 157/157；旧 chumpy/NumPy 弃用提示保留，无升级依赖 |
| `scripts/107_verify_language_pipeline.py --output data/processed/stage6_language_v1/framework --capture` | 8 动作 + 2 停止通过，1 张截图；标准答案夹具，非模型预测 |
| `scripts/108_publish_stage6_framework.py` | 重跑测试、核验冻结数据和 DAPG 受保护输入，发布小型答辩证据 |

物理检查沿用：场景穿透 <=1 mm、仅初态恢复一次、执行期间不写物体状态、专家参考 qpos/qvel 最大误差 <=1e-7，运行时继续检查支撑和安全契约。3 类新拒绝情况均未进入物理执行。

## 训练收尾

DAPG 14:42:17 启动，18:23:19 完成，约 3 小时 41 分。200 次非零更新，573,400 仿真步；400 条训练轨迹任务通过 395、旧严格通过 364，最大实测 KL 0.000636。此统计来自持续变化的训练策略，不是固定模型测试。

训练后两标准场景任务与旧严格门槛均通过，目标误差 1.579 / 3.722 mm，最大场景穿透 0.585 / 0.889 mm。第二视频闭合与抬杯阶段无名指仍未持续接触；旧严格末段指标不代表视频全程四指复现。新 checkpoint 独立留出未执行，原稳定策略未自动替换。

## 未完成与下一步

1. 取得继续安装/训练确认，使用共享盘安装隔离的现代 Python 包与模型；本轮提供的版本组合仅源码/API 核对，尚未运行验证。
2. 用真实 tokenizer 检查模板边界、长度及显存，再做 LoRA 小试；仅验证集选择模型。
3. 冻结 LoRA，与基础模型做原始生成对照；不允许规则回退掩盖模型错误。
4. 模型正确计划经新的物理验收后才生成可部署收据。没有模型时，`model_acceptance_passed` 明确为 false。
5. 200 次 DAPG 的新独立留出评估与 C3 学习技能后端另行安排，不声称本轮已完成。

官方方法参考和完整命令见 [阶段 6 操作说明](../STAGE6_LANGUAGE.md)，一张图与答辩提纲见 [阶段 6 材料](../presentation/stage6/README.md)。
