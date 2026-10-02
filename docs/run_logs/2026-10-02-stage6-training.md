# 阶段 6 隔离实训与验收记录

## 本轮操作

1. 确认 shared 宿主机可写，不修盘、不重挂载。按用户澄清，lora 仅存语言模型相关内容。
2. `110_setup_language_study.sh` 安装隔离包到共享盘；Linux 仅保留 exFAT 所需的 65 MB 引导环境。
3. 官方 Qwen 模型固定 revision 后下载，核对 SHA256；CUDA BF16、模板和 `pip check` 通过。
4. `111_language_study.py` 完成基础模型验证和首次 3 epoch 小试：34/42、1 条越界误接受，门槛失败，未启动该版正式训练。
5. `115_language_contrast_study.py` 保留原数据并额外冻结 296 条对比训练指令，不改系统提示、超参数或门槛。
6. 新版小试 42/42、误接受 0，通过后从原基础模型独立开始 8 epoch 正式 SFT。
7. 正式 800 步、524.283 秒结束，按验证 loss 选第 500 步权重；正式验证 42/42。
8. 首次打开冻结留出：基础模型 4/56，LoRA 52/56；LoRA 有 1 个越界误接受和 1 个未注册目标，部署未通过。
9. 只将语义正确计划送入物理接口：去重后 8 个动作、2 个停止全部通过，三个拒绝门控检查通过。
10. 实测 `116_run_stage6_model.sh ... --execute` 在创建仿真前返回拒绝；不带 execute 的生成检查能输出 lift 前置完整计划。

## 关键命令

```bash
export STUDY_ROOT=/media/smgbro/shared/lora
bash scripts/stage6_study_python.sh scripts/115_language_contrast_study.py pilot
bash scripts/stage6_study_python.sh scripts/115_language_contrast_study.py formal
bash scripts/stage6_study_python.sh scripts/115_language_contrast_study.py heldout
# 原 dexmv + MuJoCo 库路径下运行：
python scripts/107_verify_language_pipeline.py \
  --model-evaluation /media/smgbro/shared/lora/language/study_v3/lora-heldout \
  --output /media/smgbro/shared/lora/language/study_v3/physical-acceptance
```

107 退出码为 1：因为语言误接受门槛未过，非运行崩溃。
训练中 GPU 温度采样约 73°C，未出现 NaN、CUDA OOM 或驱动错误。原 torch 1.13.1+cu117 不变。

## 验证与未解决项

最终 163 项回归测试、冻结原模型/源码校验及两张阶段 3 图像重放检查通过。
早先直接在沙箱运行测试失败于旧 MuJoCo 编译锁只读；宿主机运行补齐 `PYTHONPATH=src:scripts` 后通过。
没有因此修改旧 MuJoCo 环境或放宽测试。

语言自动执行仍锁定。后续需新的开发数据和新的独立留出，不在已看过测试上继续挑 checkpoint。
阶段 3 同期评估也未通过不退化门槛，旧稳定策略保留；两问题分别记录，不互相宣称完成。
源码、资料和小型证据同步 GitHub；大依赖、基础模型、LoRA 权重只留共享盘。

详见 [模型结果与答辩提纲](../presentation/stage6/STUDY_RESULTS.md) 和 [完整命令](../STAGE6_STUDY.md)。
