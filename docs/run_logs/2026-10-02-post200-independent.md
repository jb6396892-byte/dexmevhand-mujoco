# 200 次训练后独立评估

## 顺序与产物

1. 冻结旧模型与第 200 次模型、物理门槛、32 个合成新工况和脚本哈希。
2. 执行同工况配对的 64 次闭环物理回放，动作仅经 `env.step` 执行。
3. 第 200 次模型补做两视频标准和半步长复核，共 4 次。
4. 原项目目录归档结果、失败、两张截图和模型索引；共享盘 lora 不保存阶段 3 内容。

运行命令（在项目根目录，使用原 dexmv 环境和记录的 MuJoCo 动态库变量）：

```bash
python scripts/109_evaluate_post200.py evaluate --output data/processed/dual_video_v14c/post200_independent_v1
python scripts/112_finalize_post200.py
```

首次 10/64 完成后按用户澄清调整输出位置，终止未完成回放并迁回项目目录，随后从已落盘结果继续。
未修改模型、工况、物理参数或冻结哈希。原 `long_training/` checkpoint 没有移动或覆盖。

## 实测结论

- 旧策略任务 31/32、严格 17/32；200 次策略任务 30/32、严格 16/32。
- 两者抬杯均 32/32；四次标准/半步长任务和严格检查全部通过。
- 最低任务率通过，但不退化要求失败，`stage3_scoped_acceptance=false`。
- 第二视频 unseen_14 拇指/杯碰撞在 7.538 s 达到 1.068 mm，超过原 1 mm 门槛。
- 未追加低层训练，未替换旧稳定模型；不能宣称阶段 3 完全完成或训练后泛化优势。

详见 [结果、截图、答辩提纲](../presentation/post200/README.md)。
