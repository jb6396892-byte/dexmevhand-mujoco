# 整桌抓取答辩资料索引

日期：2026-10-06。本轮只有框架，没有新增仿真或训练结果。

## 建议 6 页结构

| 页 | 内容 | 当前资料 / 后续补充 |
| --- | --- | --- |
| 1 | 从局部抓取到整桌任务的需求差异 | v6 15×15 cm 与桌面 85×80 cm；不能混淆旧成功率 |
| 2 | 平台 + 局部策略 + 安全监督架构 | [任务书架构图](../../WHOLE_TABLE_TASKBOOK.md#3-方案与控制边界) |
| 3 | 坐标和控制接口 | 三个米制平台通道 + 原 30 维归一化通道；139 维输入适配 |
| 4 | 路径规划和接触控制的方法来源 | [开源与论文对照](../../WHOLE_TABLE_REFERENCES.md) |
| 5 | 开发/留出划分与失败分母 | 160 布局 × 两视频建议协议；未执行 |
| 6 | 结果与局限 | 待 F6 填写；现在只报告框架检查，不写整桌成功率 |

接口代码片段：

```python
@dataclass(frozen=True)
class CompositeCommand:
    platform_joint_target_m: Vec3
    local_normalized_action: Tuple[float, ...]
```

关键思路：旧策略继续管理手部局部接触，平台负责远距离定位；大范围运动引入的惯性和障碍需要单独验证。代码中的 `ScaffoldOnlyError` 明确阻止把未实现的框架当作可运行控制器。

未来只保留必要图：F1 整桌/机构图一张，F3 带载避障轨迹一张，F6 覆盖热图和一个失败案例。当前不复用旧图冒充整桌结果。

核验命令：`python3 scripts/174_inspect_whole_table.py`。实际执行记录见 [F0 日志](../../run_logs/2026-10-06-whole-table-framework.md)。
