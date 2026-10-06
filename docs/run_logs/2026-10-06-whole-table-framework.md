# F0：整桌 XYZ 平台框架

## 本轮授权范围

用户选择移动平台，要求先搭建框架、写任务书和搜索开源资料，不急于实施。已确认任务只到搬运并保持，放下另做。

## 已做

- 阅读现有场景、学习控制、139 维特征构造和冻结评估入口，识别固定前 30 关节的兼容风险。
- 新增独立 `fromrealhand.whole_table` 包、草案配置和只读检查入口；未改旧物理、Qt 或模型加载路径。
- 写明全桌采样、碰撞检查、平台/局部控制分工、坐标变换、物理门槛、失败分母和阶段验收。
- 在线查阅 MuJoCo、Menagerie、OMPL、Ruckig、MimicGen、DexTrack、DexMachina 的官方/作者资料；链接与取舍收录参考表。
- 未生成物理截图、没有下载大数据、没有运行仿真或训练。答辩资料为架构图和后续取证清单。

## 检查命令

```bash
python3 scripts/174_inspect_whole_table.py
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_whole_table_scaffold.py'
bash scripts/137_tabletop_gpu.sh -m unittest discover -s tests
git diff --check
```

结果：检查入口返回 `configuration_valid=true`、`runtime_ready=false`、`physics_executed=false`、`training_started=false`；新增 7/7 单元测试通过，完整 259/259 测试通过，`git diff --check` 通过。

完整回归运行在旧 Python 3.7 环境中，未运行整桌物理回放；原 MANO/chumpy 测试仍有已有的弃用和资源警告，没有测试失败。配置与接口也可在系统 Python 下独立检查，不依赖共享盘模型可读。

本轮新增配置、四个包文件、检查脚本、测试文件和四份文档，README 增加入口。旧 v6 控制代码和默认 Qt 启动器保持不变。输出只说明框架质量，不能替代 F1–F6 验收。
