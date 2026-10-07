# 随机桌面连续导航与抓取测试

## 范围和实现

用户接受从障碍上方通过，要求把抓取接入并多轮测试、保留答辩资料、同步 GitHub。本轮不训练、不改 Qt 默认模式、不重置执行中的杯子。

- `RandomTask.run(scene_layout=...)` 接受独立冻结的整桌布局，并复制输入，避免局部坐标适配污染 manifest。
- `translate_initial_scene(..., translate_object=False)` 只移动参考坐标和手基座，保持已经放在目标位置的杯子，避免重复位移。
- `181` 支持冻结布局、种子、原生窗口；`185` 批量执行并保留失败分母；`186` 导出精简报告和三张图；`187/188` 根据原 manifest 在窗口重新执行案例。
- 带杯失败增加起终点冲突物体名称与执行净空诊断。
- v3 实验增加 10 mm 规划余量下的短程退让，执行净空仍为 8 mm；没有在本批中获准进入运动，未验证收益，不改默认 v2。

## 结果

4301–4304，各第一/第二视频，两轮均 5/8。结果完全复现，属于相同开发工况对照，不作为独立泛化样本翻倍。第一轮与第二轮各：

- 导航和低速接近 8/8。
- reach/grasp/lift 7/8。
- 完整搬运与末秒保持 5/8。
- 完整严格动态验收 2/8，其余成功任务采用原工程口径。
- 成功任务杯子目标误差 0.379–0.661 mm，末秒支持率 100%。

三个失败：4301-first 糖盒侵入搬运起点包络的 8 mm 执行安全区；4303-first 芥末瓶侵入目标包络；4303-second 在 reach 的第 411 次动作中食指末节接触芥末瓶，接触深度约 0.140 mm 后子步停止。没有删除这些布局或移动障碍再冒充同一工况。

## 命令

```bash
bash scripts/137_tabletop_gpu.sh scripts/185_test_navigation_grasp.py --output /media/smgbro/shared/visual_grasp/navigation-grasp-v1
bash scripts/137_tabletop_gpu.sh scripts/185_test_navigation_grasp.py --config configs/adroit-navigation-v3.json --output /media/smgbro/shared/visual_grasp/navigation-grasp-v2
bash scripts/137_tabletop_gpu.sh -m unittest discover -s tests
bash scripts/137_tabletop_gpu.sh scripts/186_publish_navigation_grasp.py --runs /media/smgbro/shared/visual_grasp/navigation-grasp-v1 /media/smgbro/shared/visual_grasp/navigation-grasp-v2
bash scripts/188_launch_navigation_grasp.sh --case /media/smgbro/shared/visual_grasp/navigation-grasp-v1/seed-4304-first --close-after-run
```

最终软件回归 277 项通过；新增测试覆盖已有杯位不重复平移、旧初始化平移兼容性、规划余量与执行安全区的区别。已有 chumpy/NumPy 废弃警告不影响结果。GLFW 原生入口的初始化先后顺序警告已通过显式初始化处理。

原生窗口全程补充验证：4304-first，3426 帧，任务和严格动态通过；杯子误差 0.461 mm，搬运穿透峰值 0.645 mm，末秒支持率 100%，搬运环境接触 0，无 GL 错误。结果位于 `/tmp/adroit-navigation-grasp-10e6iqb6/run`，精简记录随报告归档。窗口物理重跑的数值不要求与无窗口运行逐位相同，也不追加为新独立统计样本。

## 证据和未解决事项

[精简结果、截图、轨迹图和答辩提纲](../presentation/navigation_grasp/README.md)。模型 SHA256、每轮源码 SHA256、冻结场景均已归档；原始轨迹放共享盘，不上传模型或数据集。

下一步需要抓取段扫掠空间检查、抓后可退出空间判断，以及明确的终态包络检查/抓姿选择。当前只是把原有局部模型接到导航系统，不代表它已经学会避开任意桌面障碍，也不声称 >80% 泛化通过。
