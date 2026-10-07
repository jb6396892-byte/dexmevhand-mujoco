# v2 开发与验收记录

用户要求：先第二视频平稳性，再近桌面交接，最后带杯绕障；允许先做宽松要求版，同步 GitHub 并整理答辩材料。

## 修改

- 新配置 `adroit-navigation-v2.json`，继承 v1，不覆盖 v1 参数。工程加速度上限 0.5 m/s²，旧严格门槛并列报告。
- 导航包络排除纯渲染几何，逐个碰撞几何检查。最后 80 mm 接近单独限速和较小规划余量；第一视频使用 6 mm 悬停。
- 导航到策略增加 1 秒五次力混合。新旧控制力换算后再混合，保持动作仅缩放一次，越出执行器范围则停止。
- 固定障碍在模型初始化时添加，执行期间不改变。手加杯一起规划和检查碰撞；增加过墙时截帧。
- 报告增加加速度峰值时间、双验收条件和失败接触对。旧接口的新增参数均为可选。

## 保留的开发失败

- `first-approach-a`：原初态贴桌，末段仍不能满足 2 mm 几何余量。
- `first-continuous-a`：悬停可达，但硬切换到学习策略出现 >1 mm 穿透，失败。
- `second-wall-a`：初版宽墙对预抓姿态的规划余量不足，拒绝。
- `first-wall-a`：宽墙伸出桌边，占据第一视频前臂预抓空间，拒绝。
- `blocked-wall`：高宽墙使预抓不可用，拒绝；局部动作数为零。这是安全拒绝，不算抓取成功。

所有中间运行保留在共享盘，没有覆盖或删除失败。完整调试清单由脚本 183 收集。

## 最终开发用例

`first-continuous-b`、`second-continuous-final`、`first-wall-final`、`second-wall-final`：工程版 4/4。每条都包括物理连续导航、低速交接、学习抓取、抬杯、搬运和末秒保持。第一视频两条也通过旧严格动态；第二视频两条严格动态失败，不隐藏。

正常墙为 X=0.02 m，尺寸 0.05 × 0.70 × 0.28 m。两视频带杯路径各 4/5 路点，长 0.723/0.671 m；目标误差 1.13/0.23 mm，搬运峰值穿透 0.655/0.462 mm。无环境碰撞，末秒支持率均为 100%。

局部阶段另查：第一视频两例 `reach` 有 44/42 个触桌控制帧，局部子步峰值穿透 0.430/0.434 mm；第二视频无触桌。这沿用旧局部策略的受限接触口径，不是全任务零触桌。新增证据字段单独公开这一点。

## 验证命令

```bash
bash scripts/137_tabletop_gpu.sh -m unittest discover -s tests
bash scripts/137_tabletop_gpu.sh scripts/181_check_free_hand_local.py --config configs/adroit-navigation-v2.json --video first --shift .2 0 --transit --carry-goal -.2 .1 .20 --wall --output <new-directory>
bash scripts/137_tabletop_gpu.sh scripts/181_check_free_hand_local.py --config configs/adroit-navigation-v2.json --video second --shift -.2 0 --transit --carry-goal .2 .1 .20 --wall --output <new-directory>
bash scripts/137_tabletop_gpu.sh scripts/183_publish_navigation_v2.py
```

274 项软件回归通过。旧 v6 第二视频 seed=30、四干扰物完整 reach/grasp/lift/transport 回归通过，穿透峰值 0.718 mm。没有新增依赖、训练或修改共享盘语言环境。

## 交付边界

两张真实过墙图和一张带精确峰值标记的加速度图；方法、代码片段、引用和结果 JSON 进入 Git。全量轨迹仍在 `visual_grasp/adroit-navigation-v2/`。新成果是已知几何、固定杯型的四个开发任务，不是全桌随机统计验收，也不是新的 Qt 模式。
