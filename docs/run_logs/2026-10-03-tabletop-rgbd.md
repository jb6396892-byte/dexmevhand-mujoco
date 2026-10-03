# 2026-10-03 桌面 RGB-D 前三步执行记录

## 范围

按用户确认，先使用虚拟 RGB-D 与 Adroit，不接实物、不推进抓取控制。现有 `dexmv`、Qt、LoRA 及阶段 3 模型保持不变，新依赖与数据写入共享盘 `visual_grasp`。

## 问题与修复

1. GitHub 截图看的是旧 main。fetch 后 main 相对 Qt 提交无独有提交、落后 42 个，安全快进 `5dea040 -> 1862317`，远端 main 与原工作分支均核对一致。
2. 无 CUDA `nvcc`，FoundationPose 原生编译前提不满足。查官方安装与 Open3D 教程后，以自动检测加几何配准作为已知直立杯基线，不升级旧环境。
3. 首个干扰物 cracker_box 在本机 DexMV 中缺少资产，首次构建失败。改用已有完整视觉和碰撞模型的香蕉、糖盒、芥末瓶、汤罐；保留失败目录，没有无提示下载其他大库。
4. 初始相机使杯柄在背面，不利于朝向判断。开发期改用侧向固定相机，明确支持杯柄可见场景，不对不可见朝向强行输出成功。
5. 默认多重采样导致桌面反投影中位偏差约 0.367 mm；查 MuJoCo 深度采样文档并关闭新场景 MSAA 后降到约 0.000857 mm。没有用杯子坐标或任意平移补偿深度。
6. Hugging Face 权重下载发生一次读超时，断点续传后完成；固定 revision 与 SHA256 已记录。没有关闭证书验证，也没有切换不明权重来源。
7. 发布器首次继承视觉环境的共享盘 TMPDIR，使旧 DexYCB 单元测试无法在 exFAT 创建符号链接。仅将回归测试进程 TMPDIR 改为 `/tmp` 后 194 项通过；模型与实际 RGB-D 数据仍在共享盘，没有修改旧转换逻辑。
8. 三项无杯子输入中，检测器仍输出误检框：两项被多目标歧义拒绝，一项被几何配准拒绝。4/4 负例拒绝是整条感知管线的结果，不是检测器本身零误检，也不代表已验证所有未知物体。

## 验证命令

```bash
cd /home/smgbro/mujoconew/GITHUB
bash scripts/134_setup_tabletop_vision.sh
bash scripts/tabletop_python.sh scripts/135_download_tabletop_detector.py
bash scripts/137_tabletop_gpu.sh scripts/139_view_tabletop.py --verify-frames 120
bash scripts/tabletop_python.sh scripts/140_verify_tabletop_perception.py \
  --mode development --output /media/smgbro/shared/visual_grasp/development/verification-v1
bash scripts/tabletop_python.sh scripts/140_verify_tabletop_perception.py \
  --mode heldout --output /media/smgbro/shared/visual_grasp/heldout-v1
bash scripts/tabletop_python.sh scripts/141_publish_tabletop_evidence.py
```

重复执行必须使用新输出目录，保留旧失败记录；发布器核验冻结源码、全量单元测试、旧语言模型凭据及截图非空，再归档小型材料。原始输入在 `observations`，真值在独立 `evaluation_only`，感知进程结束后才运行误差评测。

## 成果与限制

详细数字由发布器从实际报告生成，见 `docs/presentation/tabletop_rgbd/RESULTS.md`。保留三张图，完整报告/冻结清单/版本及单元测试输出见同目录 `evidence`。场景的沉降测试不是抓取稳定性测试；目前没有动作输出，没有把估计位姿传给旧策略。

下一步是任务书第 4 步：比较真值输入和视觉输入下的控制能力，逐项替换原模型的物体真值观测，解决参考动作随目标位姿调整及感知延迟问题。完成控制验收后，才进入 Qt 新视觉面板和指令触发的整体集成。
