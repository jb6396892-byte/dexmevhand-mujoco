"use strict";

const stages = [
  {
    id: "A0",
    phase: "A",
    phaseLabel: "基础框架",
    title: "跑通 relocate-mug",
    objective: "确认原始 DexMV 训练链路和策略可视化在本机可用。",
    tasks: [
      "完成 relocate-mug DAPG 训练并保存策略",
      "加载 best_policy.pickle 完成 MuJoCo 可视化",
      "确认 mug 的 object_scale 使用已验证值 0.8",
    ],
    outputs: [
      "可加载的 best_policy.pickle",
      "完整训练日志和迭代结果",
      "已验证的 relocate-mug 运行环境",
    ],
    acceptance: [
      "训练可以正常采样和更新参数",
      "策略文件能够加载且仿真窗口正常显示",
      "mug 模型的位置和尺寸符合预期",
    ],
    commands: [
      { label: "检查项目环境", text: "bash scripts/00_check_env.sh" },
    ],
    defaultDone: true,
  },
  {
    id: "A1",
    phase: "A",
    phaseLabel: "基础框架",
    title: "建立真实视频项目框架",
    objective: "建立视频、位姿、retarget、demonstration、训练和验证的基础目录与脚本。",
    tasks: [
      "建立 data、configs、scripts 和 src/fromrealhand 目录",
      "封装 retarget、demo 生成、demo 检查和训练入口",
      "补充环境检查、路径配置和中文项目文档",
      "将项目独立保存到 GitHub 仓库",
    ],
    outputs: [
      "scripts/00_check_env.sh 至 scripts/08_visualize_policy.sh",
      "src/fromrealhand 数据处理基础模块",
      "configs/dapg-mug-real.yaml",
    ],
    acceptance: [
      "环境检查脚本可以定位 DexMV 和 MuJoCo",
      "项目不复制 dexmv-sim 和 dexmv-learn",
      "大数据、模型、日志和 checkpoint 已被 Git 忽略",
    ],
    commands: [
      { label: "环境检查", text: "bash scripts/00_check_env.sh" },
    ],
    defaultDone: true,
  },
  {
    id: "B1",
    phase: "B",
    phaseLabel: "真实视频",
    title: "准备第一条 DexYCB 真实轨迹",
    objective: "取得一条右手抓取 025_mug 的真实 RGB-D 序列，并保留完整标定和位姿标签。",
    tasks: [
      "确认至少有 30 GB 可用磁盘空间，并通过环境检查",
      "下载一个 DexYCB subject、calibration 和 models",
      "筛选目标为 025_mug、mano_sides 为 right 的序列",
      "选择无遮挡、画面稳定的固定相机视角",
      "建立 seq_dexycb_001/meta.json 并记录来源、相机、帧率和物体 ID",
      "确认 RGB、depth、相机标定、手部标签和 mug 6D 位姿都可读取",
    ],
    outputs: [
      "data/external/dexycb/",
      "data/real_data/relocate_mug/seq_dexycb_001/meta.json",
      "数据版本、下载日期和文件校验值记录",
    ],
    acceptance: [
      "RGB 和 depth 帧数一致",
      "相机内参和外参可以读取",
      "MANO/3D 手关节和 025_mug 6D 位姿可以读取",
      "连续播放包含接近、抓住和抬起完整动作",
    ],
    commands: [
      { label: "检查磁盘", text: "df -h /home/smgbro" },
      { label: "检查环境", text: "bash scripts/00_check_env.sh" },
    ],
  },
  {
    id: "B2",
    phase: "B",
    phaseLabel: "真实视频",
    title: "扫描并转换 DexYCB 数据",
    objective: "把 DexYCB 原始格式转换为本项目约定的逐帧位姿、标定和图像目录。",
    tasks: [
      "实现 src/fromrealhand/dexycb_io.py 统一读取数据",
      "实现 09_scan_dexycb.py 并输出 JSON 或 CSV 序列清单",
      "实现 10_convert_dexycb.py 转换 RGB、depth、标定和位姿",
      "保存 source frame 到输出 frame 的编号映射",
      "保存 valid_frames.npy，明确标记低置信度和无效帧",
      "统一把所有平移转换成米，并在 meta.json 声明坐标系",
    ],
    outputs: [
      "scripts/09_scan_dexycb.py",
      "scripts/10_convert_dexycb.py",
      "scripts/11_visualize_source_pose.py",
      "符合 DATA_FORMAT.md 的 seq_dexycb_001 目录",
    ],
    acceptance: [
      "RGB、depth、hand pose 和 object pose 数量一致",
      "所有逐帧编号一一对应",
      "所有平移使用米",
      "meta.json 明确记录每种位姿的坐标系",
    ],
    commands: [
      {
        label: "扫描 mug 序列",
        text: "python scripts/09_scan_dexycb.py \\\n  --root data/external/dexycb \\\n  --object 025_mug \\\n  --hand-side right \\\n  --output data/processed/dexycb_mug_sequences.json",
      },
      {
        label: "转换选定序列",
        text: "python scripts/10_convert_dexycb.py \\\n  --root data/external/dexycb \\\n  --sequence SUBJECT/SEQUENCE \\\n  --camera CAMERA_SERIAL \\\n  --output data/real_data/relocate_mug/seq_dexycb_001",
      },
    ],
  },
  {
    id: "B3",
    phase: "B",
    phaseLabel: "真实视频",
    title: "验证坐标、尺度和位姿",
    objective: "在进入 retarget 前，通过 2D 重投影和 3D 可视化证明几何转换正确。",
    tasks: [
      "把 3D 手关节投影回 RGB 图像",
      "按 mug 6D 位姿把 YCB mesh 渲染到原图",
      "同时显示相机轴、世界轴、桌面、手骨架和 mug mesh",
      "绘制手腕与 mug 的平移轨迹并检查速度尖峰",
      "抽查开头、中间和结尾至少 10 帧",
    ],
    outputs: [
      "data/processed/seq_dexycb_001/overlay/",
      "data/processed/seq_dexycb_001/trajectory_report.json",
      "有效帧比例、最大位移跳变和运动范围统计",
    ],
    acceptance: [
      "手关节投影与图像中的真实手基本重合",
      "mug mesh 的位置、方向和尺度与画面一致",
      "桌面法向正确，mug 起点位于桌面上方",
      "无镜像、轴交换、单位混用或矩阵方向错误",
    ],
    commands: [
      {
        label: "可视化源位姿",
        text: "python scripts/11_visualize_source_pose.py \\\n  --sequence-dir data/real_data/relocate_mug/seq_dexycb_001 \\\n  --output data/processed/seq_dexycb_001",
      },
    ],
  },
  {
    id: "B4",
    phase: "B",
    phaseLabel: "真实视频",
    title: "生成并回放第一条 demonstration",
    objective: "把真实手部动作 retarget 到 Adroit，并生成 DexMV 可训练的 demonstration。",
    tasks: [
      "运行 03_retarget_one.py 并检查帧数、关节范围和 NaN",
      "可视化 Adroit 手与 mug 的空间和时序对齐",
      "运行 05_generate_demo.py 生成 relocate-mug-real.pkl",
      "运行 06_validate_demo.py 检查字段、长度和 action 分布",
      "在 MuJoCo 中回放并检查跳动、穿模和抓取时机",
    ],
    outputs: [
      "seq_dexycb_001/retargeting.pkl",
      "data/demonstrations/relocate-mug-real.pkl",
      "第一条可回放的真实视频 demonstration",
    ],
    acceptance: [
      "retargeting 与有效视频帧数一致",
      "手杯整体位置和手指闭合时机合理",
      "demo 包含 observations、actions、rewards、sim_data、model_data",
      "action 不会大量持续饱和到 -1 或 1",
      "杯子在抓取前不会跳动或穿过手掌",
    ],
    commands: [
      {
        label: "Retarget",
        text: "/home/smgbro/miniconda3/bin/conda run -n dexmv \\\n  python scripts/03_retarget_one.py \\\n  --hand-dir data/real_data/relocate_mug/seq_dexycb_001/hand_pose \\\n  --output data/real_data/relocate_mug/seq_dexycb_001/retargeting.pkl",
      },
      {
        label: "检查手杯对齐",
        text: "/home/smgbro/miniconda3/bin/conda run -n dexmv \\\n  python scripts/04_visualize_retargeting.py \\\n  --retargeting data/real_data/relocate_mug/seq_dexycb_001/retargeting.pkl \\\n  --object-dir data/real_data/relocate_mug/seq_dexycb_001/object_pose \\\n  --camera-to-world data/real_data/relocate_mug/seq_dexycb_001/calib/camera_to_world.npy",
      },
      {
        label: "生成 demonstration",
        text: "/home/smgbro/miniconda3/bin/conda run -n dexmv \\\n  python scripts/05_generate_demo.py \\\n  --sequence-dir data/real_data/relocate_mug/seq_dexycb_001 \\\n  --output data/demonstrations/relocate-mug-real.pkl \\\n  --trajectory-id seq_dexycb_001",
      },
      {
        label: "验证 demonstration",
        text: "/home/smgbro/miniconda3/bin/conda run -n dexmv \\\n  python scripts/06_validate_demo.py data/demonstrations/relocate-mug-real.pkl",
      },
    ],
  },
  {
    id: "B5",
    phase: "B",
    phaseLabel: "真实视频",
    title: "使用真实 demonstration 短训练",
    objective: "用 20 次迭代验证真实 demo 能完成行为克隆、采样和策略更新。",
    tasks: [
      "建立 configs/dapg-mug-real-smoke.yaml",
      "设置 NUM_ITER=20、较小 NUM_CPU 和独立 JOB_DIR",
      "保持 BC_INIT=true 和 USE_DAPG=true",
      "训练前重新验证 demonstration",
      "完成 smoke training 并加载策略可视化",
      "通过后把正式训练 NUM_ITER 恢复为 2000",
    ],
    outputs: [
      "training_log/relocate-mug-real-smoke/",
      "docs/run_logs/<日期>-real-demo-smoke.md",
      "可加载的 smoke best_policy.pickle",
    ],
    acceptance: [
      "20 次迭代无 NaN 和 shape 错误",
      "policy、日志和迭代结果正常保存",
      "reward 不会立即异常或全零",
      "训练后的策略能够加载并可视化",
    ],
    commands: [
      {
        label: "再次验证 demo",
        text: "/home/smgbro/miniconda3/bin/conda run -n dexmv \\\n  python scripts/06_validate_demo.py data/demonstrations/relocate-mug-real.pkl",
      },
      { label: "运行短训练", text: "bash scripts/07_train_dapg.sh configs/dapg-mug-real-smoke.yaml" },
      { label: "加载训练策略", text: "bash scripts/08_visualize_policy.sh /path/to/best_policy.pickle" },
    ],
  },
  {
    id: "C1",
    phase: "C",
    phaseLabel: "低层技能",
    title: "扩展到多条真实抓杯轨迹",
    objective: "增加受试者、视角和初始位姿变化，避免策略记住单条轨迹。",
    tasks: [
      "第一轮加入至少 5 条人工检查通过的右手 mug 轨迹",
      "逐条生成并验证 demo，再使用 --append 合并",
      "记录保留、剔除轨迹及剔除原因",
      "扩展不同受试者、视角和 mug 初始位姿",
      "按 subject 或原始 sequence 划分训练集和测试集",
    ],
    outputs: [
      "多轨迹 relocate-mug-real.pkl",
      "轨迹质量与剔除原因清单",
      "按来源隔离的训练、验证和测试划分",
    ],
    acceptance: [
      "所有轨迹字段和维度一致",
      "轨迹长度分布合理且无异常短轨迹",
      "训练和测试不存在相邻帧或同序列泄漏",
    ],
    commands: [
      {
        label: "追加一条轨迹",
        text: "python scripts/05_generate_demo.py \\\n  --sequence-dir data/real_data/relocate_mug/SEQ_ID \\\n  --output data/demonstrations/relocate-mug-real.pkl \\\n  --trajectory-id SEQ_ID \\\n  --append",
      },
    ],
  },
  {
    id: "C2",
    phase: "C",
    phaseLabel: "低层技能",
    title: "拆分可复用低层技能",
    objective: "把完整抓杯动作拆成 reach、grasp、lift 和 transport 四个可独立训练的片段。",
    tasks: [
      "实现基于距离、接触和杯子高度的自动边界规则",
      "实现 12_segment_skills.py 生成初始切分",
      "实现 13_review_skill_segments.py 人工审核边界",
      "实现 14_export_skill_demos.py 导出独立技能 demo",
      "为所有审核后的片段写入 reviewed: true",
    ],
    outputs: [
      "annotations/skill_segments.json",
      "reach、grasp、lift、transport 四类 demo pkl",
      "src/fromrealhand/skill_segmentation.py",
      "configs/skills/*.yaml",
    ],
    acceptance: [
      "每个技能片段都可以独立回放",
      "技能边界无缺帧或重复执行造成的明显跳变",
      "所有自动边界均经过人工审核",
    ],
    commands: [
      {
        label: "切分技能",
        text: "python scripts/12_segment_skills.py \\\n  --demo data/demonstrations/relocate-mug-real.pkl \\\n  --output annotations/skill_segments.json",
      },
      {
        label: "导出技能 demo",
        text: "python scripts/14_export_skill_demos.py \\\n  --segments annotations/skill_segments.json \\\n  --output-dir data/demonstrations",
      },
    ],
  },
  {
    id: "C3",
    phase: "C",
    phaseLabel: "低层技能",
    title: "训练低层技能策略",
    objective: "分别训练四个能够单独调用、具备明确成功条件的技能策略。",
    tasks: [
      "为每个技能建立独立 MuJoCo task 和配置",
      "先用行为克隆复现平均轨迹",
      "使用 DAPG 分别微调四个策略",
      "随机化手、mug 和目标的初始位置与方向",
      "为每个技能实现成功条件和失败分类",
      "保存最佳策略、训练日志和独立评估报告",
    ],
    outputs: [
      "reach、grasp、lift、transport 最佳策略",
      "每个技能的训练配置和日志",
      "独立成功率与失败类型报告",
    ],
    acceptance: [
      "reach 到达预抓取区域且不碰倒杯子",
      "grasp 建立稳定抓取且短时间内不滑落",
      "lift 达到目标高度且不过度旋转",
      "transport 到达目标位姿范围且不掉落",
    ],
    commands: [
      { label: "训练技能配置", text: "bash scripts/07_train_dapg.sh configs/skills/SKILL.yaml" },
    ],
  },
  {
    id: "C4",
    phase: "C",
    phaseLabel: "低层技能",
    title: "实现技能注册表和执行器",
    objective: "按固定计划调用多个策略，并在每一步根据状态决定继续、重试或停止。",
    tasks: [
      "实现 skill_registry.py 加载技能、参数和策略路径",
      "实现 conditions.py 统一前置条件、成功条件和终止原因",
      "实现 affordance.py 估计当前状态下的技能成功概率",
      "实现 executor.py 的超时、重试和安全停止",
      "实现 15_run_skill_plan.py 执行固定技能序列",
      "记录技能输入、成功概率、终止原因、耗时和重试次数",
    ],
    outputs: [
      "src/fromrealhand/hierarchy/skill_registry.py",
      "src/fromrealhand/hierarchy/executor.py",
      "src/fromrealhand/hierarchy/conditions.py",
      "src/fromrealhand/hierarchy/affordance.py",
    ],
    acceptance: [
      "reach -> grasp -> lift -> transport 可以完成",
      "任一技能失败后不会盲目执行后续技能",
      "支持超时、重试和安全停止",
    ],
    commands: [
      {
        label: "执行固定技能计划",
        text: "python scripts/15_run_skill_plan.py \\\n  --plan reach,grasp,lift,transport \\\n  --object mug",
      },
    ],
  },
  {
    id: "D1",
    phase: "D",
    phaseLabel: "高层规划",
    title: "实现规则高层规划器",
    objective: "先用确定性规则验证指令、计划 Schema、技能注册表和执行器的完整接口。",
    tasks: [
      "定义严格的技能计划 JSON Schema",
      "实现 planner.py 将常见指令映射为技能序列",
      "拒绝未注册技能、错误物体和越界参数",
      "每个技能执行后根据新状态继续、重试或重新规划",
      "实现 16_run_instruction.py 端到端执行中文指令",
    ],
    outputs: [
      "src/fromrealhand/hierarchy/planner.py",
      "src/fromrealhand/hierarchy/plan_schema.py",
      "configs/skill_registry.yaml",
      "scripts/16_run_instruction.py",
    ],
    acceptance: [
      "所有计划通过 JSON Schema 检查",
      "非法技能、物体和参数被明确拒绝",
      "抓起杯子可稳定映射为 reach -> grasp -> lift",
      "执行器能够依据新状态重试或重新规划",
    ],
    commands: [
      {
        label: "执行中文指令",
        text: "python scripts/16_run_instruction.py \\\n  --instruction \"抓起杯子\" \\\n  --planner rule",
      },
    ],
  },
  {
    id: "D2",
    phase: "D",
    phaseLabel: "高层规划",
    title: "训练语言模型生成技能计划",
    objective: "让多种自然语言表达稳定输出符合 Schema 且可由低层执行的技能计划。",
    tasks: [
      "从规则规划器与成功日志生成指令、场景和计划样本",
      "定义训练集、验证集和指令改写测试集",
      "选择预训练小型语言模型并进行 LoRA/SFT",
      "把可用技能和参数约束放入模型输入",
      "只接受通过 Schema 和技能注册表校验的 JSON 输出",
      "结合语言分数与 affordance 成功概率选择技能",
    ],
    outputs: [
      "高层规划训练数据集和版本说明",
      "LoRA 适配器及训练配置",
      "计划合法率、顺序正确率和泛化评估报告",
    ],
    acceptance: [
      "JSON 合法率达到预先设定目标",
      "不生成技能库以外的动作",
      "mug 未抓住时不会直接调用 lift 或 tilt",
      "同义指令得到一致技能序列",
    ],
    commands: [
      {
        label: "语言规划评估",
        text: "python scripts/16_run_instruction.py \\\n  --instruction \"把杯子拿起来\" \\\n  --planner language \\\n  --dry-run",
      },
    ],
  },
  {
    id: "E1",
    phase: "E",
    phaseLabel: "倒水任务",
    title: "增加倒水视频和新技能",
    objective: "补充 DexYCB 缺少的倾斜、恢复竖直、放下和松手真实动作。",
    tasks: [
      "固定相机并标定内参和 camera_to_world",
      "选用尺寸接近 MuJoCo 模型的 mug，并记录尺寸",
      "为第一版 mug 和目标容器布置 ArUco 标记",
      "录制抓取、抬起、移动、倾斜、恢复、放下和松手完整视频",
      "记录目标容器的位置、尺寸和任务目标",
      "复用位姿恢复和 retarget 管线生成新增技能 demo",
    ],
    outputs: [
      "包含完整倒水动作的标定视频数据",
      "tilt、upright、place、release 四类技能 demo",
      "杯子和目标容器标定记录",
    ],
    acceptance: [
      "新视频可以通过现有位姿恢复与 retarget 管线",
      "tilt、upright、place、release 均可独立训练和回放",
      "杯子和目标容器位姿在同一世界坐标系中",
    ],
    commands: [
      { label: "准备视频帧", text: "python scripts/01_extract_frames.py --help" },
      { label: "准备位姿目录", text: "python scripts/02_prepare_pose_dirs.py --help" },
    ],
  },
  {
    id: "E2",
    phase: "E",
    phaseLabel: "倒水任务",
    title: "建立 pour-mug 环境并完整评估",
    objective: "让自然语言指令驱动完整的抓杯、倒水、恢复和放回任务。",
    tasks: [
      "建立 pour-mug MuJoCo 环境、目标容器和任务状态",
      "实现杯口位置、倾角、保持时间和掉落检测成功条件",
      "暂不模拟液体，先完成稳定的几何倒水任务",
      "组合全部低层技能并接入高层语言规划器",
      "评估感知、retarget、技能、规划、完整任务和泛化",
      "记录平均重试次数、恢复率和失败类型",
    ],
    outputs: [
      "可训练和可视化的 pour-mug 环境",
      "抓杯 -> 倒水 -> 放回的完整策略与计划",
      "可重复运行的配置、模型和最终评估报告",
    ],
    acceptance: [
      "指定数据和配置可以重复训练",
      "自然语言可以生成可解释且合法的技能计划",
      "每个技能结果和失败原因均可追踪",
      "完整任务可以在 MuJoCo 中稳定运行",
    ],
    commands: [
      {
        label: "执行倒水指令",
        text: "python scripts/16_run_instruction.py \\\n  --instruction \"拿起杯子，把水倒进目标容器，再把杯子放回桌面\" \\\n  --planner language",
      },
    ],
  },
];

const storageKey = "fromrealhand-dashboard-progress-v1";
const stageList = document.querySelector("#stageList");
const emptyState = document.querySelector("#emptyState");
const searchInput = document.querySelector("#taskSearch");
const phaseButtons = [...document.querySelectorAll(".phase-button")];
const statusButtons = [...document.querySelectorAll(".segmented-control button")];

let activePhase = "all";
let activeStatus = "all";
let progress = loadProgress();
let openStages = new Set();

function defaultProgress() {
  return Object.fromEntries(
    stages.flatMap((stage) =>
      stage.tasks.map((_, index) => [`${stage.id}-${index}`, Boolean(stage.defaultDone)]),
    ),
  );
}

function loadProgress() {
  const defaults = defaultProgress();

  try {
    const saved = JSON.parse(localStorage.getItem(storageKey));
    if (!saved || typeof saved !== "object") return defaults;

    return Object.fromEntries(
      Object.entries(defaults).map(([key, defaultValue]) => [
        key,
        typeof saved[key] === "boolean" ? saved[key] : defaultValue,
      ]),
    );
  } catch {
    return defaults;
  }
}

function saveProgress() {
  try {
    localStorage.setItem(storageKey, JSON.stringify(progress));
  } catch {
    // The dashboard remains usable when storage is unavailable.
  }
}

function completedTaskCount(stage) {
  return stage.tasks.filter((_, index) => progress[`${stage.id}-${index}`]).length;
}

function getCurrentStage() {
  return stages.find((stage) => completedTaskCount(stage) < stage.tasks.length) ?? null;
}

function getStageStatus(stage, currentStage) {
  if (completedTaskCount(stage) === stage.tasks.length) return "done";
  if (currentStage?.id === stage.id) return "current";
  return "pending";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function stageSearchText(stage) {
  return [
    stage.id,
    stage.phaseLabel,
    stage.title,
    stage.objective,
    ...stage.tasks,
    ...stage.outputs,
    ...stage.acceptance,
    ...stage.commands.flatMap((command) => [command.label, command.text]),
  ]
    .join(" ")
    .toLocaleLowerCase("zh-CN");
}

function stageCard(stage, currentStage) {
  const doneTasks = completedTaskCount(stage);
  const status = getStageStatus(stage, currentStage);
  const statusLabels = { done: "已完成", current: "当前", pending: "待处理" };
  const isOpen = openStages.has(stage.id);
  const bodyId = `stage-body-${stage.id}`;
  const taskRows = stage.tasks
    .map((task, index) => {
      const taskId = `${stage.id}-${index}`;
      const checked = Boolean(progress[taskId]);
      return `
        <label class="task-row${checked ? " is-checked" : ""}">
          <input type="checkbox" data-task-id="${taskId}" ${checked ? "checked" : ""}>
          <span>${escapeHtml(task)}</span>
        </label>`;
    })
    .join("");
  const outputs = stage.outputs.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  const acceptance = stage.acceptance.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  const commands = stage.commands
    .map(
      (command, index) => `
        <div class="command-block">
          <div class="command-header">
            <span>${escapeHtml(command.label)}</span>
            <button class="copy-button" type="button" data-command-stage="${stage.id}" data-command-index="${index}">
              <i data-lucide="copy" aria-hidden="true"></i>
              <span>复制</span>
            </button>
          </div>
          <pre><code>${escapeHtml(command.text)}</code></pre>
        </div>`,
    )
    .join("");

  return `
    <article class="stage-card is-${status}${isOpen ? " is-open" : ""}" id="stage-${stage.id}" data-stage-id="${stage.id}">
      <button class="stage-header" type="button" aria-expanded="${isOpen}" aria-controls="${bodyId}">
        <span class="stage-index">${stage.id}</span>
        <span class="stage-title-wrap">
          <span class="stage-kicker">${escapeHtml(stage.phaseLabel)}</span>
          <h2>${escapeHtml(stage.title)}</h2>
          <p>${escapeHtml(stage.objective)}</p>
        </span>
        <span class="stage-status">
          <span class="stage-fraction">${doneTasks}/${stage.tasks.length}</span>
          <span class="status-chip ${status}">${statusLabels[status]}</span>
        </span>
        <span class="stage-chevron"><i data-lucide="chevron-down" aria-hidden="true"></i></span>
      </button>
      <div class="stage-body" id="${bodyId}">
        <section class="stage-section" aria-labelledby="tasks-${stage.id}">
          <h3 id="tasks-${stage.id}">操作清单</h3>
          <div class="task-list">${taskRows}</div>
        </section>
        <section class="stage-section detail-grid">
          <div>
            <h3>阶段产出</h3>
            <ul class="plain-list">${outputs}</ul>
          </div>
          <div>
            <h3>验收标准</h3>
            <ul class="plain-list acceptance-list">${acceptance}</ul>
          </div>
        </section>
        <section class="stage-section">
          <h3>参考命令</h3>
          <div class="command-list">${commands}</div>
        </section>
      </div>
    </article>`;
}

function updateMetrics(currentStage) {
  const totalTasks = stages.reduce((total, stage) => total + stage.tasks.length, 0);
  const doneTasks = stages.reduce((total, stage) => total + completedTaskCount(stage), 0);
  const doneStages = stages.filter((stage) => completedTaskCount(stage) === stage.tasks.length).length;
  const percentage = totalTasks === 0 ? 0 : Math.round((doneTasks / totalTasks) * 100);

  document.querySelector("#sidebarProgressText").textContent = `${percentage}%`;
  document.querySelector("#sidebarProgressBar").style.width = `${percentage}%`;
  document.querySelector("#sidebarTaskCount").textContent = `${doneTasks} / ${totalTasks} 项操作已完成`;
  document.querySelector("#completedStages").textContent = `${doneStages} / ${stages.length}`;
  document.querySelector("#completedTasks").textContent = `${doneTasks} / ${totalTasks}`;

  const currentStageElement = document.querySelector("#currentStage");
  const currentHintElement = document.querySelector("#currentStageHint");
  if (currentStage) {
    currentStageElement.textContent = `${currentStage.id} · ${currentStage.title}`;
    currentHintElement.textContent = currentStage.objective;
  } else {
    currentStageElement.textContent = "全部阶段已完成";
    currentHintElement.textContent = "所有操作项均已通过验收";
  }
}

function refreshIcons() {
  if (window.lucide?.createIcons) window.lucide.createIcons();
}

function render() {
  const currentStage = getCurrentStage();
  if (openStages.size === 0 && currentStage) openStages.add(currentStage.id);

  const searchTerm = searchInput.value.trim().toLocaleLowerCase("zh-CN");
  const visibleStages = stages.filter((stage) => {
    const status = getStageStatus(stage, currentStage);
    const phaseMatches = activePhase === "all" || stage.phase === activePhase;
    const statusMatches = activeStatus === "all" || status === activeStatus;
    const searchMatches = !searchTerm || stageSearchText(stage).includes(searchTerm);
    return phaseMatches && statusMatches && searchMatches;
  });

  stageList.innerHTML = visibleStages.map((stage) => stageCard(stage, currentStage)).join("");
  emptyState.hidden = visibleStages.length > 0;
  updateMetrics(currentStage);
  refreshIcons();
}

stageList.addEventListener("click", (event) => {
  const header = event.target.closest(".stage-header");
  if (header) {
    const card = header.closest(".stage-card");
    if (openStages.has(card.dataset.stageId)) openStages.delete(card.dataset.stageId);
    else openStages.add(card.dataset.stageId);
    render();
    return;
  }

  const copyButton = event.target.closest(".copy-button");
  if (copyButton) copyCommand(copyButton);
});

stageList.addEventListener("change", (event) => {
  const checkbox = event.target.closest("input[data-task-id]");
  if (!checkbox) return;

  progress[checkbox.dataset.taskId] = checkbox.checked;
  saveProgress();
  render();
});

phaseButtons.forEach((button) => {
  button.addEventListener("click", () => {
    activePhase = button.dataset.phase;
    phaseButtons.forEach((item) => item.classList.toggle("is-active", item === button));
    render();
  });
});

statusButtons.forEach((button) => {
  button.addEventListener("click", () => {
    activeStatus = button.dataset.status;
    statusButtons.forEach((item) => item.classList.toggle("is-active", item === button));
    render();
  });
});

searchInput.addEventListener("input", render);

document.querySelector("#expandCurrent").addEventListener("click", () => {
  const currentStage = getCurrentStage();
  if (!currentStage) return;

  activePhase = "all";
  activeStatus = "all";
  searchInput.value = "";
  phaseButtons.forEach((button) => button.classList.toggle("is-active", button.dataset.phase === "all"));
  statusButtons.forEach((button) => button.classList.toggle("is-active", button.dataset.status === "all"));
  openStages.add(currentStage.id);
  render();
  document.querySelector(`[data-stage-id="${currentStage.id}"]`)?.scrollIntoView({ behavior: "smooth" });
});

document.querySelector("#resetProgress").addEventListener("click", () => {
  const confirmed = window.confirm("重置当前浏览器中的全部任务进度？A0 和 A1 会恢复为已完成状态。");
  if (!confirmed) return;

  progress = defaultProgress();
  openStages = new Set([getCurrentStage()?.id].filter(Boolean));
  saveProgress();
  render();
});

document.querySelector("#printPlan").addEventListener("click", () => window.print());

async function copyCommand(button) {
  const stage = stages.find((item) => item.id === button.dataset.commandStage);
  const command = stage?.commands[Number(button.dataset.commandIndex)]?.text;
  if (!command) return;

  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(command);
    } else {
      const textarea = document.createElement("textarea");
      textarea.value = command;
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.append(textarea);
      textarea.select();
      document.execCommand("copy");
      textarea.remove();
    }
    const label = button.querySelector("span");
    label.textContent = "已复制";
    window.setTimeout(() => {
      label.textContent = "复制";
    }, 1200);
  } catch {
    window.prompt("复制下面的命令：", command);
  }
}

render();

if (window.location.hash) {
  window.requestAnimationFrame(() => {
    const target = document.querySelector(window.location.hash);
    if (!target) return;

    const previousScrollBehavior = document.documentElement.style.scrollBehavior;
    document.documentElement.style.scrollBehavior = "auto";
    target.scrollIntoView();
    document.documentElement.style.scrollBehavior = previousScrollBehavior;
  });
}
