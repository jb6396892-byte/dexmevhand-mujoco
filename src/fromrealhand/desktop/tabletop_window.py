"""Candidate visual-control panel; the accepted legacy desktop remains unchanged."""
import base64
import json
import math
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices, QImage
from PySide6.QtWidgets import QComboBox, QLabel, QSpinBox, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView
from .window import GraspWindow, FrameView
from .runtime import ROOT


class TabletopWindow(GraspWindow):
    def __init__(self, storage, visual_root, allow_candidate=False, checkpoint=None):
        super().__init__(storage)
        self.setStyleSheet(self.styleSheet()+'\nQPushButton#run:disabled { background: #edf0f2; color: #9ca5ad; border-color: #c8d0d5; }')
        self.visual_root = Path(visual_root).resolve()
        self.allow_candidate = allow_candidate
        self.checkpoint = Path(checkpoint).resolve() if checkpoint else None
        if self.checkpoint and not self.checkpoint.is_file():
            raise ValueError('Policy checkpoint does not exist: '+str(self.checkpoint))
        self.scene.setCurrentIndex(0)
        self.visual_output = None
        self.setWindowTitle('抓杯实验台 · RGB-D 接触跟踪（开发验证）')
        self.workflow = QComboBox()
        self.workflow.addItem('桌面 RGB-D · 候选后端', 'tabletop')
        self.workflow.addItem('原标准场景 · 已验收', 'legacy')
        self.sidebar.layout().insertWidget(1, self.workflow)
        self.seed = QSpinBox(); self.seed.setRange(0, 99999); self.seed.setPrefix('桌面种子 ')
        self.sidebar.layout().insertWidget(2, self.seed)
        self.lock_label = QLabel(); self.lock_label.setWordWrap(True)
        self.sidebar.layout().insertWidget(3, self.lock_label)
        # Reuse the old screen without changing its frozen source or callbacks.
        layout = self.frame.parentWidget().layout(); index = layout.indexOf(self.frame)
        layout.removeWidget(self.frame)
        self.views = QTabWidget(); self.views.addTab(self.frame, '仿真画面')
        self.rgb, self.depth, self.overlay = FrameView(), FrameView(), FrameView()
        for name, view in [('RGB',self.rgb),('深度',self.depth),('杯子定位',self.overlay)]:
            view.placeholder = '尚未采集'; self.views.addTab(view, name)
        layout.insertWidget(index, self.views, 1)
        self.parameters = QTableWidget(0, 2)
        self.parameters.setHorizontalHeaderLabels(['视觉参数','数值'])
        self.parameters.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.parameters.verticalHeader().hide()
        self.parameters.setEditTriggers(QTableWidget.NoEditTriggers)
        tabs = next(t for t in self.findChildren(QTabWidget) if t.indexOf(self.raw) >= 0)
        tabs.addTab(self.parameters, 'RGB-D 参数')
        self.info_tabs = tabs
        self.workflow.currentIndexChanged.connect(self.mode_changed)
        self.mode_changed()

    def mode_changed(self):
        tabletop = self.workflow.currentData() == 'tabletop'
        self.lock_label.setText(('仿真执行已显式解锁 · 验收范围见报告' if self.allow_candidate else '桌面开发模式 · 默认执行锁定')
                               if tabletop else '原标准场景执行口径不变')
        self.seed.setEnabled(tabletop and not self.busy)
        self.scene.setItemText(0, '第一视频 · 参考抓法' if tabletop else '第一视频 · 标准场景')
        self.scene.setItemText(1, '第二视频 · 直立杯功能抓法' if tabletop else '第二视频 · 标准场景')
        self.camera.setEnabled(not tabletop); self.zoom.setEnabled(not tabletop)
        self.run_button.setEnabled(not self.busy and (not tabletop or self.allow_candidate))
        self.run_button.setToolTip('桌面执行尚未显式授权' if tabletop and not self.allow_candidate else '执行当前技能计划')
        for label in self.sidebar.findChildren(QLabel):
            if label.text().startswith('低层后端：'):
                label.setText(('低层后端：学习策略 · '+self.checkpoint.parent.name if self.checkpoint
                    else '低层后端：接触约束与参考跟踪') if tabletop else '低层后端：已验证专家参考')

    def set_busy(self, value):
        super().set_busy(value)
        self.workflow.setEnabled(not value); self.mode_changed()

    def start(self, execute=True):
        if execute and self.workflow.currentData() == 'tabletop' and not self.allow_candidate:
            self.state.setText('测试前锁定'); return
        self.visual_output = None
        for view in (self.rgb, self.depth, self.overlay): view.set_image(QImage())
        self.parameters.setRowCount(0)
        super().start(execute)

    def start_worker(self, kind, program, args, environment):
        if kind == 'physics' and self.workflow.currentData() == 'tabletop':
            if not self.allow_candidate: raise RuntimeError('Tabletop candidate remains locked')
            args = [str(ROOT/'scripts/143_stream_visual_grasp.py'), '--root', str(self.storage),
                    '--output', str(self.output), '--visual-root', str(self.visual_root),
                    '--seed', str(self.seed.value()), '--allow-unvalidated-tabletop']
            if self.checkpoint: args.extend(['--checkpoint',str(self.checkpoint)])
            self.note('新桌面候选执行；不继承旧场景验收，定位期间暂停物理时间')
        super().start_worker(kind, program, args, environment)

    def show_estimate(self, row):
        pose = row.get('T_world_object')
        camera_pose = row.get('T_camera_object')
        calibration = row.get('calibration', {})
        xyz = lambda t: ', '.join('%.2f'%(t[i][3]*1000) for i in range(3)) if t else '--'
        values = [('准入', str(row.get('reason','--'))), ('世界 XYZ (mm)', xyz(pose)),
                  ('相机 XYZ (mm)', xyz(camera_pose)),
                  ('世界偏航 (deg)', '%.2f'%math.degrees(math.atan2(pose[1][0],pose[0][0])) if pose else '--'),
                  ('配准内点比例', '%.3f'%row.get('fitness',0)),
                  ('原始点云内点比例', '%.3f'%row['raw_fitness'] if 'raw_fitness' in row else '--'),
                  ('表面筛选保留比例', '%.3f'%row['retained_fraction'] if 'retained_fraction' in row else '--'),
                  ('配准 RMSE (mm)', '%.3f'%(1000*row['rmse_m']) if 'rmse_m' in row else '--'),
                  ('定位墙钟时间 (s)', '%.3f'%row.get('wall_latency_s',row.get('total_s',0))),
                  ('采样仿真时间 (s)', '%.3f'%row.get('camera_time_s',0)),
                  ('目标 XYZ (mm)', ', '.join('%.2f'%(v*1000) for v in row['goal_world_m']) if row.get('goal_world_m') else '--'),
                  ('RGB-D 相机',calibration.get('camera','--')),
                  ('检测续接',row.get('association',{}).get('mode','--')),
                  ('旋转稳健重配准','是' if row.get('point_to_point_rotation_fallback') else '否'),
                  ('深度单位', calibration.get('depth_units','metres')),
                  ('相机 K', json.dumps(calibration.get('K',[]))),
                  ('相机到世界 T', json.dumps(calibration.get('T_world_camera',[]))),
                  ('低层策略', '参考条件学习策略' if self.checkpoint else '专家参考控制'),
                  ('模型文件', str(self.checkpoint) if self.checkpoint else '--'),
                  ('运行口径', '受控仿真开发，不代表实物验收')]
        self.parameters.setRowCount(len(values))
        for i,(name,value) in enumerate(values):
            self.parameters.setItem(i,0,QTableWidgetItem(name))
            item=QTableWidgetItem(value); item.setToolTip(value); self.parameters.setItem(i,1,item)

    def message(self, message):
        if message['type'] == 'perception':
            self.visual_output = Path(message['output'])
            self.show_estimate(message['estimate'])
            for name,view in [('rgb',self.rgb),('depth',self.depth),('overlay',self.overlay)]:
                image = QImage.fromData(base64.b64decode(message['images'][name]), 'JPG')
                if image.isNull(): raise ValueError('Invalid RGB-D image')
                view.set_image(image)
            return
        super().message(message)
        if message['type'] == 'frame' and self.workflow.currentData() == 'tabletop':
            row = message['metrics']
            self.view_state.setText('MuJoCo · '+('学习策略' if self.checkpoint else '专家参考')+' | 杯子指标来自 RGB-D')
            self.statusBar().showMessage('控制步数 %d | 参考步 %d | 位姿年龄 %.3f s | 对握 %s'%
                (message.get('actual_control_steps',message['step']),message['step'],row.get('pose_sim_age_s',0),
                 '是' if row.get('opposition') else '否'))

    def open_folder(self):
        if self.visual_output:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.visual_output)))
        else: super().open_folder()

    def load_offline_evidence(self):
        """Presentation preview only: no process, simulator, model or control starts."""
        folder = ROOT/'docs/presentation/tabletop_rgbd/evidence'
        for view,file in [(self.frame,'tabletop-rgb.png'),(self.rgb,'tabletop-rgb.png'),
                          (self.depth,'tabletop-depth.png'),(self.overlay,'noisy-pose.png')]:
            view.set_image(QImage(str(folder/file)))
        row = json.loads((folder/'example-estimate.json').read_text())
        self.show_estimate(row)
        self.views.setCurrentWidget(self.overlay); self.info_tabs.setCurrentWidget(self.parameters)
        self.state.setText('离线资料预览 · 未执行')
        self.view_state.setText('上一轮感知截图 · 非本轮抓取回放')
        self.lock_label.setText('离线资料预览 · 执行锁定')
        self.run_button.setEnabled(False); self.preview.setEnabled(False)
        self.workflow.setEnabled(False)
        self.statusBar().showMessage('仅界面排版截图；没有启动语言、感知或物理进程')
