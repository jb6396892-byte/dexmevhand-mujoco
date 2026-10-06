"""Random tabletop controls; language planning remains unchanged."""
import json
from pathlib import Path
from PySide6.QtWidgets import QWidget, QFormLayout, QVBoxLayout, QTabWidget, QComboBox, QDoubleSpinBox, QSpinBox, QToolButton, QStyle, QTableWidgetItem, QLabel
from .tabletop_window import TabletopWindow
from .runtime import ROOT


class RandomTabletopWindow(TabletopWindow):
    def __init__(self, storage, visual_root, allow_candidate=False, checkpoint=None, protocol=None):
        super().__init__(storage, visual_root, allow_candidate, checkpoint)
        self.random_protocol=Path(protocol).resolve() if protocol else ROOT/'configs/tabletop-random-v5.json'
        self.random_config=json.loads(self.random_protocol.read_text())
        self.workflow.addItem('随机桌面 · 已知初始位置', 'random')
        self.count=QSpinBox(); self.count.setRange(-1,4); self.count.setValue(-1)
        self.count.setSpecialValueText('干扰物数量：随机'); self.count.setPrefix('干扰物 ')
        self.count.setToolTip('不包含待抓取杯子，0 至 4 件')
        self.position_tabs=QTabWidget()
        self.cup_mode=QComboBox(); self.cup_mode.addItems(['初始杯位置：跟随种子','初始杯位置：手动坐标'])
        self.cup_box=QWidget(); cup_form=QFormLayout(self.cup_box); cup_form.setContentsMargins(0,0,0,0)
        self.cup_inputs=[]
        for i,axis in enumerate('XY'):
            spin=QDoubleSpinBox(); spin.setDecimals(1); spin.setSingleStep(5); spin.setSuffix(' mm')
            spin.setRange(self.random_config['cup_xy_min_m'][i]*1000,self.random_config['cup_xy_max_m'][i]*1000)
            spin.setToolTip('初始杯子世界坐标；高度由桌面支撑确定')
            cup_form.addRow('初始 '+axis,spin); self.cup_inputs.append(spin)
        self.target_mode=QComboBox(); self.target_mode.addItems(['目标位置：跟随种子','目标位置：手动坐标'])
        self.target_box=QWidget(); row=QFormLayout(self.target_box); row.setContentsMargins(0,0,0,0); row.setSpacing(3)
        self.targets=[]
        for i,axis in enumerate('XYZ'):
            spin=QDoubleSpinBox(); spin.setDecimals(1); spin.setSingleStep(5)
            spin.setRange(self.random_config['goal_min_m'][i]*1000,self.random_config['goal_max_m'][i]*1000)
            spin.setSuffix(' mm'); spin.setMinimumWidth(0)
            spin.setToolTip('杯子模型原点的世界坐标，不是杯底高度')
            row.addRow('目标 '+axis,spin); self.targets.append(spin)
        for title,mode,box in [('初始位置',self.cup_mode,self.cup_box),('目标位置',self.target_mode,self.target_box)]:
            page=QWidget(); layout=QVBoxLayout(page); layout.setContentsMargins(4,4,4,4)
            layout.addWidget(mode); layout.addWidget(box); layout.addStretch()
            self.position_tabs.addTab(page,title)
        self.shuffle=QToolButton(); self.shuffle.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        self.shuffle.setToolTip('切换下一桌面种子'); self.shuffle.clicked.connect(lambda:self.seed.setValue(self.seed.value()+1))
        for i,widget in enumerate((self.count,self.position_tabs,self.shuffle)):
            self.sidebar.layout().insertWidget(3+i,widget)
        self.target_mode.currentIndexChanged.connect(self.update_random_controls)
        self.cup_mode.currentIndexChanged.connect(self.update_random_controls)
        self.seed.valueChanged.connect(self.update_random_controls)
        self.workflow.setCurrentIndex(self.workflow.findData('random'))
        self.setWindowTitle('抓杯实验台 · 随机桌面与可调目标')
        self.update_random_controls()

    def update_random_controls(self):
        if not hasattr(self,'targets'): return
        random=self.workflow.currentData()=='random'
        for widget in (self.count,self.position_tabs,self.shuffle):
            widget.setVisible(random); widget.setEnabled(random and not self.busy)
        for mode,spins in ((self.target_mode,self.targets),(self.cup_mode,self.cup_inputs)):
            manual=mode.currentIndex()==1
            for spin in spins:
                if not manual:
                    spin.setSpecialValueText('自动'); spin.setValue(spin.minimum())
                else:
                    if spin.specialValueText(): spin.setValue((spin.minimum()+spin.maximum())/2)
                    spin.setSpecialValueText('')
                spin.setEnabled(random and manual and not self.busy)

    def mode_changed(self):
        super().mode_changed()
        if self.workflow.currentData()=='random':
            self.lock_label.setText('已知初始杯位姿 · 随机桌面' if self.allow_candidate else '随机桌面 · 执行锁定')
            self.seed.setEnabled(not self.busy)
            self.run_button.setEnabled(self.allow_candidate and not self.busy)
            self.camera.setEnabled(False); self.zoom.setEnabled(False)
            self.scene.setItemText(0,'第一视频 · 随机位置抓法'); self.scene.setItemText(1,'第二视频 · 直立杯功能抓法')
            for label in self.sidebar.findChildren(QLabel):
                if label.text().startswith('低层后端：'): label.setText('低层后端：学习残差 + 手部跟踪')
        if hasattr(self,'views'):
            for i in (1,2,3): self.views.setTabEnabled(i,self.workflow.currentData()!='random')
            if self.workflow.currentData()=='random': self.views.setCurrentIndex(0)
        self.update_random_controls()

    def start(self, execute=True):
        if execute and self.workflow.currentData()=='random' and not self.allow_candidate:
            self.state.setText('执行锁定'); return
        super().start(execute)

    def start_worker(self, kind, program, args, environment):
        if kind=='physics' and self.workflow.currentData()=='random':
            if not self.allow_candidate: raise RuntimeError('Random tabletop execution locked')
            args=[str(ROOT/'scripts/168_stream_random_tabletop.py'),'--root',str(self.storage),
                  '--output',str(self.output),'--visual-root',str(self.visual_root),'--seed',str(self.seed.value()),
                  '--checkpoint',str(self.checkpoint),'--protocol',str(self.random_protocol)]
            if self.count.value()>=0: args.extend(['--count',str(self.count.value())])
            if self.target_mode.currentIndex()==1:
                args.extend(['--target-world']+[str(spin.value()/1000) for spin in self.targets])
            if self.cup_mode.currentIndex()==1:
                args.extend(['--cup-xy']+[str(spin.value()/1000) for spin in self.cup_inputs])
        super().start_worker(kind,program,args,environment)

    def message(self, packet):
        if packet['type']=='random_scene':
            from pathlib import Path
            self.visual_output=Path(packet['output']); layout=packet['layout']
            fields=[('桌面种子',str(layout['seed'])),('干扰物数量',str(layout['distractor_count'])),
                ('物品',', '.join(o['name'] for o in layout['objects'])),
                ('初始杯位置 XYZ (mm)',', '.join('%.2f'%(packet['initial_known_pose'][i][3]*1000) for i in range(3))),
                ('目标 XYZ (mm)',', '.join('%.2f'%(v*1000) for v in layout['goal_world_m'])),
                ('控制输入','已知初始位姿 + 关节反馈；非在线视觉定位'),
                ('画面指标','仿真真值，仅作阶段验收与显示'),('学习模型',packet['checkpoint'])]
            self.parameters.setHorizontalHeaderLabels(['随机任务参数','数值']); self.parameters.setRowCount(len(fields))
            for i,(name,value) in enumerate(fields):
                self.parameters.setItem(i,0,QTableWidgetItem(name)); self.parameters.setItem(i,1,QTableWidgetItem(value))
            return
        super().message(packet)
        if packet['type']=='frame' and self.workflow.currentData()=='random':
            self.view_state.setText('MuJoCo · 学习残差 | 已知初始位置')
