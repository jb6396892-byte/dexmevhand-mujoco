"""Known-pose full-table navigation controls on the existing language/Qt console."""
from PySide6.QtWidgets import QComboBox,QDoubleSpinBox,QFormLayout,QWidget,QLabel,QTableWidgetItem
from .random_window import RandomTabletopWindow
from .runtime import ROOT


class NavigationWindow(RandomTabletopWindow):
    def __init__(self,storage,visual_root,allow_candidate=False,checkpoint=None,protocol=None):
        super().__init__(storage,visual_root,allow_candidate,checkpoint,
                         protocol or ROOT/'configs/tabletop-navigation-v5.json')
        self.workflow.addItem('整桌导航 · 抓取 · 搬运','navigation')
        self.tuning=QWidget();form=QFormLayout(self.tuning);form.setContentsMargins(0,0,0,0)
        self.grasp_mode=QComboBox();self.grasp_mode.addItem('自动选择可行抓法','auto');self.grasp_mode.addItem('固定所选视频抓法','fixed')
        self.speed=QDoubleSpinBox();self.speed.setRange(.5,1.);self.speed.setSingleStep(.1);self.speed.setValue(1.)
        self.speed.setSuffix(' x');self.speed.setToolTip('导航和搬运速度倍率')
        self.clearance=QDoubleSpinBox();self.clearance.setRange(25,40);self.clearance.setSingleStep(5);self.clearance.setValue(25)
        self.clearance.setSuffix(' mm');self.clearance.setToolTip('远距离规划净空；物理安全门槛不变')
        self.completion=QComboBox();self.completion.addItem('搬运并保持','hold');self.completion.addItem('放置、松手并返回起点','place')
        self.lower_speed=QDoubleSpinBox();self.lower_speed.setRange(3,10);self.lower_speed.setValue(8);self.lower_speed.setSuffix(' mm/s')
        self.release_time=QDoubleSpinBox();self.release_time.setRange(4,8);self.release_time.setValue(5);self.release_time.setSuffix(' s')
        self.retreat_distance=QDoubleSpinBox();self.retreat_distance.setRange(100,200);self.retreat_distance.setValue(120);self.retreat_distance.setSuffix(' mm')
        form.addRow('完成动作',self.completion)
        form.addRow('抓法',self.grasp_mode);form.addRow('速度',self.speed);form.addRow('规划余量',self.clearance)
        form.addRow('下降速度',self.lower_speed);form.addRow('释放时长',self.release_time);form.addRow('撤离距离',self.retreat_distance)
        self.placement_state=QLabel('');self.placement_state.setWordWrap(True);form.addRow(self.placement_state)
        self.completion.currentIndexChanged.connect(self.completion_changed)
        self.sidebar.layout().insertWidget(6,self.tuning)
        self.sidebar.layout().removeWidget(self.stop_button)
        self.camera.parentWidget().layout().itemAt(0).layout().insertWidget(1,self.stop_button)
        # Other workflows use different protocol schemas and keep their own launchers.
        self.workflow.blockSignals(True)
        self.workflow.clear();self.workflow.addItem('整桌导航 · 抓取 · 搬运','navigation')
        self.workflow.blockSignals(False);self.mode_changed()
        self.setWindowTitle('抓杯实验台 · 整桌导航与抓取')
        self.joint.setWordWrap(True)
        self.seed.setValue(5301);self.count.setValue(2)
        self.completion_changed()

    def completion_changed(self):
        active=self.completion.currentData()=='place'
        for widget in (self.lower_speed,self.release_time,self.retreat_distance):widget.setEnabled(active and not self.busy)
        self.targets[2].setEnabled(not active and self.target_mode.currentIndex()==1 and not self.busy)
        self.targets[2].setToolTip('放置高度按杯底和桌面自动计算；此处为搬运悬停高度' if active else '目标 Z')
        if active:
            self.targets[2].setSpecialValueText('');self.targets[2].setValue(200)
            self.instruction.setPlainText('把杯子放到目标位置并返回起点')
            self.placement_state.setText('放置计划：规则控制；等待执行')
        else:
            self.placement_state.setText('')
            from .placement_planning import INSTRUCTIONS
            if self.instruction.toPlainText() in INSTRUCTIONS:self.instruction.setPlainText('把杯子搬到目标位置')

    def update_random_controls(self):
        if not hasattr(self,'targets'):return
        active=self.workflow.currentData() in ('random','navigation')
        for widget in (self.count,self.position_tabs,self.shuffle):
            widget.setVisible(active);widget.setEnabled(active and not self.busy)
        for mode,spins in ((self.target_mode,self.targets),(self.cup_mode,self.cup_inputs)):
            manual=mode.currentIndex()==1
            for spin in spins:
                if not manual:spin.setSpecialValueText('自动');spin.setValue(spin.minimum())
                else:
                    if spin.specialValueText():spin.setValue((spin.minimum()+spin.maximum())/2)
                    spin.setSpecialValueText('')
                spin.setEnabled(active and manual and not self.busy)
        if hasattr(self,'tuning'):
            self.tuning.setVisible(self.workflow.currentData()=='navigation');self.tuning.setEnabled(not self.busy)
            if hasattr(self,'completion'):
                place=self.completion.currentData()=='place'
                for widget in (self.lower_speed,self.release_time,self.retreat_distance):widget.setEnabled(place and not self.busy)
                if place:
                    self.targets[2].setSpecialValueText('');self.targets[2].setValue(200);self.targets[2].setEnabled(False)

    def mode_changed(self):
        super().mode_changed()
        for i in (1,2,3):self.views.setTabVisible(i,self.workflow.currentData()!='navigation')
        navigation=self.workflow.currentData()=='navigation'
        self.camera.setVisible(not navigation);self.zoom.setVisible(not navigation)
        self.info_tabs.setTabText(self.info_tabs.indexOf(self.parameters),'导航参数' if navigation else 'RGB-D 参数')
        if self.workflow.currentData()!='navigation':return
        self.lock_label.setText('整桌物理任务 · 候选预演后执行' if self.allow_candidate else '整桌任务 · 执行锁定')
        self.seed.setEnabled(not self.busy);self.run_button.setEnabled(self.allow_candidate and not self.busy)
        self.camera.setEnabled(False);self.zoom.setEnabled(False)
        self.scene.setItemText(0,'优先第一视频抓法');self.scene.setItemText(1,'优先第二视频抓法')
        self.views.setCurrentIndex(0)
        for i in (1,2,3):self.views.setTabEnabled(i,False)
        for label in self.sidebar.findChildren(QLabel):
            if label.text().startswith('低层后端：'):label.setText('低层后端：局部学习策略 + 避障伺服')
        self.update_random_controls()

    def start(self,execute=True):
        if execute and self.workflow.currentData()=='navigation' and not self.allow_candidate:
            self.state.setText('执行锁定');return
        super().start(execute)

    def start_worker(self,kind,program,args,environment):
        place=self.completion.currentData()=='place'
        if kind=='plan' and place:
            from .runtime import LEGACY_PYTHON,physics_environment
            args=list(args);args[0]=str(ROOT/'scripts/196_plan_placement.py')
            program=LEGACY_PYTHON;environment=physics_environment()
        if kind=='physics' and self.workflow.currentData()=='navigation':
            if not self.allow_candidate:raise RuntimeError('Navigation execution locked')
            args=[str(ROOT/'scripts/190_stream_navigation_task.py'),'--root',str(self.storage),'--output',str(self.output),
                '--visual-root',str(self.visual_root),'--checkpoint',str(self.checkpoint),'--protocol',str(self.random_protocol),
                '--seed',str(self.seed.value()),'--mode',self.grasp_mode.currentData(),
                '--speed',str(self.speed.value()),'--clearance',str(self.clearance.value()/1000)]
            if place:
                args[args.index('--protocol')+1]=str(ROOT/'configs/tabletop-placement-v1.json')
                args+=['--place','--lower-speed',str(self.lower_speed.value()/1000),
                       '--release-seconds',str(self.release_time.value()),'--retreat',str(self.retreat_distance.value()/1000)]
            if self.count.value()>=0:args+=['--count',str(self.count.value())]
            if self.target_mode.currentIndex()==1:args+=['--target-world']+[str(v.value()/1000) for v in self.targets]
            if self.cup_mode.currentIndex()==1:args+=['--cup-xy']+[str(v.value()/1000) for v in self.cup_inputs]
        super().start_worker(kind,program,args,environment)

    def message(self,packet):
        if packet['type']=='navigation_settings':
            values=[('实际抓法',packet['selected_video']),('候选选择',packet['mode']),
                    ('速度倍率',str(packet['speed'])),('规划余量 mm',str(packet['clearance_m']*1000)),
                    ('局部入口参考帧',str(packet.get('entry_reference_step',0))),
                    ('抓取方位 deg',str(packet.get('grasp_yaw_deg',0)))]
            for name,value in values:
                i=self.parameters.rowCount();self.parameters.insertRow(i)
                self.parameters.setItem(i,0,QTableWidgetItem(name));self.parameters.setItem(i,1,QTableWidgetItem(value))
            return
        super().message(packet)
        if packet['type']=='frame' and self.workflow.currentData()=='navigation':
            name=dict(navigate='导航',approach='低速接近',reach='接近杯子',grasp='闭合',lift='抬杯',transport='带杯搬运',
                preplace='放置纠姿',lower='低速下降',settle='桌面承重',release='渐进松手',retreat='空手撤离',
                verify='站稳确认',return_home='返回起点').get(packet['skill'],packet['skill'])
            self.view_state.setText('MuJoCo · '+name+' | 已知初始位置')
            if packet.get('control_kind')=='navigation_servo':self.action.setText('控制方式：位置伺服')
            metrics=packet['metrics']
            if 'placement' in metrics:
                q=metrics['placement']
                self.placement_state.setText('%s | 倾角 %.1f° | 承重 %.2f N\n手杯力 %.2f N | 杯底间隙 %.2f mm'%
                    (name,q['tilt_deg'],q['table_force_n'],q['hand_cup_force_n'],q['bottom_gap_m']*1000))
            if 'root_translation_violation_m' in metrics:
                self.joint.setText('转动越限 %.5f rad\n平移越限 %.3f mm'%
                    (metrics['joint_violation_rad'],metrics['root_translation_violation_m']*1000))

    def finish(self,report):
        super().finish(report)
        if self.workflow.currentData()=='navigation' and report.get('passed'):
            for skill in report.get('completed',[]):self.phase_bars[skill].setValue(100)
            if report.get('carry',{}).get('passed'):self.phase_bars['transport'].setValue(100)
