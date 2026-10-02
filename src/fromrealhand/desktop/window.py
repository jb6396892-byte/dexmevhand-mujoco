"""Qt Widgets control surface. No torch or MuJoCo imports in the UI process."""
import base64
import json
from pathlib import Path
import time

from PySide6.QtCore import Qt,QProcess,QProcessEnvironment,QTimer,QUrl,QRectF,Signal
from PySide6.QtGui import QColor,QDesktopServices,QImage,QPainter,QPalette,QPen
from PySide6.QtWidgets import (QApplication,QComboBox,QFileDialog,QFormLayout,QFrame,QGridLayout,
    QHBoxLayout,QHeaderView,QLabel,QMainWindow,QPlainTextEdit,QProgressBar,QPushButton,QSlider,
    QLayout,QScrollArea,QSplitter,QStyle,QTabWidget,QTableWidget,QTableWidgetItem,QToolButton,QVBoxLayout,QWidget)

from .runtime import ROOT,LEGACY_PYTHON,JsonLines,language_environment,physics_environment

SKILLS = [('reach','接近'),('grasp','闭合'),('lift','抬杯'),('transport','搬运')]
STYLE = '''
QMainWindow, QWidget#body { background: #f4f6f7; color: #20262c; }
QWidget { font-family: "Noto Sans CJK SC", "Sans Serif"; font-size: 13px; color: #20262c; }
QLabel#title { font-size: 21px; font-weight: 600; }
QLabel#section { font-size: 14px; font-weight: 600; color: #303941; }
QLabel#muted { color: #67727c; font-size: 12px; }
QLabel#value { font-size: 23px; font-weight: 600; }
QFrame#sidebar { background: white; border: 1px solid #dce1e4; border-radius: 4px; }
QPushButton { padding: 8px 12px; border: 1px solid #c8d0d5; border-radius: 4px; background: white; }
QPushButton:hover { background: #edf1f3; }
QPushButton#run { background: #087f68; color: white; border-color: #087f68; }
QPushButton#run:hover { background: #066a57; }
QPushButton#stop { color: #ad252b; border-color: #e6b5b8; }
QPushButton:disabled, QToolButton:disabled { color: #9ca5ad; background: #edf0f2; }
QComboBox,QPlainTextEdit { background: white; border: 1px solid #c8d0d5; border-radius: 3px; padding: 6px; }
QComboBox { min-height: 24px; }
QToolButton { border: 1px solid #c8d0d5; border-radius: 3px; background: white; padding: 6px; }
QToolButton:hover { background: #e7eef0; }
QProgressBar { border: none; border-radius: 2px; background: #dfe5e8; height: 7px; }
QProgressBar::chunk { background: #148773; }
QTableWidget { border: none; background: white; gridline-color: #edf0f2; }
QHeaderView::section { background: #f1f4f5; border: none; padding: 5px; color: #53606b; }
QTabWidget::pane { border: 1px solid #d5dde1; background: white; }
QTabBar::tab { padding: 6px 14px; background: #e6ebee; }
QTabBar::tab:selected { background: white; color: #087f68; }
'''


class FrameView(QWidget):
    def __init__(self):
        super().__init__(); self.image=QImage(); self.placeholder='待执行'; self.setMinimumSize(400,260)
        self.setObjectName('frameView')

    def set_image(self,image): self.image=image; self.update()

    def paintEvent(self,event):
        p=QPainter(self); p.fillRect(self.rect(),QColor('#252a2e'))
        if self.image.isNull():
            p.setPen(QColor('#c2ccd2')); p.drawText(self.rect(),Qt.AlignCenter,self.placeholder); return
        size=self.image.size().scaled(self.size(),Qt.KeepAspectRatio)
        x=(self.width()-size.width())//2; y=(self.height()-size.height())//2
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.drawImage(QRectF(x,y,size.width(),size.height()),self.image)


class TraceView(QWidget):
    def __init__(self):
        super().__init__(); self.points=[]; self.setFixedHeight(108)

    def add(self,height,distance):
        self.points.append((height,distance)); self.points=self.points[-600:]; self.update()

    def paintEvent(self,event):
        p=QPainter(self); p.fillRect(self.rect(),QColor('#ffffff'))
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QColor('#68757d')); p.drawText(10,18,'杯底高度 / 目标距离 (mm)')
        w,h=self.width()-24,self.height()-34
        p.setPen(QPen(QColor('#e3e8eb'),1)); p.drawLine(12,h+24,w+12,h+24)
        if len(self.points)<2: return
        maximum=max(100.,max(max(v) for v in self.points))
        for j,color in enumerate(('#07836a','#b86b10')):
            p.setPen(QPen(QColor(color),1.7))
            for i in range(1,len(self.points)):
                x1=12+(i-1)*w/(len(self.points)-1); x2=12+i*w/(len(self.points)-1)
                y1=24+h*(1-self.points[i-1][j]/maximum); y2=24+h*(1-self.points[i][j]/maximum)
                p.drawLine(int(x1),int(y1),int(x2),int(y2))
        p.setPen(QColor('#07836a')); p.drawText(max(12,self.width()-210),18,'高度')
        p.setPen(QColor('#b86b10')); p.drawText(max(70,self.width()-140),18,'目标距离')


class GraspWindow(QMainWindow):
    run_finished=Signal(dict)
    frame_received=Signal(dict)

    def __init__(self,storage):
        super().__init__(); self.storage=Path(storage).resolve()
        palette=QPalette()
        for role,color in ((QPalette.Window,'#f4f6f7'),(QPalette.WindowText,'#20262c'),
            (QPalette.Base,'#ffffff'),(QPalette.Text,'#20262c'),(QPalette.Button,'#ffffff'),
            (QPalette.ButtonText,'#20262c'),(QPalette.Highlight,'#087f68'),(QPalette.HighlightedText,'#ffffff')):
            palette.setColor(role,QColor(color))
        QApplication.instance().setPalette(palette)
        self.process=None; self.busy=False; self.kind=None; self.cancelled=False; self.closing=False
        self.output=None; self.pending_plan=None; self.want_execute=False; self.last_report=None
        self.frames=0; self.total_steps=0; self.bounds={}; self.run_started=0.; self.started_processes=[]
        self.setWindowTitle('真实视频抓杯 · 语言与技能实验台')
        self.resize(1360,880); self.setMinimumSize(980,700); self.setStyleSheet(STYLE)
        body=QWidget(); body.setObjectName('body'); self.setCentralWidget(body)
        layout=QVBoxLayout(body); layout.setContentsMargins(16,12,16,10); layout.setSpacing(10)
        top=QHBoxLayout(); title=QLabel('抓杯实验台'); title.setObjectName('title'); top.addWidget(title)
        top.addSpacing(18); model=QLabel('Qwen2.5-0.5B · LoRA v4 / guard2'); model.setObjectName('muted'); top.addWidget(model)
        top.addStretch(); self.state=QLabel('就绪'); self.state.setMinimumWidth(130); top.addWidget(self.state); layout.addLayout(top)
        split=QSplitter(Qt.Horizontal); layout.addWidget(split,1)
        self.sidebar_scroll=QScrollArea(); self.sidebar_scroll.setWidgetResizable(True)
        self.sidebar_scroll.setFrameShape(QFrame.NoFrame)
        self.sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.sidebar_scroll.setMinimumWidth(270); self.sidebar_scroll.setMaximumWidth(340)
        side=QFrame(); side.setObjectName('sidebar'); self.sidebar=side
        left=QVBoxLayout(side); left.setContentsMargins(14,14,14,14); left.setSpacing(10)
        left.setSizeConstraint(QLayout.SetMinimumSize)
        section=QLabel('任务'); section.setObjectName('section'); left.addWidget(section)
        self.scene=QComboBox(); self.scene.addItem('第一视频 · 标准场景','first'); self.scene.addItem('第二视频 · 标准场景','second'); self.scene.setCurrentIndex(1)
        left.addWidget(self.scene)
        self.templates=QComboBox(); self.templates.addItems(['指令模板','接近杯子','握住杯子','抓起杯子','把杯子搬到目标位置','停止'])
        self.templates.activated.connect(lambda i:self.instruction.setPlainText(self.templates.itemText(i)) if i else None)
        left.addWidget(self.templates)
        self.instruction=QPlainTextEdit('把杯子搬到目标位置'); self.instruction.setFixedHeight(90); self.instruction.setObjectName('instruction')
        left.addWidget(self.instruction)
        buttons=QHBoxLayout(); self.preview=QPushButton('检查计划'); self.preview.setIcon(self.style().standardIcon(QStyle.SP_FileDialogContentsView))
        self.run_button=QPushButton('执行'); self.run_button.setObjectName('run'); self.run_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.preview.clicked.connect(lambda:self.start(False)); self.run_button.clicked.connect(lambda:self.start(True))
        buttons.addWidget(self.preview); buttons.addWidget(self.run_button); left.addLayout(buttons)
        self.stop_button=QPushButton('停止'); self.stop_button.setObjectName('stop'); self.stop_button.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.stop_button.clicked.connect(self.stop); self.stop_button.setEnabled(False); left.addWidget(self.stop_button)
        execution=QLabel('低层后端：已验证专家参考'); execution.setObjectName('muted'); execution.setWordWrap(True); left.addWidget(execution)
        margin=QLabel('穿透上限 1.00 mm'); margin.setObjectName('muted'); left.addWidget(margin)
        heading=QLabel('仿真接触力'); heading.setObjectName('section'); left.addWidget(heading)
        self.force_table=QTableWidget(5,2); self.force_table.setHorizontalHeaderLabels(['手指','法向力 N'])
        self.force_table.verticalHeader().hide(); self.force_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.force_table.setEditTriggers(QTableWidget.NoEditTriggers); self.force_table.setSelectionMode(QTableWidget.NoSelection)
        self.force_table.setFocusPolicy(Qt.NoFocus)
        self.force_table.setFixedHeight(178)
        for i,name in enumerate(['拇指','食指','中指','无名指','小指']):
            self.force_table.setItem(i,0,QTableWidgetItem(name)); self.force_table.setItem(i,1,QTableWidgetItem('--')); self.force_table.setRowHeight(i,28)
        left.addWidget(self.force_table)
        self.joint=QLabel('关节越限 -- rad'); self.action=QLabel('最大动作幅值 --'); left.addWidget(self.joint); left.addWidget(self.action)
        self.peak=QLabel('峰值穿透 -- mm'); self.peak_m=0.; left.addWidget(self.peak)
        self.plan_label=QLabel('技能计划：--'); self.plan_label.setWordWrap(True); left.addWidget(self.plan_label)
        left.addStretch(); self.folder_button=QPushButton('运行记录'); self.folder_button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.folder_button.clicked.connect(self.open_folder); left.addWidget(self.folder_button)
        self.sidebar_scroll.setWidget(side); split.addWidget(self.sidebar_scroll)
        main=QWidget(); center=QVBoxLayout(main); center.setContentsMargins(12,0,0,0); center.setSpacing(8)
        tools=QHBoxLayout(); self.view_state=QLabel('MuJoCo · 待执行'); tools.addWidget(self.view_state); tools.addStretch()
        self.camera=QComboBox(); self.camera.addItem('正面','front'); self.camera.addItem('侧面','side'); self.camera.addItem('俯视','top')
        self.camera.currentIndexChanged.connect(self.send_camera); tools.addWidget(self.camera)
        self.zoom=QSlider(Qt.Horizontal); self.zoom.setRange(60,160); self.zoom.setValue(100); self.zoom.setFixedWidth(100); self.zoom.setToolTip('视距缩放')
        self.zoom.valueChanged.connect(self.send_camera); tools.addWidget(self.zoom)
        self.capture=QToolButton(); self.capture.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton)); self.capture.setToolTip('保存当前界面截图')
        self.capture.clicked.connect(self.save_screenshot); tools.addWidget(self.capture); center.addLayout(tools)
        self.frame=FrameView(); center.addWidget(self.frame,1)
        metrics=QGridLayout(); self.values={}
        for column,(key,name) in enumerate([('height','杯底高度'),('distance','目标距离'),('penetration','当前穿透'),('time','仿真时间')]):
            label=QLabel(name); label.setObjectName('muted'); metrics.addWidget(label,0,column)
            value=QLabel('--'); value.setObjectName('value'); self.values[key]=value; metrics.addWidget(value,1,column); metrics.setColumnStretch(column,1)
        center.addLayout(metrics)
        phase=QHBoxLayout(); self.phase_bars={}; self.phase_labels={}
        for key,name in SKILLS:
            part=QVBoxLayout(); label=QLabel(name); self.phase_labels[key]=label
            bar=QProgressBar(); bar.setRange(0,100); bar.setValue(0); bar.setTextVisible(False); bar.setFixedHeight(7)
            self.phase_bars[key]=bar; part.addWidget(label); part.addWidget(bar); phase.addLayout(part)
        center.addLayout(phase)
        self.trace=TraceView(); center.addWidget(self.trace)
        tabs=QTabWidget(); tabs.setMaximumHeight(152); tabs.setMinimumHeight(100)
        self.log=QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumBlockCount(300)
        self.raw=QPlainTextEdit(); self.raw.setReadOnly(True); self.raw.setMaximumBlockCount(100)
        tabs.addTab(self.log,'执行记录'); tabs.addTab(self.raw,'模型输出'); center.addWidget(tabs)
        split.addWidget(main); split.setSizes([290,1040])
        self.statusBar().showMessage('RTX 渲染 / 两视频标准场景')

    def note(self,message):
        self.log.appendPlainText(time.strftime('%H:%M:%S')+'  '+message)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def set_busy(self,value):
        self.busy=value
        for widget in (self.scene,self.templates,self.instruction,self.preview,self.run_button): widget.setEnabled(not value)
        self.stop_button.setEnabled(value)

    def start(self,execute=True):
        if self.busy: return
        instruction=self.instruction.toPlainText().strip()
        if not instruction or len(instruction)>120:
            self.state.setText('指令为空或超过 120 字'); return
        from datetime import datetime,timezone
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        self.output=self.storage/'language/desktop_runs'/stamp
        self.want_execute=execute; self.cancelled=False; self.pending_plan=None; self.last_report=None
        self.frames=0; self.total_steps=0; self.bounds={}; self.run_started=time.monotonic()
        self.trace.points=[]; self.trace.update(); self.frame.placeholder='待执行'; self.frame.set_image(QImage())
        self.view_state.setText('MuJoCo · 待执行'); self.plan_label.setText('技能计划：--'); self.raw.clear(); self.log.clear()
        self.sidebar_scroll.verticalScrollBar().setValue(0)
        self.statusBar().showMessage('规划中  |  '+stamp)
        for value in self.values.values(): value.setText('--')
        for bar in self.phase_bars.values(): bar.setValue(0)
        for label in self.phase_labels.values(): label.setStyleSheet('color: #53606b')
        for i in range(5): self.force_table.item(i,1).setText('--')
        self.joint.setText('关节越限 -- rad'); self.action.setText('最大动作幅值 --')
        self.peak_m=0.; self.peak.setText('峰值穿透 -- mm')
        self.set_busy(True); self.state.setText('规划中'); self.note(instruction)
        command=[str(ROOT/'scripts/125_plan_desktop.py'),instruction,'--scene',self.scene.currentData(),
                 '--root',str(self.storage),'--output',str(self.output)]
        self.start_worker('plan',str(ROOT/'data/runtime/stage6-study-venv/bin/python'),command,language_environment(self.storage))

    def start_worker(self,kind,program,args,environment):
        process=QProcess(self); self.process=process; self.kind=kind; decoder=JsonLines()
        env=QProcessEnvironment()
        for name,value in environment.items(): env.insert(name,value)
        process.setProcessEnvironment(env); process.setWorkingDirectory(str(ROOT)); process.setProgram(program); process.setArguments(args)
        process.readyReadStandardOutput.connect(lambda:self.read_output(process,decoder))
        process.readyReadStandardError.connect(lambda:self.note(bytes(process.readAllStandardError()).decode('utf-8','replace').strip()[-3000:]))
        process.finished.connect(lambda code,status:self.worker_finished(process,kind,code,status))
        process.errorOccurred.connect(lambda error:self.process_error(process,error))
        process.started.connect(lambda:self.started_processes.append(int(process.processId())))
        process.start()

    def process_error(self,process,error):
        if process!=self.process: return
        self.note(process.errorString())
        if error==QProcess.FailedToStart:
            self.process=None; process.deleteLater()
            self.finish(dict(status='error',reason='worker_start_failed',detail=process.errorString()))

    def read_output(self,process,decoder):
        try:
            for message in decoder.feed(bytes(process.readAllStandardOutput())): self.message(message)
        except Exception as error:
            self.note('工作进程消息异常：'+str(error)); self.last_report=dict(status='error',reason='invalid_worker_message')
            process.kill()

    def message(self,message):
        kind=message['type']
        if kind=='status': self.state.setText(message['message']); self.note(message['message'])
        elif kind=='plan':
            self.pending_plan=message
            self.raw.setPlainText(json.dumps({k:v for k,v in message.items() if k!='type'},ensure_ascii=False,indent=2))
            if message['accepted']:
                names=dict(SKILLS); plan=message['guard']['response']['plan']
                text=' → '.join(names[s] for s in plan['skills']) or '停止'
                self.plan_label.setText('技能计划：'+text); self.note('计划通过：'+text)
            else: self.note('拒绝：'+message['guard']['reason'])
        elif kind=='ready':
            self.bounds=message['bounds']; self.total_steps=message['total_steps']
            self.state.setText('仿真中'); self.view_state.setText('MuJoCo · 实时')
            self.note(message['gl']['renderer']+' / OpenGL '+message['gl']['version'])
            self.send_camera()
        elif kind=='stage': self.note('阶段：'+dict(SKILLS).get(message['skill'],message['skill']))
        elif kind=='frame':
            image=QImage.fromData(base64.b64decode(message['jpeg']), 'JPG')
            if image.isNull(): raise ValueError('Invalid frame image')
            self.frame.set_image(image); self.frames+=1
            row=message['metrics']; h=row['bottom_m']*1000; d=row['target_distance_m']*1000
            self.values['height'].setText('%.1f mm'%h); self.values['distance'].setText('%.1f mm'%d)
            self.values['penetration'].setText('%.3f mm'%(row['scene_penetration_m']*1000))
            self.values['penetration'].setStyleSheet('color: '+('#ad252b' if row['scene_penetration_m']>.001 else '#087f68'))
            self.values['time'].setText('%.2f s'%message['sim_time_s'])
            for i,key in enumerate(('th','ff','mf','rf','lf')): self.force_table.item(i,1).setText('%.3f'%row[key+'_force_n'])
            self.joint.setText('关节越限 %.5f rad'%row['joint_violation_rad'])
            self.action.setText('最大动作幅值 %.3f'%message['action_abs_max'])
            self.peak_m=max(self.peak_m,message.get('max_penetration_m',row['scene_penetration_m']))
            self.peak.setText('峰值穿透 %.3f mm'%(self.peak_m*1000))
            for skill,bounds in self.bounds.items():
                value=round(100*(message['step']-bounds[0])/max(1,bounds[1]-bounds[0]))
                self.phase_bars[skill].setValue(max(0,min(100,value)))
                self.phase_labels[skill].setStyleSheet('color: '+('#087f68' if skill==message['skill'] else '#53606b'))
            self.trace.add(h,d)
            fps=self.frames/max(.01,message['elapsed_s'])
            self.statusBar().showMessage('步数 %d / %d  |  %.1f 帧/秒  |  %s'%(message['step'],self.total_steps,fps,self.output.name))
            self.frame_received.emit(message)
        elif kind=='result': self.last_report=message['report']
        elif kind=='error':
            self.last_report=dict(status='error',reason=message.get('error_type','worker_error'),detail=message['message'])
            self.note(message['message'])

    def worker_finished(self,process,kind,code,status):
        if process!=self.process: process.deleteLater(); return
        self.process=None; process.deleteLater()
        if kind=='plan' and code==0 and self.pending_plan and self.pending_plan['accepted'] and not self.cancelled:
            if self.want_execute:
                try:
                    self.state.setText('初始化仿真')
                    self.start_worker('physics',LEGACY_PYTHON,[str(ROOT/'scripts/126_stream_simulation.py'),
                        '--root',str(self.storage),'--output',str(self.output)],physics_environment())
                    return
                except Exception as error:
                    self.finish(dict(status='error',reason='render_environment',detail=str(error))); return
            self.finish(dict(status='preview',reason='plan_accepted')); return
        if self.last_report:
            report=self.last_report
        elif self.cancelled: report=dict(status='stopped',reason='cancelled',worker_exit=code)
        elif kind=='plan' and code==2 and self.pending_plan:
            report=dict(status='rejected',reason=self.pending_plan['guard']['reason'],steps=0,simulation_created=False)
        else: report=dict(status='error',reason='worker_exited_without_result',worker_exit=code)
        self.finish(report)

    def finish(self,report):
        self.last_report=report; self.set_busy(False)
        labels={'success':'已完成','stopped':'已停止','rejected':'已拒绝','preview':'计划已通过','error':'错误'}
        self.state.setText(labels.get(report['status'],report['status']))
        self.view_state.setText('MuJoCo · '+('最终状态' if self.frames else '未执行'))
        if not self.frames:
            self.frame.placeholder=self.state.text(); self.frame.update()
        self.statusBar().showMessage('%s  |  步数 %d  |  %s'%(self.state.text(),report.get('steps',0),self.output.name))
        self.note(report.get('reason',report['status']))
        if self.output and self.output.exists():
            try:
                (self.output/'desktop-result.json').write_text(json.dumps(dict(report=report,frames_received=self.frames,
                    instruction=self.instruction.toPlainText(),scene=self.scene.currentData()),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            except OSError as error: self.note('记录写入失败：'+str(error))
        self.run_finished.emit(report)
        if self.closing: QTimer.singleShot(0,self.close)

    def stop(self):
        if not self.busy or self.process is None: return
        self.cancelled=True; self.stop_button.setEnabled(False); self.state.setText('正在停止')
        process=self.process
        if self.kind=='physics': process.write(b'{"command":"stop"}\n')
        else: process.terminate()
        QTimer.singleShot(5000,lambda:self.kill_if_current(process))

    def kill_if_current(self,process):
        if self.process is process and process.state()!=QProcess.NotRunning:
            self.note('工作进程未响应，终止进程'); process.kill()

    def send_camera(self,*unused):
        if self.kind=='physics' and self.process is not None:
            packet=dict(command='camera',preset=self.camera.currentData(),zoom=self.zoom.value())
            self.process.write((json.dumps(packet)+'\n').encode())

    def open_folder(self):
        path=self.output if self.output and self.output.exists() else self.storage/'language/desktop_runs'
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def save_screenshot(self):
        target=(self.output/'desktop.png') if self.output and self.output.exists() else Path.home()/'grasp-desktop.png'
        name,_=QFileDialog.getSaveFileName(self,'保存截图',str(target),'PNG (*.png)')
        if name and not self.grab().save(name): self.note('截图保存失败')

    def closeEvent(self,event):
        if self.busy:
            self.closing=True; self.stop(); event.ignore()
        else: event.accept()
