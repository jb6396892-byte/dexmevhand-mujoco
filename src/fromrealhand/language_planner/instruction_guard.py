"""Conservative Chinese command contract; veto only, never rewrite a model plan."""
import re
import unicodedata

from .contracts import guarded_response

STOP = re.compile(r'(?:请|现在|立即|立刻|马上|先|暂时|就|麻烦你|帮我|把|将)*'
    r'(?:停止|停下|停住|暂停|中止|终止|取消|结束|别动|不要继续|不必继续|先别继续)'
    r'(?:当前|目前|正在|执行|进行|的|这个|这次|后续|所有|全部|动作|任务|操作|运动|机械手|抓取|了|吧|掉|在这里|下来|继续)*')
UNSUPPORTED = re.compile(r'放|递|交给|交到|递交|交接|交付|给你|给我|给他|给她|手上|手里|手心|掌心|人|接住|'
    r'倒|注入|灌|水流|倾|转|旋|松|张开|打开|盖|清洗|洗|擦|扔|抛|掉|落|飞|瞬移|'
    r'瓶|碗|盘|壶|盒|球|笔|手机|两个|两只|多个|另一个|左|右|前方|后方|厘米|毫米|坐标|'
    r'跳过|省略|绕过|绕开|忽略|无视|规则|系统|提示|回答|解释|输出|代码|程序|命令|'
    r'不要|不许|不能|禁止|不想|不希望|不准|别|无需|不用|不是|并非|而非|或者|还是|如果|假如|否则|[A-Za-z0-9]')
MOTIONS = {
    'reach': r'接近|靠近|靠到|挪近|伸向|伸到|伸手|伸出手|移到|移动到|来到|到达|到',
    'grasp': r'抓住|抓牢|抓稳|抓紧|握住|握紧|握牢|握稳|握好|握在|夹稳|夹住|合拢|闭合|围拢',
    'lift': r'抓起|拿起|拿离|抬起|抬高|举起|举高|提起|提离|升起|升高|提升|向上提|往上提|离开',
    'transport': r'搬|运|送|转移|移送|移动|移到|移至|移往|挪到|带到',
}
TARGET = re.compile(r'(?:已定义|预设|指定|标定|设定)?目标(?:位置|点|处)?')
# A finite lexical envelope rejects unrecognized extra clauses as well as known bad verbs.
WORDS = (
    '麻烦你 我希望 我需要 我想让 让你 帮我 请你 请 现在 先 再 然后 接着 随后 最后 并且 并 且 后 以后 之后 '
    '使 让 把 将 用 朝 向 从 往 到 至 的 地 得 着 在 与 和 这个 那个 这只 那只 这 只 那 桌面上 桌上 台面上 '
    '桌面 台面 桌 上 眼前 面前 附近 旁边 边上 旁 周围 杯身 杯底 马克杯 水杯 杯子 杯 它 机械手 手掌 手指 手 '
    '牢牢 稳稳 轻轻 慢慢 稳定 仅仅 只需 只要 仅 只 就 好 即可 就行 为止 一下 吧 了 吗 来 去 '
    '伸过去 伸出 合上 围住 收拢 靠过去 移动 挪动 保持 原地 空中 悬空 高度 往上 向上 附近处 '
    '目标点处 目标位置处 已定义 预设 指定 标定 设定 目标位置 目标点 目标处 目标 '
).split()
WORDS += [part for pattern in MOTIONS.values() for part in pattern.split('|')]
LEXICON = re.compile('|'.join(re.escape(word) for word in sorted(set(WORDS), key=len, reverse=True)))


def instruction_contract(instruction):
    if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 120:
        return dict(allowed=False, reason='invalid_instruction')
    if any(unicodedata.category(c).startswith('C') for c in instruction):
        return dict(allowed=False, reason='hidden_control_character')
    text = unicodedata.normalize('NFKC', instruction)
    text = re.sub(r'[\s，。！？、,!.?：:；;]', '', text)
    if STOP.fullmatch(text):
        return dict(allowed=True, goal='stop', reason='explicit_stop')
    if UNSUPPORTED.search(text):
        return dict(allowed=False, reason='unsupported_request')
    if any(word in text for word in ('停止','取消','结束','暂停','中止','终止')):
        return dict(allowed=False, reason='mixed_stop_and_motion')
    if not re.search(r'杯子|水杯|马克杯|杯身|杯底', text):
        return dict(allowed=False, reason='missing_explicit_object')
    if LEXICON.sub('', text):
        return dict(allowed=False, reason='unrecognized_clause')
    matches = {goal: bool(re.search(pattern, text)) for goal, pattern in MOTIONS.items()}
    if TARGET.search(text):
        if re.search(r'把手|将手|手掌|机械手', text) and not (matches['grasp'] or matches['lift']):
            return dict(allowed=False, reason='ambiguous_moving_subject')
        if matches['transport']:
            return dict(allowed=True, goal='transport', reason='registered_target')
        return dict(allowed=False, reason='ambiguous_target_request')
    if re.search(r'搬|运|送|移送|带到', text):
        return dict(allowed=False, reason='missing_registered_target')
    if matches['lift']:
        return dict(allowed=True, goal='lift', reason='explicit_lift')
    if matches['grasp']:
        return dict(allowed=True, goal='grasp', reason='explicit_grasp')
    if matches['reach'] and ('手' in text or re.search(r'接近|靠近', text)):
        return dict(allowed=True, goal='reach', reason='explicit_reach')
    return dict(allowed=False, reason='ambiguous_request')


def semantic_guard(raw, instruction, scene, schema, evidence, threshold=.05, nominal=True):
    contract = instruction_contract(instruction)
    old = guarded_response(raw, scene, schema, evidence, threshold, nominal)
    if not old['accepted']:
        return dict(old, instruction_contract=contract)
    if not contract['allowed']:
        return dict(accepted=False, reason='instruction_not_admitted', instruction_contract=contract,
                    response=old['response'])
    if old['response']['plan']['goal'] != contract['goal']:
        return dict(accepted=False, reason='instruction_plan_mismatch', instruction_contract=contract,
                    response=old['response'])
    return dict(old, instruction_contract=contract)
