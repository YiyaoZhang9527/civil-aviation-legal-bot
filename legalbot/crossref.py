"""法条交叉引用解析与补全。

从证据文本中识别对其他法条的引用，自动读取被引用条文内容。
只做后处理，不参与检索/排序。

覆盖模式：
  《民用航空法》第三十八条      — 跨法引用
  本法第四十六条                — 法内自引
  民用航空安全管理规定第八条    — 无书名号跨法引用
"""

from __future__ import annotations

import re

from .retrieval import IndexRepository, read_law_node
from .utils import chinese_to_int

_CN_NUM = r"[一二三四五六七八九十百零\d]+"
_ARTICLE_NUM_RE = rf"第({_CN_NUM})条"

_KNOWN_LAWS = [
    "关于修改定期国际航空运输管理规定、民用航空国内航线经营许可规定和公共航空运输企业经营许可规定的决定",
    "外国航空运输企业在中国境内指定的销售代理直接进入和使用外国计算机订座系统许可管理暂行规定",
    "国内投资民航业需要特别管理的公共航空运输企业、枢纽机场和战略机场名单",
    "国务院、中央军委关于重新颁发关于保护机场净空的规定的通知",
    "国务院、中央军委关于使用飞机进行人工降水问题的通知",
    "民用航空空中交通通信导航监视设备使用许可管理办法",
    "民用航空飞行标准委任代表和委任单位代表管理规定",
    "运输机场专业工程建设质量和安全生产监督管理规定",
    "中国民用航空计量技术委任代表和委任单位代表规定",
    "特殊商业和私用大型航空器运营人运行合格审定规则",
    "小型商业运输和空中游览运营人运行合格审定规则",
    "中华人民共和国民用航空器权利登记条例实施办法",
    "大型飞机公共航空运输承运人运行合格审定规则",
    "民用航空材料、零部件和机载设备技术标准规定",
    "出境入境航空器载运人员信息预报预检实施办法",
    "民用航空适航委任代表和委任单位代表管理规定",
    "民用航空通信导航监视设备飞行校验管理规则",
    "民用航空空中交通管理运行单位安全管理规则",
    "外国航空运输企业常驻代表机构审批管理办法",
    "民用航空器飞行事故应急反应和家属援助规定",
    "公共航空旅客运输飞行中安全保卫工作规则",
    "民用航空气象探测设施及探测环境管理办法",
    "涡轮发动机飞机燃油排泄和排气排出物规定",
    "民用机场飞行程序和运行最低标准管理规定",
    "外国公共航空运输承运人运行合格审定规则",
    "民用机场和民用航空器内禁止吸烟的规定",
    "民用运输机场突发事件应急救援管理规则",
    "民用机场航空器活动区道路交通安全管理",
    "中国民用航空部门计量检定规程管理办法",
    "中国民用航空总局关于废止部分民用航空",
    "外籍飞行人员体检合格证认可证书样式",
    "民用航空导航设备开放与运行管理规定",
    "民用航空空中交通管制员执照管理规则",
    "民用航空器维修培训机构合格审定规则",
    "民用无人驾驶航空器运行安全管理规则",
    "中国民用航空国内航线经营许可规定",
    "民用航空产品和零部件合格审定规定",
    "国务院关于保障民用航空安全的通告",
    "民用航空飞行签派员执照和训练机构",
    "民用航空空中交通管制培训管理规则",
    "民用航空器飞行机械员合格审定规则",
    "民用航空器驾驶员学校合格审定规则",
    "公共航空运输企业航空安全保卫规则",
    "民用航空运输机场航空安全保卫规则",
    "外国航空运输企业航线经营许可规定",
    "中国民用航空总局规章制定程序规定",
    "航空器型号和适航合格审定噪声规定",
    "民用航空器维修单位合格审定规则",
    "关于修订和废止部分民用航空规章",
    "民用航空人员体检合格证管理规则",
    "民用航空器维修人员执照管理规则",
    "民用航空通信导航监视工作规则",
    "民用航空气象人员执照管理规则",
    "民用航空预先飞行计划管理办法",
    "民航企业安全保障财务考核办法",
    "民用航空电信人员执照管理规则",
    "公共航空运输企业经营许可规定",
    "平行跑道同时仪表运行管理规定",
    "公共航空运输旅客服务管理规定",
    "中国民用航空监察员管理规定",
    "外国民用航空器飞行管理规则",
    "民用航空用化学产品适航规定",
    "中国民用航空无线电管理规定",
    "民用航空器事件技术调查规定",
    "民用航空人员体检合格证样式",
    "民用航空情报员执照管理规则",
    "民用航空危险品运输管理规定",
    "外国航空运输企业不定期飞行",
    "飞行训练中心合格审定规则",
    "空勤人员和空中交通管制员",
    "民用航空安全信息管理规定",
    "民用航空空中交通管理规则",
    "中国民用航空应急管理规定",
    "定期国际航空运输管理规定",
    "运输类旋翼航空器适航规定",
    "民航行政机关行政赔偿办法",
    "民用航空情报培训管理规则",
    "民用航空行政处罚实施办法",
    "正常类旋翼航空器适航规定",
    "中国民用航空气象工作规则",
    "中华人民共和国民用航空法",
    "民用航空行政许可工作规则",
    "运输机场运行安全管理规定",
    "民用航空财经信息管理办法",
    "民用航空货物运输管理规定",
    "国际航空运输价格管理规定",
    "民用机场专用设备管理规定",
    "民用航空行政检查工作规则",
    "民用航空器国籍登记规定",
    "民用航空器适航指令规定",
    "航空安全员合格审定规则",
    "国内投资民用航空业规定",
    "民用航空标准化管理规定",
    "民用航空计量管理规定",
    "民航总局行政复议办法",
    "民用航空使用空域办法",
    "民用机场建设管理设定",
    "通用航空安全保卫规则",
    "飞行模拟训练设备管理",
    "运输机场使用许可规定",
    "民用航空油料适航规定",
    "载人自由气球适航规定",
    "民用航空统计管理规定",
    "民用航空情报工作规则",
    "民用航空安全管理规定",
    "民用航空安全检查规则",
    "民用机场建设管理规定",
    "航空发动机适航规定",
    "正常类飞机适航规定",
    "一般运行和飞行规则",
    "运输类飞机适航标准",
    "航班正常管理规定",
    "民用航空器驾驶员",
    "通用机场管理规定",
    "螺旋桨适航规定",
]
_KNOWN_LAWS.sort(key=len, reverse=True)

_CROSS_BOOK_RE = re.compile(rf"《([\u4e00-\u9fff（）()]+?)》\s*{_ARTICLE_NUM_RE}")
_SELF_REF_RE = re.compile(rf"(?:本法|本条例|本规定|本办法)\s*{_ARTICLE_NUM_RE}")
_NO_BOOK_RE = re.compile(rf'({"|".join(re.escape(n) for n in _KNOWN_LAWS)})\s*{_ARTICLE_NUM_RE}')


def _article_to_node_id(article_text: str) -> str | None:
    """「第三十八条」→ 'article:38'"""
    m = re.search(_ARTICLE_NUM_RE, article_text)
    if not m:
        return None
    num = chinese_to_int(m.group(1))
    if num is None:
        return None
    return f"article:{num}"


def extract_references(text: str) -> list[dict]:
    """从文本中提取法律条文引用，返回 [{'raw': ..., 'law_name': ..., 'article': ..., 'self_ref': bool}]"""
    seen: set[str] = set()
    refs: list[dict] = []

    # 《XX法》第X条
    for m in _CROSS_BOOK_RE.finditer(text):
        article_full = f"第{m.group(2)}条"
        key = f"{m.group(1)}:{article_full}"
        if key not in seen:
            seen.add(key)
            refs.append({"raw": m.group(0), "law_name": m.group(1), "article": article_full, "self_ref": False})

    # 本法第X条
    for m in _SELF_REF_RE.finditer(text):
        article_full = f"第{m.group(1)}条"
        key = f"_self:{article_full}"
        if key not in seen:
            seen.add(key)
            refs.append({"raw": m.group(0), "law_name": None, "article": article_full, "self_ref": True})

    # 民用航空安全管理规定第X条（无书名号）
    for m in _NO_BOOK_RE.finditer(text):
        article_full = f"第{m.group(2)}条"
        key = f"{m.group(1)}:{article_full}"
        if key not in seen:
            seen.add(key)
            refs.append({"raw": m.group(0), "law_name": m.group(1), "article": article_full, "self_ref": False})

    return refs


def resolve_references(
    refs: list[dict],
    current_law_id: str,
    max_items: int = 5,
) -> list[dict]:
    """解析引用列表，读取被引用条文。

    返回 [{'raw': ..., 'law_title': ..., 'article': ..., 'text': ...}]
    """
    resolved: list[dict] = []
    for ref in refs:
        if len(resolved) >= max_items:
            break

        article_text = ref["article"]
        node_id = _article_to_node_id(article_text)
        if node_id is None:
            continue

        # 确定目标 law_id
        if ref["self_ref"]:
            doc = IndexRepository.find_document(current_law_id)
            if doc is None:
                continue
            target_law_id = doc.law_id
        elif ref["law_name"]:
            doc = IndexRepository.find_document(ref["law_name"])
            if doc is None:
                continue
            target_law_id = doc.law_id
        else:
            continue

        result = read_law_node(target_law_id, node_id, include_context=False)
        if result.get("found") and len(result.get("text", "")) > 10:
            resolved.append({
                "raw": ref["raw"],
                "law_id": target_law_id,
                "law_title": result.get("law_title", ""),
                "article": result.get("article", ""),
                "node_id": node_id,
                "text": result["text"],
            })

    return resolved


def expand_evidence_references(
    evidence_list: list,
    max_items: int = 5,
) -> list[dict]:
    """对证据列表做交叉引用补全。返回引用条文的列表，不修改原 evidence。"""
    all_refs: list[dict] = []
    seen_keys: set[str] = set()

    for ev in evidence_list:
        text = getattr(ev, "text", "")
        law_id = getattr(ev, "law_id", "")
        if not text or not law_id:
            continue
        refs = extract_references(text)
        resolved = resolve_references(refs, law_id, max_items=max_items - len(all_refs))
        for r in resolved:
            key = f"{r['law_title']}:{r['article']}"
            if key not in seen_keys:
                seen_keys.add(key)
                all_refs.append(r)
        if len(all_refs) >= max_items:
            break

    return all_refs
