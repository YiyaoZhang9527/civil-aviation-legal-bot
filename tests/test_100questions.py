"""100题完整测试：输出CSV格式结果。"""

import csv
import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import legalbot.config as cfg
from legalbot.agents import LegalOrchestrator
from legalbot.types import Evidence

# ── 100道测试题 ──
# 覆盖：CCAR-121(13题)、CCAR-91(5题)、CCAR-135(3题)、CCAR-92/无人机(5题)、
#       机场管理(6题)、旅客服务(3题)、航空安全(5题)、适航管理(7题)、
#       航空器维修(3题)、空管(3题)、通信导航监视(3题)、航空气象(2题)、
#       航空情报(2题)、人员资质(4题)、民用航空法(3题)、事故调查(2题)、
#       应急管理(2题)、通用航空(5题)、旅客服务与消费者权益(3题)、
#       航空器权利登记(2题)、公共航空运输企业经营(2题)、
#       飞行训练中心(1题)、飞行模拟训练设备管理(1题)
QUESTIONS = [
    # ─── 原有30题 (Q01-Q30) ───
    # CCAR-121 大型航空器运输
    {"id": "Q01", "category": "CCAR-121", "question": "不在《运行规范》的机场是否可以去备降"},
    {"id": "Q02", "category": "CCAR-121", "question": "飞机起飞前需要满足哪些燃油量要求"},
    {"id": "Q03", "category": "CCAR-121", "question": "航空公司在什么情况下可以低于最低天气标准运行"},
    {"id": "Q04", "category": "CCAR-121", "question": "运行规范中需要包含哪些内容"},
    {"id": "Q05", "category": "CCAR-121", "question": "飞行机组的值勤期限制是多少"},
    {"id": "Q06", "category": "CCAR-121", "question": "飞机延程运行EDTO需要满足什么条件"},
    {"id": "Q07", "category": "CCAR-121", "question": "签派员放行飞机时需要检查哪些事项"},
    {"id": "Q08", "category": "CCAR-121", "question": "飞机在结冰条件下运行有什么要求"},
    # CCAR-91 一般运行
    {"id": "Q09", "category": "CCAR-91", "question": "机长在紧急情况下有哪些权力"},
    {"id": "Q10", "category": "CCAR-91", "question": "民用航空器在什么情况下可以进行特技飞行"},
    # CCAR-135 小型运输
    {"id": "Q11", "category": "CCAR-135", "question": "小型航空器商业运输的备降机场要求是什么"},
    {"id": "Q12", "category": "CCAR-135", "question": "CCAR-135部运营人的飞行员资质要求是什么"},
    # CCAR-92 无人机
    {"id": "Q13", "category": "CCAR-92", "question": "无人机在什么情况下需要申请空域许可"},
    {"id": "Q14", "category": "CCAR-92", "question": "无人机飞行的安全距离要求是什么"},
    # 机场管理
    {"id": "Q15", "category": "机场管理", "question": "机场使用许可证的申请条件是什么"},
    {"id": "Q16", "category": "机场管理", "question": "机场运行安全管理中谁来负责飞行区安全"},
    {"id": "Q17", "category": "机场管理", "question": "航班备降时机场运营人有什么义务"},
    # 旅客服务
    {"id": "Q18", "category": "旅客服务", "question": "航班延误超过多长时间航空公司需要为旅客提供餐饮"},
    {"id": "Q19", "category": "旅客服务", "question": "航空公司拒载旅客的合法理由有哪些"},
    {"id": "Q20", "category": "旅客服务", "question": "旅客行李丢失后航空公司如何赔偿"},
    # 航空安全
    {"id": "Q21", "category": "航空安全", "question": "航空安全检查中哪些物品禁止带上飞机"},
    {"id": "Q22", "category": "航空安全", "question": "民用航空安全信息报告的时限要求是什么"},
    {"id": "Q23", "category": "航空安全", "question": "航空器发生事故后谁来负责调查"},
    # 适航管理
    {"id": "Q24", "category": "适航管理", "question": "民用航空器适航指令是做什么的"},
    {"id": "Q25", "category": "适航管理", "question": "航空器维修单位需要什么资质"},
    # 空中交通管理
    {"id": "Q26", "category": "空管", "question": "空中交通管制服务由哪些单位提供"},
    {"id": "Q27", "category": "空管", "question": "飞行程序设计和运行最低标准由谁审批"},
    # 民用航空法
    {"id": "Q28", "category": "民用航空法", "question": "中华人民共和国对领空享有什么权利"},
    {"id": "Q29", "category": "民用航空法", "question": "民用航空器所有权的取得和转让有什么要求"},
    # 人员资质
    {"id": "Q30", "category": "人员资质", "question": "飞行员执照的种类有哪些"},

    # ─── 新增题目 (Q31-Q100) ───
    # CCAR-121 续
    {"id": "Q31", "category": "CCAR-121", "question": "飞机上有个设备坏了但不是关键设备，航空公司说按最低设备清单MEL可以飞，这个MEL是什么？哪些设备绝对不能放到MEL里面带故障飞行？"},
    {"id": "Q32", "category": "CCAR-121", "question": "我们公司要扩大经营范围，想修改运行规范增加新的运行种类，该怎么向民航局申请？如果民航局拒绝了怎么办？"},
    {"id": "Q33", "category": "CCAR-121", "question": "客舱乘务员最多能连续值勤多少小时？如果人手不够多加几个乘务员，值勤上限会变吗？"},
    {"id": "Q34", "category": "CCAR-121", "question": "飞机每次起飞前的装载舱单上面都写了些什么？机长需要在起飞前亲自确认舱单吗？"},
    {"id": "Q35", "category": "CCAR-121", "question": "双发飞机要飞跨洋航线做延程运行EDTO，备降机场的救援和消防服务有什么具体要求？超过180分钟和不超过180分钟要求一样吗？"},
    # CCAR-91 续
    {"id": "Q36", "category": "CCAR-91", "question": "开小飞机按目视飞行规则出去飞一圈，油箱里最少要带多少油才合规"},
    {"id": "Q37", "category": "CCAR-91", "question": "飞机在不同高度的巡航飞行高度层是怎么划分的，跟飞的方向有关系吗"},
    {"id": "Q38", "category": "CCAR-91", "question": "想从飞机上跳伞需要走哪些审批流程，在人口密集区上空能不能跳"},
    # CCAR-135 续
    {"id": "Q39", "category": "CCAR-135", "question": "我想开一家空中游览公司，只做短途游览飞行，起降都在同一个地方，飞行距离不超过40公里，需要满足CCAR-135的哪些要求"},
    {"id": "Q40", "category": "CCAR-136", "question": "我朋友有一个私人飞机想请人代管，这种航空器代管人需要具备什么条件，对驾驶员的飞行时间有什么限制"},
    {"id": "Q41", "category": "CCAR-135", "question": "CCAR-135部的运营人什么情况下需要建立安全管理体系，这个体系要具备哪些功能"},
    # CCAR-92 无人机续
    {"id": "Q42", "category": "CCAR-92", "question": "我买了一台大疆无人机，自己玩的话需要考执照吗？什么样的无人机不需要操控员执照"},
    {"id": "Q43", "category": "CCAR-92", "question": "我们公司想做无人机物流配送，需要办理什么运营资质？什么情况下不需要办运营合格证"},
    {"id": "Q44", "category": "CCAR-92", "question": "无人机操控员执照有效期是多久？过期了还能用吗，怎么重新办理"},
    {"id": "Q45", "category": "CCAR-92", "question": "无人机运营合格证的有效期是多长？如果中间停运了好几个月还能继续飞吗"},
    {"id": "Q46", "category": "CCAR-92", "question": "无人机发生事故后操控员执照会怎么处理？无证驾驶无人机飞要罚多少钱"},
    # 通用航空
    {"id": "Q47", "category": "通用航空经营许可管理规定", "question": "我想开一家通航公司做空中游览和跳伞服务，需要买几架飞机才能申请经营许可？"},
    {"id": "Q48", "category": "通用机场管理规定", "question": "我们公司想建一个通用机场，A1级和A2级机场有什么区别？A类和B类机场在审批上有什么不同？"},
    {"id": "Q49", "category": "通用航空安全保卫规则", "question": "通航公司用直升机搞空中游览，起飞前需要对乘客做什么安保检查？乘客信息要保存多久？"},
    {"id": "Q50", "category": "通用航空经营许可管理规定", "question": "通航公司拿到经营许可证后，如果不按规定报送年度报告或者经营活动信息，会受什么处罚？"},
    {"id": "Q51", "category": "通用航空经营与机场管理", "question": "通航公司在运输机场开展不定期包机载客业务，在安保方面需要做哪些准备？如果公司没有核实乘客身份就起飞了，会有什么法律后果？"},
    # 机场管理续
    {"id": "Q52", "category": "机场管理", "question": "机场发生突发事件的时候，消防指挥官和医疗指挥官分别穿什么衣服戴什么头盔来区分身份"},
    {"id": "Q53", "category": "机场管理", "question": "机场净空保护巡查有什么频率要求，哪些区域需要每天查两次"},
    {"id": "Q54", "category": "机场管理", "question": "在机场航空器活动区开车，如果碰撞了航空器或者导致航空器复飞，驾照会被扣多少分，有什么后果"},
    {"id": "Q55", "category": "机场管理", "question": "机场要搞平行跑道同时仪表运行需要怎么审批，试验运行要多久"},
    {"id": "Q56", "category": "机场管理", "question": "机场应急救援预案和飞行程序分别需要向哪个部门报备或审批，发现障碍物影响飞行时机场该怎么处理"},
    # 旅客服务与消费者权益
    {"id": "Q57", "category": "旅客服务与消费者权益", "question": "我买机票被超售了，航空公司可以不先征求志愿者就直接按优先规则把我赶下飞机吗"},
    {"id": "Q58", "category": "旅客服务与消费者权益", "question": "飞机上有人闹事打空姐，机长有权怎么处理？普通乘客可以帮忙吗"},
    {"id": "Q59", "category": "旅客服务与消费者权益", "question": "我带70多岁的老母亲坐飞机遇到航班取消，航空公司是不是应该优先给我们安排"},
    # 航空安全续
    {"id": "Q60", "category": "航空安全", "question": "民航企业的安全管理体系需要包含哪些组成部分，是不是每个单位都必须建立这个体系"},
    {"id": "Q61", "category": "航空安全", "question": "机场安检用的X射线安检仪操作员连续工作多长时间必须休息，国家对安检设备有什么管理要求"},
    {"id": "Q62", "category": "航空安全", "question": "如果机场收到飞机上有炸弹威胁或劫机威胁的消息，机场方面应当按照什么程序处置，航空公司又该怎么做"},
    {"id": "Q63", "category": "航空安全", "question": "机场的航空安保方案什么情况下需要修订，修订后要提前多久报给管理局备案"},
    {"id": "Q64", "category": "航空安全", "question": "民航企业安全保障财务考核的评分标准和结果怎么划分，考核不合格会怎么样"},
    # 事故调查
    {"id": "Q65", "category": "事故调查", "question": "民航事故调查的目的是什么，调查结论能不能用来追究责任"},
    {"id": "Q66", "category": "事故调查", "question": "航空器事故调查报告一般多长时间内必须向社会公布"},
    # 应急管理
    {"id": "Q67", "category": "应急管理", "question": "航空公司发生飞行事故后，对遇难者家属应当提供哪些帮助，费用谁出"},
    {"id": "Q68", "category": "应急管理", "question": "民航单位和机场在突发事件应急方面分别需要做哪些准备工作，如果不做会面临什么处罚"},
    # 适航管理续
    {"id": "Q69", "category": "适航管理", "question": "如果我想在国内生产飞机零部件卖给别人装在飞机上，需要办什么证件？和型号合格证有什么区别"},
    {"id": "Q70", "category": "适航管理", "question": "什么是适航委任代表？个人可以申请当适航委任代表吗，需要满足什么条件"},
    {"id": "Q71", "category": "适航管理", "question": "飞机的适航证一直有效吗？什么情况下适航证会失效"},
    {"id": "Q72", "category": "适航管理", "question": "如果飞机还没有拿到适航证但需要飞去其他地方维修，可以飞吗？有什么限制条件"},
    {"id": "Q73", "category": "适航管理", "question": "航空公司使用的航空燃油也需要通过适航审批吗？油料供应商需要什么资质"},
    {"id": "Q74", "category": "适航管理", "question": "飞机飞过之后发动机排出的尾气也有排放标准吗？如果不达标还能继续飞吗？"},
    # 航空器维修
    {"id": "Q75", "category": "航空器维修", "question": "我想考一个飞机维修执照，请问需要满足什么条件"},
    {"id": "Q76", "category": "航空器维修", "question": "维修培训机构给学生上课有什么人数限制，学员缺课多了还能参加考试吗"},
    {"id": "Q77", "category": "航空器维修", "question": "我拿到维修执照以后想给大型客机做维修放行，除了执照本身还需要什么资格，资格到期了怎么续"},
    # 空管续
    {"id": "Q78", "category": "空管", "question": "我国的高空管制空域属于哪类空域，在里面飞行的飞机只能按什么规则飞行"},
    {"id": "Q79", "category": "空管", "question": "空中交通管制员在岗位上最多能连续工作多久，每周最长能值多少小时班"},
    {"id": "Q80", "category": "空管", "question": "空管运行单位在什么情况下必须进行安全评估，如果一架飞机与另一架飞机之间的间隔小于了规定的最小飞行间隔，这属于什么级别的事件需要怎么处理"},
    # 通信导航监视
    {"id": "Q81", "category": "通信导航监视", "question": "机场的仪表着陆系统多久需要做一次飞行校验，校验周期是怎么算的"},
    {"id": "Q82", "category": "通信导航监视", "question": "民航无线电频率受到干扰了应该怎么处理，谁来负责排除干扰"},
    {"id": "Q83", "category": "通信导航监视", "question": "导航设备停机超过三个月后想重新投入使用，需要经过哪些程序和审批"},
    # 航空气象
    {"id": "Q84", "category": "航空气象", "question": "机场的自动气象观测设备应该装在跑道的什么位置？测量跑道视程的设备有严格的安装距离要求吗？"},
    {"id": "Q85", "category": "航空气象", "question": "我想当民航气象预报员，需要什么学历才能申请执照？和气象观测员的要求一样吗？"},
    # 航空情报
    {"id": "Q86", "category": "航空情报", "question": "机场下雪后跑道湿滑，雪情通告最长能管多久，超过时间会怎么样"},
    {"id": "Q87", "category": "航空情报", "question": "想当航空情报员需要什么学历和执照，见习期至少多久才能独立上岗"},
    # 人员资质续
    {"id": "Q88", "category": "人员资质", "question": "航空安全员申请执照需要满足哪些条件"},
    {"id": "Q89", "category": "人员资质", "question": "飞行签派员执照申请人需要什么学历和经历要求，执照考试多少分合格"},
    {"id": "Q90", "category": "人员资质", "question": "不同岗位的民航人员分别需要持有哪种等级的体检合格证，有效期多长"},
    {"id": "Q91", "category": "人员资质", "question": "飞行员多久需要做一次定期检查，检查包含哪些内容"},
    # 飞行训练
    {"id": "Q92", "category": "飞行训练中心合格审定规则", "question": "我想开一家飞行训练中心给航空公司飞行员做培训，需要拿到什么资质，证书有效期多久"},
    {"id": "Q93", "category": "飞行模拟训练设备管理", "question": "我们公司有一台飞行模拟机，合格证快到期了，续期需要提前多久申请，如果没有建立质量管理系统证书有效期有什么区别"},
    # 民用航空法续
    {"id": "Q94", "category": "民用航空法", "question": "飞机上掉下来的东西砸坏了地面上的人或者房子，航空公司要赔偿吗？飞机正常飞过时的噪音震动能要求赔偿吗"},
    {"id": "Q95", "category": "民用航空法", "question": "一架飞机被用来做担保借款，如果飞机上的优先权（比如救援费）和抵押权同时存在，谁先拿到钱"},
    {"id": "Q96", "category": "民用航空法", "question": "想成立一家航空公司，需要满足什么条件，注册资本最低要多少钱，审批要多久"},
    # 航空器权利登记
    {"id": "Q97", "category": "航空器权利登记", "question": "我从国外租赁了一架飞机，想在中国登记国籍，需要满足什么条件？"},
    {"id": "Q98", "category": "航空器权利登记", "question": "我买了一架飞机想做抵押贷款，航空器抵押权登记怎么办，生效时间怎么算？"},
    # 公共航空运输企业经营
    {"id": "Q99", "category": "公共航空运输企业经营", "question": "开办一家航空公司需要多少架飞机"},
    {"id": "Q100", "category": "公共航空运输企业经营", "question": "国内航线经营许可核准管理和登记管理有什么区别，哪些航线需要核准"},
]

NUM_QUESTIONS = len(QUESTIONS)


def extract_citation_pairs(text: str) -> list[str]:
    pairs = re.findall(r'《([^》]+)》[^。；\n]*?(第[一二三四五六七八九十百千\d]+条)', text)
    return [f"《{n}》{a}" for n, a in pairs]


def extract_article_numbers(evidence: list[Evidence]) -> list[str]:
    return [ev.article for ev in evidence if ev.article]


def run_question(orch: LegalOrchestrator, q: dict) -> dict:
    start = time.time()
    result = orch.answer(q["question"])
    elapsed = time.time() - start

    ev_articles = extract_article_numbers(result.evidence or [])
    cit_statuses = [(c.node_id, c.status, round(c.confidence, 2)) for c in (result.citations or [])]
    text_refs = extract_citation_pairs(result.answer)

    # 统计
    supported = sum(1 for _, s, _ in cit_statuses if s == "supported")
    partial = sum(1 for _, s, _ in cit_statuses if s == "partial")
    unsupported = sum(1 for _, s, _ in cit_statuses if s == "unsupported")

    # 结论提取（取答案第一句或前100字）
    answer_clean = result.answer.strip()
    conclusion = answer_clean[:150].replace("\n", " ")

    return {
        "question_id": q["id"],
        "category": q["category"],
        "question": q["question"],
        "elapsed_sec": round(elapsed, 1),
        "answer_len": len(result.answer),
        "evidence_count": len(result.evidence or []),
        "evidence_articles": " | ".join(ev_articles),
        "citation_count": len(result.citations or []),
        "supported": supported,
        "partial": partial,
        "unsupported": unsupported,
        "supported_rate": f"{supported}/{len(cit_statuses)}" if cit_statuses else "0/0",
        "text_refs": " | ".join(text_refs) if text_refs else "",
        "reflexion_iterations": result.reflexion_iterations,
        "conclusion_preview": conclusion,
        "answer_full": result.answer,
        "config": f"A1={cfg.QUERY_GATE_ENABLED}/C1={cfg.CROSS_ENCODER_CITATION}/B3={cfg.CONFIDENCE_CUTOFF_ENABLED}/E1={cfg.LEXICAL_REFLEXION_ENABLED}/reranker_min={cfg.RERANKER_MIN_SCORE}",
    }


CSV_COLUMNS = [
    # 基础信息
    "question_id",       # 题号 Q01-Q100
    "category",          # 法规分类
    "question",          # 问题
    # 性能指标
    "elapsed_sec",       # 耗时(秒)
    "answer_len",        # 答案字数
    # Agent执行信息
    "evidence_count",    # 证据条数
    "reflexion_iterations",  # 自检迭代次数
    # 检索结果
    "evidence_articles", # 检索到的法条列表
    "text_refs",         # 答案中的文本引用
    # 引用校验
    "citation_count",    # 校验条数
    "supported",         # supported数
    "partial",           # partial数
    "unsupported",       # unsupported数
    "supported_rate",    # supported率
    # 答案
    "conclusion_preview", # 结论预览(前150字)
    "answer_full",        # 完整答案（不截断），csv 模块自动加引号包裹多行文本
    "config",             # 运行配置快照
]


def _run_questions_sequential(date_str):
    """原版顺序执行。"""
    csv_path = PROJECT_ROOT / "tests" / f"test100_{date_str}.csv"
    json_path = PROJECT_ROOT / "tests" / f"test100_{date_str}.json"
    orch = LegalOrchestrator(logger=None)
    all_results = []
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for i, q in enumerate(QUESTIONS):
            print(f"[{q['id']}] {q['question'][:40]}...", end=" ", flush=True)
            try:
                r = run_question(orch, q)
                writer.writerow(r)
                all_results.append(r)
                print(f"完成 {r['elapsed_sec']}s, {r['answer_len']}字, supported={r['supported_rate']}")
            except Exception as e:
                print(f"失败: {e}")
                error_row = {col: "" for col in CSV_COLUMNS}
                error_row["question_id"] = q["id"]
                error_row["category"] = q["category"]
                error_row["question"] = q["question"]
                error_row["conclusion_preview"] = f"ERROR: {e}"
                writer.writerow(error_row)
                all_results.append(error_row)
            time.sleep(1)
    return all_results, csv_path, json_path


def _worker_chunk(chunk):
    """Worker：处理一个 question 列表，返回结果列表。"""
    from legalbot.agents import LegalOrchestrator
    orch = LegalOrchestrator(logger=None)
    results = []
    for q in chunk:
        try:
            results.append(run_question(orch, q))
        except Exception as e:
            error_row = {col: "" for col in CSV_COLUMNS}
            error_row["question_id"] = q["id"]
            error_row["category"] = q.get("category", "")
            error_row["question"] = q["question"]
            error_row["conclusion_preview"] = f"ERROR: {e}"
            results.append(error_row)
        time.sleep(0.5)
    return results


def _run_questions_parallel(date_str, n_workers):
    """按题目并行：轮询切分 → mp.Pool → 合并。"""
    import multiprocessing as mp
    chunks = [[] for _ in range(n_workers)]
    for i, q in enumerate(QUESTIONS):
        chunks[i % n_workers].append(q)
    chunks = [c for c in chunks if c]

    print(f"切分: {n_workers} worker")
    for i, c in enumerate(chunks):
        print(f"  Worker {i}: {len(c)} 题 ({c[0]['id']} - {c[-1]['id']})")

    start = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        all_chunk_results = pool.map(_worker_chunk, chunks)
    elapsed = time.time() - start

    all_results = []
    for chunk_result in all_chunk_results:
        all_results.extend(chunk_result)
    all_results.sort(key=lambda x: x.get("question_id", ""))

    print(f"\n并行完成！{elapsed/60:.1f} 分钟")

    csv_path = PROJECT_ROOT / "tests" / f"test100_{date_str}.csv"
    json_path = PROJECT_ROOT / "tests" / f"test100_{date_str}.json"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)
    return all_results, csv_path, json_path


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1, help="并行 worker 数（1=串行，2=2x 加速，8GB GPU 推荐 2）")
    args = parser.parse_args()

    import datetime
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"{NUM_QUESTIONS}题完整测试 (workers={args.workers})")
    print(f"配置: {cfg.QUERY_GATE_ENABLED=}, {cfg.CROSS_ENCODER_CITATION=}, {cfg.RERANKER_MIN_SCORE=}")
    print(f"开始时间: {date_str}\n")

    if args.workers == 1:
        all_results, csv_path, json_path = _run_questions_sequential(date_str)
    else:
        all_results, csv_path, json_path = _run_questions_parallel(date_str, args.workers)

    # 保存完整JSON（含answer_full）
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "test_date": date_str,
            "config": all_results[0]["config"] if all_results else "",
            "results": all_results,
        }, f, ensure_ascii=False, indent=2, default=str)

    # 汇总统计
    print(f"\n{'='*60}")
    print("汇总统计")
    print(f"{'='*60}")
    ok = [r for r in all_results if not r.get("conclusion_preview", "").startswith("ERROR")]
    if ok:
        avg_time = sum(r["elapsed_sec"] for r in ok) / len(ok)
        avg_len = sum(r["answer_len"] for r in ok) / len(ok)
        total_supported = sum(r["supported"] for r in ok)
        total_citations = sum(r["citation_count"] for r in ok)
        print(f"完成: {len(ok)}/{NUM_QUESTIONS}")
        print(f"平均耗时: {avg_time:.1f}s")
        print(f"平均答案长度: {avg_len:.0f}字")
        print(f"整体supported率: {total_supported}/{total_citations} ({total_supported/max(total_citations,1)*100:.0f}%)")

        # 按分类统计
        by_cat = {}
        for r in ok:
            cat = r["category"]
            by_cat.setdefault(cat, []).append(r)
        print(f"\n按分类:")
        for cat, rows in sorted(by_cat.items()):
            s = sum(r["supported"] for r in rows)
            c = sum(r["citation_count"] for r in rows)
            t = sum(r["elapsed_sec"] for r in rows) / len(rows)
            print(f"  {cat:15s}: {len(rows)}题, supported={s}/{c}, 平均{t:.0f}s")

    print(f"\nCSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
