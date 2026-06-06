"""消融测试参数网格与探针题集定义。

可被 runner / worker / analyzer 共用，避免硬编码散落各处。
"""

# 30 道探针题（与 test_30questions.py 相同），覆盖各类法规
PROBE_QUESTIONS = [
    {"id": "Q01", "category": "CCAR-121", "question": "不在《运行规范》的机场是否可以去备降"},
    {"id": "Q02", "category": "CCAR-121", "question": "飞机起飞前需要满足哪些燃油量要求"},
    {"id": "Q03", "category": "CCAR-121", "question": "航空公司在什么情况下可以低于最低天气标准运行"},
    {"id": "Q04", "category": "CCAR-121", "question": "运行规范中需要包含哪些内容"},
    {"id": "Q05", "category": "CCAR-121", "question": "飞行机组的值勤期限制是多少"},
    {"id": "Q07", "category": "CCAR-121", "question": "签派员放行飞机时需要检查哪些事项"},
    {"id": "Q08", "category": "CCAR-121", "question": "飞机在结冰条件下运行有什么要求"},
    {"id": "Q09", "category": "CCAR-91", "question": "机长在紧急情况下有哪些权力"},
    {"id": "Q11", "category": "CCAR-135", "question": "小型航空器商业运输的备降机场要求是什么"},
    {"id": "Q12", "category": "CCAR-135", "question": "CCAR-135部运营人的飞行员资质要求是什么"},
    {"id": "Q13", "category": "CCAR-92", "question": "无人机在什么情况下需要申请空域许可"},
    {"id": "Q14", "category": "CCAR-92", "question": "无人机飞行的安全距离要求是什么"},
    {"id": "Q15", "category": "机场管理", "question": "机场使用许可证的申请条件是什么"},
    {"id": "Q16", "category": "机场管理", "question": "机场运行安全管理中谁来负责飞行区安全"},
    {"id": "Q18", "category": "旅客服务", "question": "航班延误超过多长时间航空公司需要为旅客提供餐饮"},
    {"id": "Q19", "category": "旅客服务", "question": "航空公司拒载旅客的合法理由有哪些"},
    {"id": "Q21", "category": "航空安全", "question": "航空安全检查中哪些物品禁止带上飞机"},
    {"id": "Q22", "category": "航空安全", "question": "民用航空安全信息报告的时限要求是什么"},
    {"id": "Q24", "category": "适航管理", "question": "民用航空器适航指令是做什么的"},
    {"id": "Q25", "category": "适航管理", "question": "航空器维修单位需要什么资质"},
    {"id": "Q26", "category": "空管", "question": "空中交通管制服务由哪些单位提供"},
    {"id": "Q27", "category": "空管", "question": "飞行程序设计和运行最低标准由谁审批"},
    {"id": "Q28", "category": "民用航空法", "question": "中华人民共和国对领空享有什么权利"},
    {"id": "Q30", "category": "人员资质", "question": "飞行员执照的种类有哪些"},
]

# 5 题快筛集（耗时短的代表性子集）
QUICK_PROBE = [
    {"id": "Q01", "category": "CCAR-121", "question": "不在《运行规范》的机场是否可以去备降"},
    {"id": "Q07", "category": "CCAR-121", "question": "签派员放行飞机时需要检查哪些事项"},
    {"id": "Q13", "category": "CCAR-92", "question": "无人机在什么情况下需要申请空域许可"},
    {"id": "Q19", "category": "旅客服务", "question": "航空公司拒载旅客的合法理由有哪些"},
    {"id": "Q26", "category": "空管", "question": "空中交通管制服务由哪些单位提供"},
]

# 参数值域（每参多个值）
PARAMETER_GRID = {
    # 检索层
    "RERANKER_MIN_SCORE": [0.0, 0.05, 0.1, 0.2, 0.3],
    "EVIDENCE_SORT_MIN_SUPPORTED_CONF": [0.3, 0.5, 0.7],
    "EVIDENCE_SORT_BY_CE": [True, False],
    "TREE_EARLY_ARTICLE_PENALTY": [0.4, 0.6, 0.8, 1.0],
    "TREE_GENERIC_ARTICLE_PENALTY": [0.0, 0.5, 1.0],
    "SYNTHESIS_EVIDENCE_TRUNCATE": [1000, 2000, 3000],
    "SYNTHESIS_EVIDENCE_LIMIT": [8, 12, 16],
}


def expand_grid(grid_dict=None):
    """把 {param: [values]} 展开成 [(param, value), ...] 列表。"""
    grid = grid_dict or PARAMETER_GRID
    out = []
    for param, values in grid.items():
        for value in values:
            out.append((param, value))
    return out


def config_id(param, value):
    """生成配置文件 id，如 RERANKER_MIN_SCORE=0.1。"""
    val = str(value)
    if isinstance(value, bool):
        val = "True" if value else "False"
    elif isinstance(value, float) and value == int(value):
        val = f"{value:.1f}"
    return f"{param}={val}"


def safe_filename(s):
    """把 config_id 转成文件系统安全的文件名。"""
    return s.replace(".", "_").replace("=", "_").replace(" ", "_")


def chunk_for_workers(items, n_workers):
    """把列表均分成 n_workers 份（最后一份可能少）。"""
    if n_workers <= 0:
        return [items]
    chunks = [[] for _ in range(n_workers)]
    for i, item in enumerate(items):
        chunks[i % n_workers].append(item)
    return [c for c in chunks if c]
