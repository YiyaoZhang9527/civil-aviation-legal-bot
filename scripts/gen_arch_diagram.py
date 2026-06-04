"""用 Pillow 画管线架构流程图。"""
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Pillow 未安装，尝试安装...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow", "-q"])
    from PIL import Image, ImageDraw, ImageFont

W, H = 2200, 900
BG = (10, 14, 23)
TEXT = (226, 232, 240)
DIM = (100, 116, 139)
BORDER = (30, 41, 59)

# Colors per stage
C_BLUE = (59, 130, 246)
C_BLUE_BG = (15, 25, 50)
C_GREEN = (16, 185, 129)
C_GREEN_BG = (6, 50, 40)
C_AMBER = (245, 158, 11)
C_AMBER_BG = (60, 35, 8)
C_PURPLE = (168, 85, 247)
C_PURPLE_BG = (45, 15, 70)
C_RED = (239, 68, 68)
C_RED_BG = (50, 15, 15)
C_CYAN = (6, 182, 212)
C_CYAN_BG = (5, 40, 50)

# Try to load a CJK font
FONT_PATHS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
]

font = None
font_bold = None
for p in FONT_PATHS:
    if Path(p).exists():
        try:
            font = ImageFont.truetype(p, 16)
            font_bold = ImageFont.truetype(p, 18)
            font_title = ImageFont.truetype(p, 28)
            font_small = ImageFont.truetype(p, 13)
            font_tag = ImageFont.truetype(p, 12)
            break
        except Exception:
            continue

if font is None:
    font = ImageFont.load_default()
    font_bold = font
    font_title = font
    font_small = font
    font_tag = font


def draw_rounded_rect(draw, xy, fill, outline, radius=12, width=1):
    x0, y0, x1, y1 = xy
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def draw_arrow(draw, x0, y0, x1, y1, color=DIM, width=2):
    draw.line([(x0, y0), (x1, y1)], fill=color, width=width)
    # arrowhead
    import math
    angle = math.atan2(y1 - y0, x1 - x0)
    a1 = angle + math.pi * 0.85
    a2 = angle - math.pi * 0.85
    sz = 8
    draw.polygon([
        (x1, y1),
        (x1 + sz * math.cos(a1), y1 + sz * math.sin(a1)),
        (x1 + sz * math.cos(a2), y1 + sz * math.sin(a2)),
    ], fill=color)


def draw_dashed_arrow(draw, x0, y0, x1, y1, color=C_RED, width=2):
    import math
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    dash_len = 8
    gap_len = 5
    steps = int(length / (dash_len + gap_len))
    for i in range(steps):
        t0 = i * (dash_len + gap_len) / length
        t1 = min((i * (dash_len + gap_len) + dash_len) / length, 1.0)
        sx = x0 + dx * t0
        sy = y0 + dy * t0
        ex = x0 + dx * t1
        ey = y0 + dy * t1
        draw.line([(sx, sy), (ex, ey)], fill=color, width=width)


def text_center(draw, text, cx, cy, fnt, fill):
    bbox = draw.textbbox((0, 0), text, font=fnt)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((cx - tw // 2, cy - th // 2), text, font=fnt, fill=fill)


def draw_node(draw, cx, cy, w, h, title, sub_lines, bg, border, title_color):
    x0, y0 = cx - w // 2, cy - h // 2
    x1, y1 = cx + w // 2, cy + h // 2
    draw_rounded_rect(draw, (x0, y0, x1, y1), bg, border, radius=10, width=1)
    text_center(draw, title, cx, cy - h // 2 + 28, font_bold, title_color)
    for i, line in enumerate(sub_lines):
        text_center(draw, line, cx, cy - h // 2 + 52 + i * 18, font_small, DIM)


def draw_tag(draw, cx, cy, text, color):
    bbox = draw.textbbox((0, 0), text, font=font_tag)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    pad = 4
    x0, y0 = cx - tw // 2 - pad, cy - th // 2 - pad
    x1, y1 = cx + tw // 2 + pad, cy + th // 2 + pad
    bg = tuple(c // 5 for c in color)
    draw_rounded_rect(draw, (x0, y0, x1, y1), bg, color, radius=4, width=1)
    draw.text((cx - tw // 2, cy - th // 2), text, font=font_tag, fill=color)


img = Image.new("RGB", (W, H), BG)
draw = ImageDraw.Draw(img)

# Title
text_center(draw, "智能民航法律问答系统 · 管线架构", W // 2, 35, font_title, TEXT)
text_center(draw, "Query → Rewrite → Retrieve → Verify → Synthesize → Reflex → Answer", W // 2, 68, font_small, DIM)

# Layout positions
y_top = 120
node_h = 80
node_w = 200
gap_y = 16
arrow_y = y_top + node_h // 2

# Column X positions
cols = {
    "input": 110,
    "understand": 340,
    "retrieve": 600,
    "verify": 860,
    "generate": 1120,
    "reflex": 1380,
    "output": 1580,
}

# Stage labels
for name, cx, color, label in [
    ("input", cols["input"], C_CYAN, "输入"),
    ("understand", cols["understand"], C_BLUE, "理解层"),
    ("retrieve", cols["retrieve"], C_GREEN, "检索层"),
    ("verify", cols["verify"], C_AMBER, "校验层"),
    ("generate", cols["generate"], C_PURPLE, "生成层"),
    ("reflex", cols["reflex"], C_RED, "自检层"),
    ("output", cols["output"], C_GREEN, "输出"),
]:
    text_center(draw, label, cx, y_top - 16, font_small, color)

# --- INPUT ---
cx = cols["input"]
draw_rounded_rect(draw, (cx - 55, y_top, cx + 55, y_top + node_h), C_CYAN_BG, C_CYAN, radius=16, width=2)
text_center(draw, "❓", cx, y_top + 22, font_bold, C_CYAN)
text_center(draw, "用户提问", cx, y_top + 52, font_bold, C_CYAN)

# --- UNDERSTAND ---
cx = cols["understand"]
nodes_u = [
    ("主体识别", ["当事人·事件链·假设"]),
    ("争点识别", ["法律问题分类提炼"]),
    ("查询改写", ["口语→法律查询", "原始query保底"]),
]
for i, (title, subs) in enumerate(nodes_u):
    ny = y_top + i * (node_h - 8)
    draw_node(draw, cx, ny + node_h // 2, node_w - 20, node_h - 8, title, subs, C_BLUE_BG, (30, 58, 95), (147, 187, 252))

# --- RETRIEVE ---
cx = cols["retrieve"]
draw_node(draw, cx, y_top + 90, node_w, node_h + 30, "🌲 树检索", ["法规 → 章节 → 条文", "三级逐层剪枝"], C_GREEN_BG, (6, 78, 59), (110, 231, 183))
draw_tag(draw, cx, y_top + 155, "8-12条证据", C_GREEN)

# --- VERIFY ---
cx = cols["verify"]
nodes_v = [
    ("Cross-Encoder 精排", ["supported / partial / unsupported"]),
    ("冲突检测", ["上位法 > 下位法"]),
    ("问题回退打分", ["claim低分→原始问题"]),
]
for i, (title, subs) in enumerate(nodes_v):
    ny = y_top + i * (node_h - 8)
    draw_node(draw, cx, ny + node_h // 2, node_w - 10, node_h - 8, title, subs, C_AMBER_BG, (120, 53, 15), (252, 211, 77))
# tags for verify
draw_tag(draw, cx - 40, y_top + node_h - 22, "[已验证]", C_GREEN)
draw_tag(draw, cx + 40, y_top + node_h - 22, "[未通过]", C_AMBER)

# --- GENERATE ---
cx = cols["generate"]
nodes_g = [
    ("结构化答案", ["JSON·claim带node_id"]),
    ("Set-Membership", ["引用不存在→⚠️待核实", "警告不删除"]),
    ("采纳判断", ["未验证证据根据问题", "自行决定是否采纳"]),
]
for i, (title, subs) in enumerate(nodes_g):
    ny = y_top + i * (node_h - 8)
    draw_node(draw, cx, ny + node_h // 2, node_w - 10, node_h - 8, title, subs, C_PURPLE_BG, (88, 28, 135), (216, 180, 254))

# --- REFLEXION ---
cx = cols["reflex"]
draw_node(draw, cx, y_top + 90, node_w - 20, node_h + 20, "🔁 自检循环", ["质量评估→补搜→重试"], C_RED_BG, (127, 29, 29), (252, 165, 165))
draw_tag(draw, cx, y_top + 145, "max 2轮", C_RED)

# --- OUTPUT ---
cx = cols["output"]
draw_rounded_rect(draw, (cx - 55, y_top + 55, cx + 55, y_top + 55 + node_h), C_GREEN_BG, C_GREEN, radius=16, width=2)
text_center(draw, "📋", cx, y_top + 77, font_bold, C_GREEN)
text_center(draw, "最终答案", cx, y_top + 107, font_bold, C_GREEN)

# --- Arrows between stages ---
arrow_pairs = [
    ("input", "understand"),
    ("understand", "retrieve"),
    ("retrieve", "verify"),
    ("verify", "generate"),
    ("generate", "reflex"),
    ("reflex", "output"),
]
mid_y = y_top + 90
for left, right in arrow_pairs:
    x0 = cols[left] + (60 if left == "input" else 95)
    x1 = cols[right] - (60 if right == "output" else 95)
    draw_arrow(draw, x0, mid_y, x1, mid_y)

# --- Reflexion loop (dashed, bottom) ---
loop_y = y_top + 220
# From reflex down
draw_dashed_arrow(draw, cols["reflex"], y_top + 120 + 25, cols["reflex"], loop_y, C_RED)
# Horizontal back to retrieve
draw_dashed_arrow(draw, cols["reflex"], loop_y, cols["retrieve"], loop_y, C_RED)
# Up to retrieve
draw_arrow(draw, cols["retrieve"], loop_y, cols["retrieve"], y_top + 120, C_RED)
text_center(draw, "质量不足 → 补搜证据 → 重新检索", (cols["reflex"] + cols["retrieve"]) // 2, loop_y + 18, font_small, C_RED)

# Legend
ly = H - 50
legend = [
    (C_BLUE, "理解层 (LLM)"),
    (C_GREEN, "检索层 (向量+BM25+精排)"),
    (C_AMBER, "校验层 (Cross-Encoder)"),
    (C_PURPLE, "生成层 (LLM JSON)"),
    (C_RED, "自检循环"),
]
lx = 100
for color, label in legend:
    draw.ellipse((lx, ly - 5, lx + 10, ly + 5), fill=color)
    draw.text((lx + 16, ly - 8), label, font=font_small, fill=DIM)
    lx += len(label) * 13 + 40

out = Path(__file__).resolve().parents[1] / "architecture-flow.png"
img.save(out, "PNG")
print(f"Saved: {out}")
