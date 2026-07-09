from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

OUT = r"C:\Users\42193\Desktop\2D转3D\论文带读笔记.docx"
BLUE = RGBColor(46, 116, 181)
DARK = RGBColor(31, 77, 120)
GRAY = RGBColor(95, 99, 104)
LIGHT = "E8EEF5"


def set_font(run, name="Microsoft YaHei", size=None, bold=None, color=None):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), name)
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color:
        run.font.color.rgb = color


def add_formula_block(doc, title, formulas):
    p = doc.add_paragraph()
    r = p.add_run(title)
    set_font(r, bold=True, color=BLUE, size=11)
    for name, formula, meaning in formulas:
        p2 = doc.add_paragraph()
        r = p2.add_run(f"  {name}:  {formula}")
        set_font(r, size=10, bold=True)
        p3 = doc.add_paragraph()
        r = p3.add_run(f"        → {meaning}")
        set_font(r, size=9.5, color=GRAY)


# Open existing document and append
doc = Document(OUT)

# ====== 3DGS ======
doc.add_page_break()
doc.add_heading("2. 3D Gaussian Splatting for Real-Time Radiance Field Rendering", level=1)

p = doc.add_paragraph()
r = p.add_run("作者：")
set_font(r, bold=True, size=10.5)
p.add_run("Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis")

p = doc.add_paragraph()
r = p.add_run("机构：")
set_font(r, bold=True, size=10.5)
p.add_run("Inria（法国国家信息与自动化研究所）+ Max-Planck-Institut für Informatik")

p = doc.add_paragraph()
r = p.add_run("发表：")
set_font(r, bold=True, size=10.5)
p.add_run("SIGGRAPH 2023")

p = doc.add_paragraph()
r = p.add_run("重要性：")
set_font(r, bold=True, size=10.5)
p.add_run("⭐⭐⭐⭐⭐ 与 NeRF 并列的领域基石")

p = doc.add_paragraph()
r = p.add_run("阅读难度：")
set_font(r, bold=True, size=10.5)
p.add_run("中等")

p = doc.add_paragraph()
r = p.add_run("开源：")
set_font(r, bold=True, size=10.5)
p.add_run("https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/")

# --- 核心问题 ---
doc.add_heading("2.1 NeRF → 3DGS 的核心跳跃", level=2)
p = doc.add_paragraph(
    "NeRF 用 MLP 隐式存场景 → 每次查询都要过网络 → 慢。"
    "3DGS 用几百万个彩色椭球显式存场景 → 直接投影到屏幕上叠加 → 极快（135+ fps）。"
    "两者的图像形成模型在数学上等价（都是 α 混合），但实现方式完全不同。"
)

# --- 什么是3D高斯 ---
doc.add_heading("2.2 什么是 3D 高斯？", level=2)
p = doc.add_paragraph(
    "一个 3D 高斯球就是一个「软椭球」——没有硬边界，中心最「密」，越远离中心越稀薄。"
    "和 NeRF 的体积密度 σ 精神一致：都是软的、连续的、可微的。"
)
p = doc.add_paragraph(
    "在计算机中，每个高斯球存为一组参数（共约 59 个/球）："
)
params = [
    ("位置 μ（均值）", "3 个浮点数 (x, y, z)", "球的中心在哪"),
    ("缩放 s", "3 个浮点数 (s_x, s_y, s_z)", "在三个轴方向上分别拉伸多长。s=(1,3,1) = 沿 Y 轴拉长的椭球"),
    ("旋转 q（四元数）", "4 个浮点数", "椭球朝向哪个方向（比如斜着的栏杆）"),
    ("不透明度 α", "1 个浮点数 [0,1)", "0=完全透明，1=完全不透明"),
    ("球谐系数 (SH)", "48 个浮点数", "从不同方向看这个球，颜色各是什么（替代 NeRF 的视角相关 MLP）"),
]
for name, val, meaning in params:
    p = doc.add_paragraph()
    r = p.add_run(f"• {name}（{val}）：")
    set_font(r, size=10.5, bold=True)
    r2 = p.add_run(meaning)
    set_font(r2, size=10.5)

# --- 三大创新 ---
doc.add_heading("2.3 三大技术创新", level=2)

doc.add_heading("创新①：3D 高斯作为场景表示（§4）", level=3)
p = doc.add_paragraph(
    "每个高斯球有协方差矩阵 Σ 描述形状。直接优化 Σ（3×3 矩阵）很难——梯度更新容易产生非法矩阵。"
)
p = doc.add_paragraph(
    "精巧解法：Σ = R S S^T R^T。将 Σ 分解为缩放矩阵 S 和旋转矩阵 R，"
    "分别优化缩放向量 s（3 个值）和旋转四元数 q（4 个值）。S S^T 天然半正定 → Σ 永远合法。"
)
p = doc.add_paragraph(
    "投影到屏幕：Σ' = J W Σ W^T J^T。3D 椭球 → 相机坐标系 → 变为 2D 椭圆斑（splat）。"
    "W 是视图变换，J 是投影的局部线性近似（雅可比矩阵）。"
)

doc.add_heading("创新②：自适应密度控制（§5）", level=3)
p = doc.add_paragraph("训练过程中动态增删高斯球：")
items = [
    ("欠重建（小球 + 梯度大）", "克隆！复制一个同样大小的球，沿梯度方向推开"),
    ("过重建（大球 + 梯度大）", "分裂！一个大球 → 两个小球，大小 ÷ 1.6"),
    ("几乎透明的球（α < 阈值）", "直接删除"),
    ("每 3000 次迭代", "强制把 α 降到接近 0 → 真正有用的球靠梯度涨回来，没用的漂浮物自然消失"),
]
for title, detail in items:
    p = doc.add_paragraph()
    r = p.add_run(f"• {title}：")
    set_font(r, size=10.5, bold=True)
    r2 = p.add_run(detail)
    set_font(r2, size=10.5)

doc.add_heading("创新③：瓦片式快速光栅化器（§6）", level=3)
p = doc.add_paragraph("五步流程：")
steps = [
    "筛掉不可见的球（视锥体裁剪）",
    "屏幕分成 16×16 瓦片，每个球判断覆盖哪些瓦片",
    "GPU Radix Sort：按深度 + 瓦片 ID 对所有球排序",
    "每个瓦片并行：从远到近叠加颜色，α 饱和则停止（前向渲染）",
    "反向传播：从近到远倒序遍历，计算梯度",
]
for i, step in enumerate(steps, 1):
    p = doc.add_paragraph()
    r = p.add_run(f"  {i}. {step}")
    set_font(r, size=10)

# --- 渲染流程 ---
doc.add_heading("2.4 渲染流程", level=2)
steps_full = [
    "输入：几百万个 3D 高斯球 + 相机位姿",
    "每个球 → 投影到 2D 屏幕 → 变成椭圆斑（splat）",
    "屏幕切成 16×16 瓦片，每个球分配给覆盖的瓦片",
    "瓦片内按深度排序所有球",
    "每个像素从远到近叠加：Color = Σ T_i · α_i · c_i（和 NeRF 数学等价！）",
    "α 累积到接近 1 → 停止该像素（后面的被挡住，处理也白费）",
    "对比真实照片 → L1 + D-SSIM Loss → 反向传播 → 更新所有球参数",
]
for i, step in enumerate(steps_full, 1):
    p = doc.add_paragraph()
    r = p.add_run(f"  {i}. {step}")
    set_font(r, size=10)

# --- NeRF vs 3DGS ---
doc.add_heading("2.5 NeRF vs 3DGS 对比", level=2)
table = doc.add_table(rows=8, cols=3)
table.style = 'Light Shading Accent 1'
headers = ["维度", "NeRF", "3DGS"]
data = [
    ["场景表示", "MLP 隐式函数", "百万级显式椭球"],
    ["数学公式", "体积渲染积分 (公式1)", "α 混合 (公式3)——数学等价！"],
    ["渲染方式", "光线步进（串行）", "瓦片光栅化（GPU并行）"],
    ["训练时间", "1-2 天", "6分钟-51分钟"],
    ["渲染速度", "~0.07 fps", "135+ fps"],
    ["存储大小", "~5 MB", "~几百 MB"],
    ["需要初始化？", "不需要", "需要 SfM 稀疏点云"],
]
for j, h in enumerate(headers):
    cell = table.cell(0, j)
    cell.text = h
    for p in cell.paragraphs:
        for r in p.runs:
            set_font(r, size=9.5, bold=True)
for i, row in enumerate(data):
    for j, val in enumerate(row):
        cell = table.cell(i+1, j)
        cell.text = val
        for p in cell.paragraphs:
            for r in p.runs:
                set_font(r, size=9.5)

# --- 与本课题关系 ---
doc.add_heading("2.6 与本课题的关系", level=2)
p = doc.add_paragraph(
    "3DGS 实现了神经渲染领域的「实时化」突破——135 fps 的高质量渲染使得实时应用成为可能。"
    "但和 NeRF 一样，3DGS 仍然需要几十张输入图 + SfM 位姿 + 为每个场景单独训练。"
)
p = doc.add_paragraph(
    "本项目关注的论文正是要解决这些限制："
)
items_relation = [
    ("pixelSplat / AnySplat（04分类）", "前馈式 3DGS：输入 1-2 张图，直接预测 3D 高斯球参数，不需要逐场景重训"),
    ("Depth Anything / Depth Pro（03分类）", "单图深度估计：为 3DGS 提供单图几何初始化，替代多视图 SfM"),
    ("3D Photo / SLIDE / TMPI（02分类）", "用分层深度表达（MPI）处理单图，是 3DGS 之外的另一种技术路线"),
    ("AAA Gaussians / EVER / StochasticSplats（05分类）", "优化 3DGS 渲染质量和稳定性"),
    ("3DGS-LM / ResGS / RegGS（06分类）", "优化 3DGS 训练速度、稀疏视图、相机位姿等"),
]
for title, detail in items_relation:
    p = doc.add_paragraph()
    r = p.add_run(f"• {title}：")
    set_font(r, size=10.5, bold=True)
    r2 = p.add_run(detail)
    set_font(r2, size=10.5)

doc.save(OUT)
print(f"Appended 3DGS notes to: {OUT}")
