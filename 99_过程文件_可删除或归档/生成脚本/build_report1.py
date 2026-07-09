from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_ORIENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import os

OUT = r"C:\Users\42193\Desktop\2D转3D\汇报\论文总结1\论文总结1_NeRF与3DGS.docx"
IMG_DIR = r"C:\Users\42193\Desktop\2D转3D\汇报\论文总结1\图片"

BLUE = RGBColor(46, 116, 181)
DARK = RGBColor(31, 77, 120)
GRAY = RGBColor(95, 99, 104)
RED = RGBColor(200, 60, 60)
GREEN = RGBColor(40, 140, 80)
LIGHT_BLUE = "E8EEF5"
LIGHT_GREEN = "E8F5E9"
LIGHT_RED = "FFEBEE"


def set_font(run, name="Microsoft YaHei", size=None, bold=None, color=None):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), name)
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color:
        run.font.color.rgb = color


def add_section_header(doc, text, color=BLUE):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    set_font(r, size=13, bold=True, color=color)
    return p


def add_body(doc, text, size=10.5):
    p = doc.add_paragraph()
    r = p.add_run(text)
    set_font(r, size=size)
    return p


def add_bullet(doc, text, bold_prefix="", size=10.5):
    p = doc.add_paragraph(style="List Bullet")
    if bold_prefix:
        r = p.add_run(bold_prefix)
        set_font(r, size=size, bold=True)
    r2 = p.add_run(text)
    set_font(r2, size=size)
    return p


def add_image(doc, img_name, width_inches=5.5):
    img_path = os.path.join(IMG_DIR, img_name)
    if os.path.exists(img_path):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run()
        r.add_picture(img_path, width=Inches(width_inches))
        # Caption
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = cap.add_run(img_name.replace('.png', '').replace('_', ' '))
        set_font(r, size=8.5, color=GRAY)
    else:
        add_body(doc, f"[图片未找到: {img_name}]", size=9)


def add_adv_lim(doc, advantages, limitations):
    """Add Advantages and Limitations side by side (as two sections)"""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    r = p.add_run("✅ Advantages / 优势")
    set_font(r, size=11, bold=True, color=GREEN)

    for adv in advantages:
        p = doc.add_paragraph(style="List Bullet")
        r = p.add_run(adv)
        set_font(r, size=10)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    r = p.add_run("⚠️ Limitations / 局限")
    set_font(r, size=11, bold=True, color=RED)

    for lim in limitations:
        p = doc.add_paragraph(style="List Bullet")
        r = p.add_run(lim)
        set_font(r, size=10)


def add_page_break(doc):
    doc.add_page_break()


# ========== BUILD DOCUMENT ==========
doc = Document()

# Page setup
sec = doc.sections[0]
sec.top_margin = Cm(2)
sec.bottom_margin = Cm(1.8)
sec.left_margin = Cm(2.2)
sec.right_margin = Cm(2.2)

# Styles
style = doc.styles["Normal"]
style.font.name = "Microsoft YaHei"
style.font.size = Pt(10.5)
style.paragraph_format.space_after = Pt(4)
style.paragraph_format.line_spacing = 1.25

for sname, size, color, before, after in [
    ("Heading 1", 20, DARK, 20, 12),
    ("Heading 2", 15, BLUE, 16, 8),
    ("Heading 3", 12, DARK, 10, 4),
]:
    st = doc.styles[sname]
    st.font.name = "Microsoft YaHei"
    st.font.size = Pt(size)
    st.font.bold = True
    st.font.color.rgb = color
    st.paragraph_format.space_before = Pt(before)
    st.paragraph_format.space_after = Pt(after)

# List Bullet style
if "List Bullet" in [s.name for s in doc.styles]:
    lb = doc.styles["List Bullet"]
    lb.font.name = "Microsoft YaHei"
    lb.font.size = Pt(10.5)

# ====== COVER ======
p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(80)
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("论文总结汇报（一）")
set_font(r, size=28, bold=True, color=DARK)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(8)
r = p.add_run("NeRF · 3D Gaussian Splatting")
set_font(r, size=16, color=BLUE)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(30)
r = p.add_run("2D 转浅 3D 项目 · 基础必读论文")
set_font(r, size=12, color=GRAY)

# Overview table
table = doc.add_table(rows=4, cols=2)
table.style = 'Light Shading Accent 1'
table.autofit = True

overview_data = [
    ("研究方向", "单张 2D 图像 → 浅 3D 表达 → 新视角合成"),
    ("本次论文", "NeRF (ECCV 2020) + 3D Gaussian Splatting (SIGGRAPH 2023)"),
    ("论文地位", "领域公认的两篇基石论文，后续几乎所有方法都建立在这两个工作之上"),
    ("阅读方法", "逐段翻译 → 大意梳理 → 概念拆解"),
]
for i, (a, b) in enumerate(overview_data):
    table.cell(i, 0).text = ""
    table.cell(i, 1).text = ""
    p_a = table.cell(i, 0).paragraphs[0]
    r_a = p_a.add_run(a)
    set_font(r_a, size=10, bold=True)
    p_b = table.cell(i, 1).paragraphs[0]
    r_b = p_b.add_run(b)
    set_font(r_b, size=10)

add_page_break(doc)

# ====== NERF ======
doc.add_heading("第一篇：NeRF", level=1)
doc.add_heading("Representing Scenes as Neural Radiance Fields for View Synthesis", level=2)

p = doc.add_paragraph()
r = p.add_run("作者：")
set_font(r, bold=True, size=10)
p.add_run("Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng")
p = doc.add_paragraph()
r = p.add_run("机构：")
set_font(r, bold=True, size=10)
p.add_run("UC Berkeley + Google Research  |  ECCV 2020  |  开源: github.com/bmild/nerf")

# --- 1.1 动机 ---
add_section_header(doc, "1.1 动机与问题")

add_body(doc, "核心问题：给你几十张从不同角度拍摄的照片 + 每张照片的相机位置，能否生成从任意新角度看的照片（新视角合成）？")

add_body(doc, "此前的方法存在两个极端：")
add_bullet(doc, "体素网格方法：3D 空间切方块存储，分辨率翻倍 → 内存 ×8，根本无法表示高精度场景")
add_bullet(doc, "神经网络隐式方法（SRN、NV）：用 MLP 隐式表示场景，但渲染质量远不够好")
add_bullet(doc, "NeRF 的目标：用神经网络实现连续、高精度的场景表示，同时克服体素网格的存储灾难")

add_image(doc, "NeRF_Fig1_全景流程图.png")

# --- 1.2 Method ---
add_section_header(doc, "1.2 方法核心")

add_body(doc, "NeRF 的方法可以总结为：用 MLP 记住空间中的每个点的「颜色」和「实在程度」，渲染时沿每根光线采点叠加。")

add_body(doc, "网络结构：", size=10.5)
add_bullet(doc, "(x,y,z) → 位置编码(L=10) → [8层FC, 256维, ReLU] → 密度 σ + 256维特征向量")
add_bullet(doc, "特征向量 + 观察方向 d → 位置编码(L=4) → [1层FC, 128维] → RGB颜色 c")
add_bullet(doc, "密度 σ 只取决于位置（一个点是不是固体跟你怎么看无关），颜色 c 取决于位置+方向（镜面反射随视角变化）——保证了多视图一致性")

add_body(doc, "三大技术创新：", size=10.5)

add_bullet(doc, "① 连续场景表示：场景编码进神经网络权重（~5MB），替代体素网格（~15GB）。空间中任意精度坐标都可查询。", "创新1: ")
add_bullet(doc, "② 位置编码（PE）：把输入坐标通过 sin/cos 映射到高维——γ(p)=[sin(p),cos(p),sin(2p),cos(2p),...]。本质是把不同频率的傅里叶基底提前摆好，MLP 只需学加权组合。去掉 PE → PSNR 降 2.24，画面明显模糊。", "创新2: ")
add_bullet(doc, "③ 分层采样（粗/细网络）：粗网络先均匀采 64 个点找到物体大致位置，细网络在物体附近额外采 128 个点做精细渲染。类比：先用余光扫一眼房间找人（粗网络），再盯着仔细看（细网络）。", "创新3: ")

add_image(doc, "NeRF_Fig2_网络与渲染管线.png")

add_body(doc, "核心体积渲染公式（通俗版）：", size=10.5)
add_bullet(doc, "C(r) = ∫ T(t) · σ(r(t)) · c(r(t), d) dt  —— 沿光线连续积分")
add_bullet(doc, "Ĉ(r) = Σ Tᵢ · αᵢ · cᵢ  —— 离散化后实际使用，αᵢ = 1−exp(−σᵢδᵢ)")
add_bullet(doc, "物理直觉：从前到后叠 N 层半透明彩色玻璃，越后面的层被前面的削弱越多")

add_image(doc, "NeRF_Fig3_视角相关辐射度.png", 5.0)
add_image(doc, "NeRF_Fig4_消融对比.png", 5.0)

# --- 1.3 Results ---
add_section_header(doc, "1.3 关键实验结果")

add_body(doc, "三个测试数据集：Diffuse Synthetic 360°（简单几何体）、Realistic Synthetic 360°（复杂材质）、Real Forward-Facing（手持手机实拍）")

add_body(doc, "消融实验（表2）核心发现：")
add_bullet(doc, "去掉位置编码 → PSNR ↓2.24（最大降幅，PE 最重要）")
add_bullet(doc, "去掉视角依赖 → PSNR ↓3.35（镜面反射场景尤其重要）")
add_bullet(doc, "去掉分层采样 → PSNR ↓0.95（主要影响效率）")
add_bullet(doc, "仅用 25 张图 → 效果已超过其他方法用 100 张图")
add_bullet(doc, "单场景存储 ~5MB vs 对比方法 ~15GB，压缩比约 3000×")

add_image(doc, "NeRF_Fig5_合成场景对比.png")
add_image(doc, "NeRF_Fig6_真实场景对比.png")

# --- 1.4 Summary ---
add_section_header(doc, "1.4 总结")

add_adv_lim(doc, [
    "首次实现连续神经场景表示，能以极高保真度渲染复杂真实场景的新视角",
    "位置编码巧妙解决了 MLP 的低频偏置问题，使网络能学到高频细节",
    "极致的空间效率：5MB 存一个完整场景",
    "可微分的体积渲染管线，端到端训练，无需 3D 真值监督",
], [
    "每个场景需单独训练 1-2 天（V100 GPU），无法泛化到新场景",
    "需要几十张输入照片 + 精确的相机位姿（依赖 COLMAP SfM）",
    "渲染慢（~0.07 fps），远不能实时——每帧需约 1.2 亿次 MLP 查询",
    "对相机位姿精度敏感，SfM 失败的场景无法处理",
])

add_page_break(doc)

# ====== 3DGS ======
doc.add_heading("第二篇：3D Gaussian Splatting", level=1)
doc.add_heading("3D Gaussian Splatting for Real-Time Radiance Field Rendering", level=2)

p = doc.add_paragraph()
r = p.add_run("作者：")
set_font(r, bold=True, size=10)
p.add_run("Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis")
p = doc.add_paragraph()
r = p.add_run("机构：")
set_font(r, bold=True, size=10)
p.add_run("Inria（法国）+ Max-Planck-Institut  |  SIGGRAPH 2023  |  开源: repo-sam.inria.fr/fungraph/3d-gaussian-splatting/")

# --- 2.1 动机 ---
add_section_header(doc, "2.1 动机与问题")

add_body(doc, "NeRF 的核心矛盾：质量好但太慢，快速方法快但质量差。在 1080p 下没有方法能实现实时渲染（≥30 fps）。")

add_body(doc, "3DGS 的野心：能否在保持 NeRF 级质量的同时，做到实时渲染？——答案是可以，135 fps。")

add_body(doc, "关键洞察：NeRF 的「体积渲染积分」和传统图形学的「α 混合」在数学上完全等价——但实现方式决定速度。NeRF 用光线步进（慢），3DGS 用 GPU 光栅化（快）。")

add_image(doc, "3DGS_Fig1_速质对比.png")

# --- 2.2 Method ---
add_section_header(doc, "2.2 方法核心")

add_body(doc, "3DGS 把场景表示从「隐式 MLP」切换为「几百万个显式 3D 椭球（高斯）」。每个椭球存 59 个浮点数：")

add_bullet(doc, "位置 μ (x,y,z) — 椭球的中心")
add_bullet(doc, "缩放 s (sx,sy,sz) — 三个轴向拉伸多长。如 s=(1,3,1) = 沿 Y 轴拉长")
add_bullet(doc, "旋转 q (四元数) — 椭球朝向（如斜栏杆）")
add_bullet(doc, "不透明度 α [0,1) — 0=透明，1=不透明")
add_bullet(doc, "球谐系数 SH (48维) — 从不同方向看颜色各不同（替代 NeRF 视角相关 MLP）")

add_body(doc, "三大技术创新：", size=10.5)

add_bullet(doc, "① 3D 高斯表示（§4）：协方差矩阵分解 Σ = R S SᵀRᵀ，分别优化缩放向量 s 和旋转四元数 q。S Sᵀ 天然半正定 → Σ 永远合法，不会因梯度更新产生非法矩阵。投影到 2D：Σ' = J W Σ Wᵀ Jᵀ。", "创新1: ")
add_bullet(doc, "② 自适应密度控制（§5）：训练中动态增删球——小球+梯度大 → 克隆；大球+梯度大 → 分裂成两个；α<阈值 → 删除。每3000次强制重置 α → 有用球的 α 靠梯度恢复，漂浮物自然消失。", "创新2: ")
add_bullet(doc, "③ 瓦片光栅化器（§6）：屏幕分16×16瓦片 → GPU Radix Sort 按深度排序所有球 → 每个瓦片并行从前到后叠加 → α 饱和则停止。充分利用 GPU 共享内存（比全局显存快~100×）。", "创新3: ")

add_image(doc, "3DGS_Fig2_优化流程全景.png")
add_image(doc, "3DGS_Fig3_各向异性高斯.png", 5.0)
add_image(doc, "3DGS_Fig4_密度控制.png", 5.0)

# --- 2.3 Results ---
add_section_header(doc, "2.3 关键实验结果")

add_body(doc, "在 Mip-NeRF360、Tanks&Temples、Deep Blending 三个真实场景数据集 + Synthetic NeRF 合成数据集上全面评估：")

add_bullet(doc, "训练 7K 迭代 (≈6分钟) 质量已与最快的先前方法相当；训练 30K 迭代 (≈51分钟) 达到 SOTA，略超 Mip-NeRF360")
add_bullet(doc, "渲染速度：135+ fps @1080p —— 比 Mip-NeRF360 (0.07 fps) 快近 2000 倍，比 InstantNGP (9.2 fps) 快 15 倍")
add_bullet(doc, "存储：约 500-750MB（百万级高斯球）—— 远大于 NeRF 的 5MB，但远小于体素方法的 15GB")
add_bullet(doc, "关键优势：即使只用 7K 迭代（6分钟），质量已具有竞争力")

add_image(doc, "3DGS_Fig5_结果对比.png")
add_image(doc, "3DGS_Table1_量化结果.png")

# --- 2.4 Summary ---
add_section_header(doc, "2.4 总结")

add_adv_lim(doc, [
    "实时渲染！135+ fps @1080p，首次在保持 SOTA 质量的同时实现真正实时",
    "训练快：6 分钟即可获得有竞争力的结果，51 分钟达到 SOTA（vs NeRF 的 1-2 天）",
    "显式表示便于理解和调试——可以直接可视化高斯球",
    "数学上与 NeRF 的体积渲染等价，但实现上选择了 GPU 友好的光栅化路径",
    "自适应密度控制——训练过程全自动，无需手动调高斯数量",
], [
    "仍需要几十张输入照片 + SfM 相机位姿（和 NeRF 一样）",
    "每个场景需单独训练，无法泛化到新场景（和 NeRF 一样）",
    "存储较大（几百 MB vs NeRF 的 5MB）",
    "对初始 SfM 点云质量有依赖——SfM 失败的区域难以覆盖",
    "显式表示带来新挑战：漂浮物管理、排序开销、各向异性球的数值稳定性",
])

add_page_break(doc)

# ====== COMPARISON & NEXT ======
doc.add_heading("两篇论文对比总结", level=1)

table = doc.add_table(rows=8, cols=3)
table.style = 'Light Shading Accent 1'
comp_headers = ["维度", "NeRF (ECCV 2020)", "3DGS (SIGGRAPH 2023)"]
comp_data = [
    ["场景表示", "MLP 隐式函数，连续表示", "百万级 3D 高斯椭球，显式表示"],
    ["数学公式", "体积渲染积分 C(r)=∫Tσc dt", "α 混合 ΣTᵢαᵢcᵢ（数学等价！）"],
    ["渲染方式", "光线步进（串行采样）", "瓦片光栅化（GPU 并行）"],
    ["训练时间", "1-2 天 (V100)", "6 分钟-51 分钟 (A6000)"],
    ["渲染速度", "~0.07 fps", "135+ fps（实时！）"],
    ["存储大小", "~5 MB（极省）", "~500 MB（可接受）"],
    ["共同局限", "都需要几十张输入照片 + SfM 位姿 + 逐场景重训", ""],
]
for j, h in enumerate(comp_headers):
    cell = table.cell(0, j)
    cell.text = ""
    p = cell.paragraphs[0]
    r = p.add_run(h)
    set_font(r, size=10, bold=True)
for i, row in enumerate(comp_data):
    for j, val in enumerate(row):
        cell = table.cell(i+1, j)
        cell.text = ""
        p = cell.paragraphs[0]
        r = p.add_run(val)
        set_font(r, size=9.5)

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(12)

add_section_header(doc, "对本项目的启示与下一步")

add_body(doc, "这两篇基石论文建立了「新视角合成」领域的通用语言和数学框架。但它们的共同局限——需要多张输入图 + SfM 位姿 + 逐场景重训——正是本项目需要解决的。")

add_body(doc, "接下来要读的论文按解决这些局限的方向组织：")

add_bullet(doc, "单图输入 + 深度估计 → 替代多视图 SfM", "03 深度估计基础模型（Depth Anything V2, Depth Pro）：")
add_bullet(doc, "单图/双图直接预测 3DGS → 不需要逐场景重训", "04 单图与前馈 3DGS（pixelSplat, AnySplat, SHARP）：")
add_bullet(doc, "分层深度表达（MPI）→ 另一种技术路线", "02 单图 3D 照片与分层表达（3D Photo, SLIDE, TMPI, MINE）：")
add_bullet(doc, "提升 3DGS 质量和稳定性", "05+06 渲染优化与稀疏视图：")

doc.save(OUT)
print(f"Saved to: {OUT}")
