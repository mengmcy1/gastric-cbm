from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

OUT = r"C:\Users\42193\Desktop\2D转3D\汇报\论文总结1\论文总结1_NeRF与3DGS.docx"
BLUE = RGBColor(46, 116, 181)
DARK = RGBColor(31, 77, 120)
GRAY = RGBColor(95, 99, 104)
ORANGE = RGBColor(220, 120, 40)


def set_font(run, name="Microsoft YaHei", size=None, bold=None, color=None):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), name)
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color:
        run.font.color.rgb = color


doc = Document(OUT)

# ====== 讲稿 ======
doc.add_page_break()
doc.add_heading("附录：汇报讲稿（精简版）", level=1)

p = doc.add_paragraph()
r = p.add_run("以下为向导师汇报时可参考的口语化讲稿。")
set_font(r, size=10, color=GRAY)

# --- 开场 ---
doc.add_heading("一、开场（约 1 分钟）", level=2)

p = doc.add_paragraph()
r = p.add_run("讲稿：")
set_font(r, size=10.5, bold=True, color=ORANGE)

p = doc.add_paragraph()
p.add_run(
    "老师好，我汇报一下最近的学习进展。我的课题方向是「单张2D图像转浅3D表达」，"
    "目标是输入一张普通照片，生成一个可以自由旋转视角的3D场景。\n\n"
    "这段时间我先从领域的两篇基石论文入手——NeRF 和 3D Gaussian Splatting。"
    "这两篇论文基本定义了整个「新视角合成」方向的数学框架和技术路线，"
    "后续几乎所有方法都是在这两个工作之上的改进。\n\n"
    "我每篇论文采用三步阅读法：先逐段翻译理解字面意思，再梳理段落在全文中的作用，"
    "最后对核心概念做通俗拆解。下面我分别汇报两篇论文的核心内容和我的理解。"
)

# --- NeRF ---
doc.add_heading("二、NeRF（约 2-3 分钟）", level=2)

p = doc.add_paragraph()
r = p.add_run("讲稿：")
set_font(r, size=10.5, bold=True, color=ORANGE)

p = doc.add_paragraph()
p.add_run(
    "第一篇是 NeRF，2020 年 ECCV 的论文，来自 UC Berkeley 和 Google Research。"
    "它解决的核心问题是：给你几十张从不同角度拍的照片和每张照片的相机位置，"
    "能不能生成任意新角度的照片？\n\n"
    "NeRF 的做法非常巧妙——它没有用传统的 3D 建模方式，而是训练了一个最简单的全连接神经网络，"
    "让这个网络「记住」整个场景。输入是一个 5D 坐标——3D 空间位置加上 2D 观察方向，"
    "输出是该位置的颜色和「密度」。密度表示那个点在空间中有多「实在」——密度高就是固体，密度低就是空气。\n\n"
    "渲染新视角时，从虚拟相机发出光线穿过场景，沿光线采几百个点，每个点问一次网络，"
    "然后把所有点的颜色按密度加权叠加，就得到了一个像素的颜色。整个过程可微分，"
    "所以可以直接用梯度下降来训练。\n\n"
    "这里我想特别讲一下 NeRF 的三个技术创新：\n\n"
    "第一，用神经网络存场景替代了传统的体素网格。体素网格就是 3D 版的像素——"
    "把空间切成小方块。问题是分辨率每翻一倍，存储量要翻 8 倍。NeRF 用网络权重存场景，"
    "一个完整场景只需 5MB，而之前的方法要 15GB，压缩了差不多 3000 倍。\n\n"
    "第二，位置编码。这是我觉得最精巧的设计。直接用 xyz 坐标输入网络，渲染结果是糊的——"
    "因为神经网络天然偏向学低频的平滑函数。NeRF 的解决方案是把输入坐标通过 sin/cos "
    "函数展开成不同频率的基底，就像傅里叶级数那样。网络只需要学会给这些基底加权组合，"
    "而不需要自己「发明」高频分量。这就像给网络配了一副高清眼镜。\n\n"
    "第三，分层采样。用两个网络——粗网络先快速扫一遍找到物体在哪，"
    "细网络再在物体表面附近精细采样。这样避免了在空气中浪费算力。\n\n"
    "但 NeRF 有明显的局限：每个场景需要训练 1-2 天，渲染一帧要几十秒，"
    "而且必须输入几十张照片加相机位姿。这些局限恰好是后续工作要解决的。"
)

# --- 3DGS ---
doc.add_heading("三、3D Gaussian Splatting（约 2-3 分钟）", level=2)

p = doc.add_paragraph()
r = p.add_run("讲稿：")
set_font(r, size=10.5, bold=True, color=ORANGE)

p = doc.add_paragraph()
p.add_run(
    "第二篇是 3D Gaussian Splatting，简称 3DGS，2023 年 SIGGRAPH 的论文，"
    "来自法国的 Inria 研究所。这篇论文解决的是 NeRF 最大的痛点——速度。\n\n"
    "NeRF 渲染慢的根本原因是它用「光线步进」——每根光线要采样 192 个点，"
    "每个点过一遍神经网络，渲染一张 800×800 的图要做约 1.2 亿次网络推理。\n\n"
    "3DGS 换了一个完全不同的思路：不用神经网络隐式存场景了，而是用几百万个显式的"
    "彩色半透明椭球——叫 3D 高斯——直接摆在 3D 空间中。每个椭球存了 59 个参数："
    "位置、形状、旋转方向、透明度、以及从不同角度看是什么颜色。\n\n"
    "渲染的时候，把所有椭球投影到屏幕上，按深度排好队，从远到近叠颜色。"
    "这个操作 GPU 极其擅长——因为它就是传统图形学里的 α 混合，和游戏引擎渲染透明物体的原理一样。"
    "结果就是：135 帧每秒，比 NeRF 快了近 2000 倍。\n\n"
    "而且数学上，3DGS 的 α 混合和 NeRF 的体积渲染积分是完全等价的——"
    "同一个数学模型，只是实现方式从「串行采样」变成了「GPU 并行光栅化」。\n\n"
    "3DGS 还有两个精妙的设计我想提一下。一是训练过程中高斯球的数量是动态变化的——"
    "哪里细节不够就克隆或分裂出新球，没用的球就删掉。整个过程全自动，"
    "从最初的几千个 SfM 稀疏点扩展到最终的几百万个球。二是训练只要 6 分钟就能出"
    "有竞争力的结果，51 分钟达到最优——对比 NeRF 的 1-2 天，快了 30 倍以上。\n\n"
    "但 3DGS 也有局限：它仍然需要几十张输入图和相机位姿，和 NeRF 一样依赖 SfM 预处理。"
    "这也是为什么我们的课题要关注单图方法。"
)

# --- 对比 ---
doc.add_heading("四、两篇对比 + 我的理解（约 1-2 分钟）", level=2)

p = doc.add_paragraph()
r = p.add_run("讲稿：")
set_font(r, size=10.5, bold=True, color=ORANGE)

p = doc.add_paragraph()
p.add_run(
    "如果把两篇放在一起看，我觉得它们的核心 tradeoff 是空间换时间，或者说隐式换显式。\n\n"
    "NeRF 用 5MB 的神经网络权重隐式存场景——极致省空间，但每次查询都要过网络，所以慢。"
    "3DGS 用几百 MB 的显式椭球存场景——空间代价更大，但查询就是读内存+代公式，所以极快。\n\n"
    "这个 tradeoff 贯穿了整个领域的发展。后续的论文基本上都在这个坐标系里找位置——"
    "有的优化训练速度，有的优化渲染质量，有的减少输入图片数量，有的去掉相机位姿的依赖。\n\n"
    "对我们课题的启示是：两篇基石论文都要求「多张输入图 + 相机位姿 + 逐场景训练」，"
    "而我们的目标是「单张图直接出 3D」。这意味着我们需要关注：\n"
    "1）深度估计模型（替代多视图几何来获取深度信息）\n"
    "2）前馈式方法（训练一次，对新图直接预测 3D 表示，不需要逐场景重训）\n"
    "3）分层深度表达（MPI 等技术路线，作为 3DGS 之外的备选方案）\n\n"
    "以上是我对这两篇论文的理解。接下来我计划继续精读后续分类的论文，"
    "逐步建立从「基础理论」到「单图方案」的完整知识链。请老师指点。"
)

# --- 讲稿要点卡片 ---
doc.add_page_break()
doc.add_heading("讲稿要点速记卡", level=1)

p = doc.add_paragraph()
r = p.add_run("以下为汇报时可随身参考的关键词卡片，每张卡一个主题。")
set_font(r, size=10, color=GRAY)

cards = [
    ("开场一句话", "我要做的是：一张照片 → 3D场景 → 自由转视角。先读了两篇基石论文打基础。"),
    ("NeRF 一句话", "用神经网络记住空间中每个点的颜色和密度，沿光线采点叠加 = 新视角。5MB存一个场景，但训练要1-2天。"),
    ("NeRF 三个创新", "① 神经网络替代体素网格（省空间3000×） ② 位置编码（sin/cos展开 → 看清细节） ③ 分层采样（粗定位+细精修）"),
    ("3DGS 一句话", "不用神经网络了，直接用几百万个彩色椭球摆在3D空间里，投影到屏幕上叠加。135fps，比NeRF快2000倍。"),
    ("3DGS 三个创新", "① 3D高斯椭球表示 ② 训练中自动增删球（克隆/分裂/删除） ③ 瓦片GPU光栅化器（充分榨干GPU并行能力）"),
    ("NeRF vs 3DGS", "数学等价（都是α混合），实现不同。NeRF=隐式MLP+光线步进（慢但省空间），3DGS=显式椭球+光栅化（快但占空间）。"),
    ("对课题的启示", "两篇都需要几十张图+位姿+逐场景重训 → 我们的单图目标需要：深度估计 + 前馈方法 + MPI备选路线。"),
    ("下一步计划", "继续精读后续论文，按分类建立完整知识链。下一批：深度估计 + 单图3DGS。"),
]

for title, content in cards:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    r = p.add_run(f"▎{title}")
    set_font(r, size=11, bold=True, color=BLUE)
    p = doc.add_paragraph()
    r = p.add_run(content)
    set_font(r, size=10)
    # Add light separator
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(2)

doc.save(OUT)
print(f"Appended script to: {OUT}")
