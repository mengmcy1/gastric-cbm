from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
import subprocess
from PIL import Image

ROOT = Path(r"C:\Users\42193\Desktop\2D转3D")
TEMPLATE = ROOT / "Review of ICCV 2025 Papers on 3D Gaussian Splatting.pptx"
OUT = ROOT / "汇报" / "单张2D图像转浅3D表达_学习进度汇报.pptx"
ASSET = ROOT / "99_过程文件_可删除或归档" / "渲染检查缓存" / "shallow3d_figures"
ASSET.mkdir(parents=True, exist_ok=True)

WHITE = RGBColor(245, 248, 252)
MUTED = RGBColor(170, 184, 204)
CYAN = RGBColor(50, 210, 240)
BLUE = RGBColor(65, 125, 255)
ORANGE = RGBColor(255, 171, 75)
GREEN = RGBColor(90, 225, 175)
RED = RGBColor(255, 110, 120)
FONT = "Microsoft YaHei"

PAPERS = {
    "nerf": ROOT / "papers/01_基础必读_NeRF与3DGS/01_NeRF_Representing_Scenes_as_Neural_Radiance_Fields.pdf",
    "gs": ROOT / "papers/01_基础必读_NeRF与3DGS/02_3D_Gaussian_Splatting_for_Real_Time_Radiance_Field_Rendering.pdf",
    "photo": ROOT / "papers/02_单图3D照片与分层表达/03_3D_Photography_Context_aware_Layered_Depth_Inpainting.pdf",
    "slide": ROOT / "papers/02_单图3D照片与分层表达/04_SLIDE_Soft_Layering_and_Depth_aware_Inpainting.pdf",
    "tmpi": ROOT / "papers/02_单图3D照片与分层表达/05_TMPI_Tiled_Multiplane_Images_for_Practical_3D_Photography.pdf",
    "depth": ROOT / "papers/03_深度估计基础模型/07_Depth_Anything_V2.pdf",
    "depthpro": ROOT / "papers/03_深度估计基础模型/08_Depth_Pro_Sharp_Monocular_Metric_Depth.pdf",
    "sharp": ROOT / "papers/04_单图与前馈3DGS/09_SHARP_Sharp_Monocular_View_Synthesis.pdf",
    "aaa": ROOT / "papers/05_3DGS渲染稳定性与效率/12_AAA_Gaussians_Anti_Aliased_and_Artifact_Free.pdf",
    "ever": ROOT / "papers/05_3DGS渲染稳定性与效率/13_EVER_Exact_Volumetric_Ellipsoid_Rendering.pdf",
}

def render_page(key, page=0):
    # Only use local crops of paper figures, never a full paper page.
    crops = {
        ("nerf", 0): ("nerf-02.png", (55, 35, 720, 210), "nerf_fig1.png"),
        ("nerf", 1): ("nerf-05.png", (70, 35, 710, 225), "nerf_method.png"),
        ("gs", 0): ("gs-01.png", (80, 285, 1030, 525), "gs_fig1.png"),
        ("photo", 0): ("photo-01.png", (75, 350, 1035, 710), "photo_fig1.png"),
        ("photo", 1): ("photo-04.png", (70, 0, 1035, 820), "photo_method_figures.png"),
        ("photo", 2): ("photo-04.png", (70, 0, 1035, 820), "photo_method_figures.png"),
        ("photo", 3): ("photo-04.png", (70, 0, 1035, 820), "photo_method_figures.png"),
        ("slide", 0): ("slide-01.png", (70, 365, 1035, 565), "slide_fig1.png"),
        ("slide", 1): ("slide-01.png", (70, 365, 1035, 565), "slide_fig1.png"),
        ("depth", 0): ("depth-01.png", (180, 410, 930, 790), "depth_fig1.png"),
    }
    item = crops.get((key, page))
    if item is None:
        raise ValueError(f"No figure crop configured for {key} page {page}")
    src_name, box, out_name = item
    out = ASSET / out_name
    src = ASSET.parent / "paper_pages" / src_name
    if not src.exists():
        raise FileNotFoundError(src)
    Image.open(src).crop(box).save(out)
    return out

def formula_crop(name):
    out = ASSET / f"{name}_formula.png"
    if name == "nerf":
        src = ASSET.parent / "paper_pages" / "nerf-05.png"
        Image.open(src).crop((85, 1000, 710, 1090)).save(out)
    elif name == "gs":
        src = Image.open(ASSET.parent / "paper_pages" / "gs-04.png")
        parts = [src.crop(box) for box in [(680, 270, 1000, 315), (680, 390, 1000, 435), (680, 745, 1000, 790)]]
        canvas = Image.new("RGB", (320, sum(p.height for p in parts) + 20), "white")
        y = 5
        for part in parts:
            canvas.paste(part, (0, y)); y += part.height + 5
        canvas.save(out)
    else:
        raise ValueError(name)
    return out

def set_run(run, size=20, color=WHITE, bold=False):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color

FORMAL_REPLACEMENTS = [
    ("一、先明确：我们要解决什么问题", "一、研究问题与任务定义"),
    ("从一张普通照片，生成可以小幅晃动的立体画面", "从单张 RGB 图像生成可交互的浅 3D 表达"),
    ("二、几个核心概念先讲清楚", "二、核心概念与任务定义"),
    ("把后面论文中的术语翻译成直觉", "统一研究对象与技术术语"),
    ("像把相机从正面轻轻移到侧面，再预测这个位置看到的画面。", "通过相机位移预测目标视角图像。"),
    ("本课题更像一个可拆解的工程系统，而不是一个黑盒网络", "本课题采用模块化、可解释的工程架构"),
    ("一句话理解", "方法概述"),
    ("把三维场景“记”进网络权重", "以网络参数隐式编码三维场景"),
    ("为什么它能表达细节", "关键设计"),
    ("NeRF 的体积渲染：像叠加多层半透明玻璃", "NeRF 的体积渲染"),
    ("密度高的点更像物体，密度低的点更像空气", "体积密度用于描述空间占据程度"),
    ("公式直觉", "体积渲染与 Alpha 混合"),
    ("为什么 3DGS 比 NeRF 快", "3DGS 的实时渲染机制"),
    ("从“沿光线查网络”换成“把高斯投到屏幕上”", "从光线采样转向屏幕空间光栅化"),
    ("六、3D Photography：从基础新视角合成走向单图浅 3D", "六、3D Photography：单图 3D 照片"),
    ("显露区域为什么是单图 3D 的核心困难", "显露区域与遮挡补全"),
    ("直接拉伸像素会产生空洞、拉伸、模糊和重影", "直接重投影的典型伪影"),
    ("把一个黑盒问题拆成多个可解释模块", "模块化处理流程"),
    ("深度预处理为什么决定最终观感", "深度预处理与边界检测"),
    ("只补 RGB 会“看起来像”，但不一定“几何上对”", "仅补全 RGB 无法保证几何一致性"),
    ("七、SLIDE：硬分层为什么还不够", "七、SLIDE：软分层与深度感知补全"),
    ("问题直觉", "细结构建模问题"),
    ("可以理解为：这个像素 80% 属于前景，20% 允许背景透出", "软可见性定义：连续权重控制前景与背景合成"),
    ("单图深度估计：它提供几何先验，但不是完整 3D", "单图深度估计：几何先验与适用边界"),
    ("深度图更像“距离排序提示”，不是从一张图恢复真实世界的唯一答案", "深度图提供相对前后关系，无法单独恢复不可见区域"),
    ("本课题中怎么用", "本课题中的使用方式"),
    ("评价指标不能只看 PSNR 和 SSIM", "浅 3D 的评价指标"),
    ("用户真正感受到的是边界是否稳定、视角变化是否自然", "评价应兼顾图像质量、边界稳定性与交互性能"),
    ("谢谢！", "感谢聆听"),
    ("问题与讨论", "交流与讨论"),
    ("把“单张图”变成“可轻微移动的空间照片”", "单图浅 3D 表达的实现路径"),
    ("目标不是一次性解决完整三维重建，而是先做稳定、可解释、可部署的浅 3D", "目标是构建稳定、可解释、可部署的浅 3D 表达"),
]

def formalize(text):
    for old, new in FORMAL_REPLACEMENTS:
        text = text.replace(old, new)
    return text

def add_text(slide, text, x, y, w, h, size=20, color=WHITE, bold=False, align=PP_ALIGN.LEFT):
    text = formalize(text)
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear(); tf.word_wrap = True; tf.margin_left = Pt(2); tf.margin_right = Pt(2)
    tf.vertical_anchor = MSO_ANCHOR.TOP
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line; p.alignment = align; p.space_after = Pt(5)
        for r in p.runs: set_run(r, size, color, bold)
    return box

def add_title(slide, title, subtitle=None):
    ph = next((p for p in slide.placeholders if p.placeholder_format.type == 1), None)
    if ph: ph.text = ""
    add_text(slide, title, .65, .22, 11.8, .48, 26, RGBColor(8, 83, 155), True)
    if subtitle: add_text(slide, subtitle, .68, .72, 11.6, .28, 10, RGBColor(80, 115, 150))

def add_footer(slide, source=None):
    add_text(slide, "单张 2D 图像转浅 3D 表达 · 学习进度汇报", .65, 7.12, 8, .18, 8, MUTED)
    if source: add_text(slide, source, 8.5, 7.12, 4.1, .18, 8, MUTED, align=PP_ALIGN.RIGHT)

def add_card(slide, x, y, w, h, title, body, accent=CYAN, body_size=16):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid(); shape.fill.fore_color.rgb = RGBColor(17, 28, 50)
    shape.line.color.rgb = RGBColor(40, 70, 105)
    add_text(slide, title, x+.18, y+.14, w-.36, .32, 16, accent, True)
    add_text(slide, body, x+.18, y+.55, w-.36, h-.68, body_size, WHITE)
    return shape

def add_image(slide, path, x, y, w, h, caption=None):
    slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w), height=Inches(h))
    if caption: add_text(slide, caption, x, y+h+.05, w, .24, 9, MUTED)

def cover(prs, title, subtitle):
    s = prs.slides.add_slide(prs.slide_layouts[11])
    add_text(s, title, .8, 2.05, 11.5, 1.2, 34, WHITE, True)
    add_text(s, subtitle, .85, 3.45, 10.8, .6, 19, CYAN)
    add_text(s, "从 NeRF / 3DGS 基础，到单图 3D Photography 与移动端浅 3D 路线", .85, 4.25, 11, .35, 14, MUTED)
    add_text(s, "汇报人：郭浩杰\n时间：2026.7.10", .85, 6.15, 4, .65, 13, WHITE)
    add_image(s, render_page("photo", 0), 9.3, 1.15, 3.0, 4.9, "论文原图：3D Photography")
    return s

def inner(prs, title, subtitle=None, source=None):
    s = prs.slides.add_slide(prs.slide_layouts[12])
    add_title(s, title, subtitle); add_footer(s, source); return s

def bullet_text(items): return "\n".join("• " + x for x in items)

def main():
    prs = Presentation(str(TEMPLATE))
    # remove template slides, retain its master/theme/layouts
    while prs.slides:
        rId = prs.slides._sldIdLst[0].rId
        prs.part.drop_rel(rId)
        del prs.slides._sldIdLst[0]

    cover(prs, "单张 2D 图像转浅 3D 表达", "NeRF · 3D Gaussian Splatting · 3D Photography · SLIDE")
    s=inner(prs,"一、先明确：我们要解决什么问题","从一张普通照片，生成可以小幅晃动的立体画面")
    add_card(s,.7,1.35,3.75,4.8,"输入","一张普通 RGB 照片\n没有多视图\n没有相机位姿\n没有真实深度",CYAN,19)
    add_card(s,4.75,1.35,3.75,4.8,"目标","手机轻微左右晃动\n或手指滑动屏幕\n画面产生运动视差\n不追求完整 360° 重建",GREEN,19)
    add_card(s,8.8,1.35,3.75,4.8,"约束","生成阶段可以较复杂\n交互阶段要轻量\n移动端渲染不依赖\n每帧 GPU 网络推理",ORANGE,19)

    s=inner(prs,"二、几个核心概念先讲清楚","把后面论文中的术语翻译成直觉")
    add_card(s,.7,1.35,3.8,4.8,"新视角合成","已知若干照片，生成一个原来没有拍到的视角。\n\n像把相机从正面轻轻移到侧面，再预测这个位置看到的画面。",CYAN,17)
    add_card(s,4.75,1.35,3.8,4.8,"运动视差","相机移动时，近处物体在画面中移动得快，远处物体移动得慢。\n\n这是浅 3D 产生立体感的主要来源。",GREEN,17)
    add_card(s,8.8,1.35,3.8,4.8,"显露区域","相机移动后，原来被前景挡住的背景露出来。\n\n原图里没有这些像素，所以必须补颜色，也要补几何。",ORANGE,17)

    s=inner(prs,"三、整体技术链路","本课题更像一个可拆解的工程系统，而不是一个黑盒网络")
    steps=[("1","普通 RGB","一张照片"),("2","深度估计","每个像素离相机多远"),("3","分层表达","LDI / MPI / 3DGS"),("4","遮挡补全","补看不见的背景"),("5","轻量渲染","移动端交互")]
    for i,(n,t,b) in enumerate(steps):
        x=.55+i*2.5; add_card(s,x,2.1,2.1,2.3,n+"  "+t,b,CYAN if i<3 else ORANGE,16)
        if i<4: add_text(s,"→",x+2.12,2.9,.35,.4,26,CYAN,True,PP_ALIGN.CENTER)
    add_text(s,"关键判断：把耗时的生成放到前处理阶段，把用户交互放到显式表达的渲染阶段。",1.0,5.35,11,.5,20,WHITE,True,PP_ALIGN.CENTER)

    s=inner(prs,"四、NeRF：用网络隐式表示整个场景","Neural Radiance Fields，神经辐射场")
    add_image(s,render_page("nerf",0),.7,1.2,5.6,4.9,"原论文首页/方法概览")
    add_card(s,6.65,1.35,5.8,4.45,"一句话理解","NeRF 用一个多层感知机 MLP 把三维场景“记”进网络权重。\n\n输入：3D 位置 + 观察方向\n输出：颜色 RGB + 体积密度 σ\n\n沿相机光线采样很多点，再把这些点的颜色和密度累积成像素。",CYAN,17)
    add_footer(s,"来源：NeRF，ECCV 2020，Fig.2 / Eq.(1)，p.5")

    s=inner(prs,"NeRF 的三个关键技术点","为什么它能表达细节")
    add_card(s,.7,1.25,3.75,4.9,"① MLP 表示","不用巨大的体素网格，而是用网络参数存储连续场景。\n\n优点：表达连续、存储紧凑。\n代价：每个采样点都要查询网络。",CYAN,16)
    add_card(s,4.75,1.25,3.75,4.9,"② 位置编码","把坐标展开为不同频率的 sin/cos。\n\n相当于给网络提供高频基底，帮助它学习细线、纹理和边缘。",GREEN,16)
    add_card(s,8.8,1.25,3.75,4.9,"③ 分层采样","粗网络先定位物体，细网络在高密度区域多采样。\n\n减少把算力浪费在空气中的问题。",ORANGE,16)

    s=inner(prs,"NeRF 的体积渲染：像叠加多层半透明玻璃","密度高的点更像物体，密度低的点更像空气")
    add_image(s,render_page("nerf",1),.7,1.2,6.0,3.75,"NeRF 原论文 Fig.2，p.5")
    add_image(s,formula_crop("nerf"),.85,5.05,5.7,1.25,"NeRF 原论文 Eq.(1)，p.5")
    add_card(s,7.0,1.45,5.4,4.35,"体积渲染与 Alpha 混合","沿相机光线对采样点进行颜色与透射率累积；前方高密度区域会降低后续采样点的可见性。\n\n该渲染形式与 3DGS 的 Alpha 混合具有一致的颜色累积结构。",GREEN,17)

    s=inner(prs,"NeRF 的局限：质量高，但不适合本课题直接部署","它建立了新视角合成的基本框架，却不是移动端工程答案")
    add_card(s,.7,1.35,3.75,4.6,"输入要求","通常需要几十张照片\n每张图还要有相机位姿\n位姿常由 COLMAP 估计",RED,19)
    add_card(s,4.75,1.35,3.75,4.6,"生成方式","每个场景单独优化\n训练时间长\n换一张新图不能直接推理",ORANGE,19)
    add_card(s,8.8,1.35,3.75,4.6,"渲染瓶颈","每条光线采样很多点\n每个点都要过 MLP\n交互速度受限",CYAN,19)

    s=inner(prs,"五、3D Gaussian Splatting：用显式高斯表示场景","3DGS：Three-Dimensional Gaussian Splatting")
    add_image(s,render_page("gs",0),.7,1.15,5.55,5.05,"原论文：高斯场景表示与渲染")
    add_card(s,6.6,1.3,5.8,4.6,"方法概述","场景由大量具有位置、尺度、旋转、不透明度和球谐颜色参数的三维高斯基元组成。\n\n渲染阶段将高斯基元投影到屏幕空间，并进行深度排序与 Alpha 混合。",CYAN,17)
    add_footer(s,"来源：3D Gaussian Splatting，SIGGRAPH 2023，Fig.1 / Eq.(4–6)")

    s=inner(prs,"为什么 3DGS 比 NeRF 快","从“沿光线查网络”换成“把高斯投到屏幕上”")
    add_card(s,.7,1.45,5.55,4.5,"NeRF：光线步进","每根光线采样约 192 个点\n每个点查询 MLP\n串行累积颜色和密度\n\n优点：连续表达\n缺点：渲染慢",ORANGE,18)
    add_text(s,"VS",6.25,3.0,.7,.5,28,CYAN,True,PP_ALIGN.CENTER)
    add_card(s,7.05,1.45,5.35,4.5,"3DGS：屏幕空间光栅化","高斯投影到屏幕\n按深度排序\nGPU 并行 Alpha 混合\n\n优点：实时交互\n缺点：显式存储更大",GREEN,18)

    s=inner(prs,"3DGS 的三个关键技术点","表示、增密、光栅化三件事共同决定效果")
    add_card(s,.7,1.25,3.75,4.95,"① 高斯椭球","协方差采用缩放与旋转参数化，保证优化过程中的正定性，并通过投影得到二维椭圆斑。",CYAN,16)
    add_image(s,formula_crop("gs"),.92,4.85,3.3,.95,"3DGS 原论文 Eq.(4–6)，p.4")
    add_card(s,4.75,1.25,3.75,4.95,"② 自适应密度控制","太小但梯度大：克隆\n太大且梯度大：分裂\n贡献很低：删除\n\n训练中动态增删高斯，逐步补足细节。",GREEN,16)
    add_card(s,8.8,1.25,3.75,4.95,"③ 瓦片光栅化","把屏幕切成小瓦片。\n\n每个瓦片内部独立排序和混合，利用 GPU 并行，提高渲染效率。",ORANGE,16)

    s=inner(prs,"NeRF 与 3DGS：共同基础与关键取舍")
    add_card(s,.7,1.25,3.75,4.9,"共同点","都解决新视角合成\n都需要多视图约束\n都需要相机位姿\n都要处理颜色与密度/透明度",CYAN,18)
    add_card(s,4.75,1.25,3.75,4.9,"NeRF","隐式 MLP\n连续表达\n存储相对紧凑\n训练和渲染慢",ORANGE,18)
    add_card(s,8.8,1.25,3.75,4.9,"3DGS","显式高斯\n容易 GPU 并行\n渲染实时\n内存和跨视角稳定性压力更大",GREEN,18)
    add_text(s,"结论：它们解决了“多图 → 新视角”，但还没有解决“单图 → 浅 3D”。",1.0,6.35,11,.35,18,WHITE,True,PP_ALIGN.CENTER)

    s=inner(prs,"六、3D Photography：从基础新视角合成走向单图浅 3D","输入是一张 RGB-D 图像，目标是小幅晃动时产生立体感")
    add_image(s,render_page("photo",1),.7,1.15,6.0,5.0,"原论文：3D Photography 方法图/显露区域")
    add_card(s,7.0,1.35,5.3,4.55,"核心问题","如果相机向右移动，前景左侧原来被挡住的背景会露出来。\n\n这些像素在原图中根本不存在，因此不能只做深度重投影，还要进行遮挡区域补全。",ORANGE,18)
    add_footer(s,"来源：3D Photography，CVPR 2020，Fig.1–5，pp.1–3")

    s=inner(prs,"显露区域为什么是单图 3D 的核心困难","直接拉伸像素会产生空洞、拉伸、模糊和重影")
    add_card(s,.7,1.3,3.75,4.8,"直接重投影","有深度就能把像素投到新位置。\n\n但被挡住的区域没有像素可投，结果出现空洞。",RED,18)
    add_card(s,4.75,1.3,3.75,4.8,"简单拉伸","把周围像素拉过去填洞。\n\n短距离看似有效，但会产生纹理拉伸、边缘撕裂和纸片感。",ORANGE,18)
    add_card(s,8.8,1.3,3.75,4.8,"深度感知补全","只补背景侧，并同时预测颜色和深度。\n\n让补出来的内容既像背景，也能参与正确的几何渲染。",GREEN,18)

    s=inner(prs,"3D Photography 的方法主线","把一个黑盒问题拆成多个可解释模块")
    flow=["RGB-D 输入","深度预处理","构建 LDI","检测深度边界","断开前景/背景连接","补颜色 + 补深度","转成 textured mesh","快速渲染"]
    for i,t in enumerate(flow):
        x=.55+(i%4)*3.05; y=1.45+(i//4)*2.5
        add_card(s,x,y,2.55,1.55,str(i+1).zfill(2),t,CYAN if i<4 else GREEN,14)
        if i in [3]: add_text(s,"↓",6.05,3.03,.3,.3,24,CYAN,True,PP_ALIGN.CENTER)
    add_text(s,"重要启发：生成阶段可以复杂，但交互阶段只需渲染已生成的显式表达。",1.0,6.25,11,.35,18,WHITE,True,PP_ALIGN.CENTER)

    s=inner(prs,"LDI：分层深度图","Layered Depth Image 允许同一个图像位置保存多层颜色和深度")
    add_image(s,render_page("photo",2),.7,1.2,5.8,4.9,"原论文中的 LDI / 分层示意")
    add_card(s,6.8,1.35,5.5,4.5,"普通 RGB-D vs LDI","普通 RGB-D：一个像素只有一个颜色和一个深度。\n\nLDI：同一个位置可以存前景、背景等多个样本，并保留像素之间的连接关系。\n\n这样能在遮挡边界附近保存“被挡住的背景线索”。",CYAN,17)

    s=inner(prs,"深度预处理为什么决定最终观感","浅 3D 最容易出错的位置不是平坦区域，而是前景—背景边界")
    add_card(s,.7,1.3,3.75,4.8,"转换到视差空间","视差约等于 1 / 深度。\n\n近处物体的视差变化更敏感，更符合小幅相机移动时的视觉变化。",CYAN,17)
    add_card(s,4.75,1.3,3.75,4.8,"锐化边界","使用双边中值滤波，降低噪声，同时尽量保留深度突变。\n\n避免前景与背景被平滑成斜坡。",GREEN,17)
    add_card(s,8.8,1.3,3.75,4.8,"检测不连续","根据相邻像素视差差异检测边界，再用连通域分析连接成稳定的深度边。",ORANGE,17)

    s=inner(prs,"颜色和深度必须一起补全","只补 RGB 会“看起来像”，但不一定“几何上对”")
    add_image(s,render_page("photo",3),.7,1.15,5.5,5.0,"原论文：上下文感知补全结果")
    add_card(s,6.55,1.3,5.75,4.55,"结构引导的补全顺序","① 先补 depth edge：先画出可能的结构边界\n② 再补 color：生成背景纹理\n③ 再补 depth：生成背景几何\n④ 迭代处理新的深度边界\n\n共同的结构线索让 RGB 与 depth 更一致。",GREEN,16)

    s=inner(prs,"3D Photography 对本课题的直接启发","四个模块可以直接转成项目技术路线")
    add_card(s,.7,1.25,2.75,4.9,"深度估计","普通 RGB → 相对深度/视差\n\n候选：Depth Anything V2、Depth Pro",CYAN,16)
    add_card(s,3.7,1.25,2.75,4.9,"分层表示","LDI、简化 Mesh、TMPI、轻量 3DGS\n\n按移动端成本取舍",GREEN,16)
    add_card(s,6.7,1.25,2.75,4.9,"遮挡补全","围绕深度不连续边界\n\n同时补颜色和深度",ORANGE,16)
    add_card(s,9.7,1.25,2.75,4.9,"实时渲染","生成一次，交互时只改变相机\n\n不重复运行大型网络",BLUE,16)

    s=inner(prs,"七、SLIDE：硬分层为什么还不够","细线、头发、树枝和半透明边缘不是简单的前景/背景二值切割")
    add_image(s,render_page("slide",0),.7,1.15,5.7,5.0,"原论文：SLIDE 方法/软分层示意")
    add_card(s,6.75,1.35,5.45,4.45,"问题直觉","一个头发像素可能同时包含头发颜色和背景颜色；桥索、栏杆、树枝也可能只有几个像素宽。\n\n硬分层容易造成断裂、缺失、撕裂和纸片感。SLIDE 用连续的软可见性表达边界。",ORANGE,17)
    add_footer(s,"来源：SLIDE，CVPR 2021，Fig.1，p.1")

    s=inner(prs,"Soft Layering：用连续可见性替代硬切割","可以理解为：这个像素 80% 属于前景，20% 允许背景透出")
    add_image(s,render_page("slide",1),.7,1.15,5.8,5.0,"原论文中的 soft visibility / layering")
    add_card(s,6.8,1.35,5.45,4.45,"两个关键量","前景软可见性 A：前景应该保留多少。\n\n软显露区域 S：背景层哪些位置需要补全。\n\nA 管前景如何显示，S 管背景哪里需要补，二者共同避免黑洞和拉伸三角形。",GREEN,17)

    s=inner(prs,"SLIDE 的轻量流程","RGB → disparity → soft layering → RGBD inpainting → layered rendering")
    add_card(s,.7,1.35,3.75,4.55,"输入与深度","普通 RGB 图像\nMiDaS v2 预测 normalized disparity\n轻微平滑 + max-pool\n保留细结构的近远关系",CYAN,16)
    add_card(s,4.75,1.35,3.75,4.55,"两层表达","前景层：RGB + disparity + soft visibility\n\n背景层：补全 RGB + 补全 disparity\n\n比复杂 LDI 更简单、更快。",GREEN,16)
    add_card(s,8.8,1.35,3.75,4.55,"分层渲染","前景和背景分别反投影为 mesh\n\n新视角时只做投影和 Alpha 合成\n\n生成和交互彻底分离。",ORANGE,16)

    s=inner(prs,"3D Photography 与 SLIDE：互补而不是互相替代")
    add_card(s,.7,1.25,5.55,4.9,"3D Photography / LDI","更强调遮挡关系和局部结构补全\n\n表达层次更细，可处理多层深度\n\n流程较复杂，硬分层对细结构不够友好",CYAN,18)
    add_card(s,7.0,1.25,5.55,4.9,"SLIDE / Soft two-layer","更强调细结构边缘和软可见性\n\n一次前向，速度更快，适合移动端\n\n两层表示对复杂多层遮挡能力有限",GREEN,18)

    s=inner(prs,"八、从当前论文自然过渡到后续路线","后续论文回答的是：如何更快、更轻、更通用")
    add_card(s,.7,1.2,3.75,4.95,"TMPI","Tiled Multiplane Images\n\n把规则 MPI 切成局部 tile，每块只保留少量自适应深度平面。\n\n目标：减少 MPI 冗余，更适合实际 3D photography。",CYAN,16)
    add_card(s,4.75,1.2,3.75,4.95,"Depth Anything V2 / Depth Pro","单图深度是整个系统的几何入口。\n\nDepth Anything V2：通用泛化\nDepth Pro：高分辨率、锐利边界、度量深度",GREEN,16)
    add_card(s,8.8,1.2,3.75,4.95,"SHARP / 前馈 3DGS","单次前馈预测 3D Gaussian 参数。\n\n更接近“输入一张图，快速生成表达”的目标，但仍要面对不可见区域和复杂材质。",ORANGE,16)

    s=inner(prs,"单图深度估计：它提供几何先验，但不是完整 3D","深度图更像“距离排序提示”，不是从一张图恢复真实世界的唯一答案")
    add_image(s,render_page("depth",0),.7,1.1,5.8,5.1,"原论文：Depth Anything V2")
    add_card(s,6.8,1.35,5.45,4.5,"本课题中怎么用","先用单目深度模型得到近远层次。\n\n再进行边界增强、分层和遮挡补全。\n\n重点检查：桥索、栏杆、头发、透明/反光物体、无纹理墙面。\n\n不能只看整体深度误差，要看新视角下是否撕裂和拉伸。",GREEN,16)

    s=inner(prs,"九、面向移动端的推荐方案","先做可解释的浅 3D 原型，再逐步替换表达形式")
    stages=[("阶段 1","RGB → 深度 → 点云/简化 Mesh","验证运动视差是否成立"),("阶段 2","边界处理 + 背景补全 + 两层 Mesh","解决空洞、拉伸和细结构"),("阶段 3","LDI / TMPI / 轻量 3DGS 对比","比较质量、内存和渲染速度"),("阶段 4","Android / OpenGL ES / Vulkan","把生成与交互渲染拆开")]
    for i,(a,b,c) in enumerate(stages):
        y=1.15+i*1.35; add_card(s,.8,y,2.0,1.05,a,b,CYAN if i<2 else ORANGE,13); add_text(s,b,3.05,y+.1,4.2,.4,18,WHITE,True); add_text(s,c,7.55,y+.1,4.5,.45,14,MUTED)

    s=inner(prs,"评价指标不能只看 PSNR 和 SSIM","用户真正感受到的是边界是否稳定、视角变化是否自然")
    add_card(s,.7,1.25,3.75,4.85,"画面质量","PSNR / SSIM\nLPIPS 感知距离\n纹理是否自然\n是否过度平滑",CYAN,18)
    add_card(s,4.75,1.25,3.75,4.85,"几何稳定性","前景背景边界\n显露区域是否自然\n是否产生重影/撕裂\n细线是否断裂",GREEN,18)
    add_card(s,8.8,1.25,3.75,4.85,"工程指标","单图生成时间\n移动端渲染 FPS\n峰值内存\n模型是否需要 GPU 网络推理",ORANGE,18)

    s=inner(prs,"十、本阶段的结论","从论文学习转向可验证的工程路线")
    add_card(s,.7,1.25,5.55,4.9,"已经建立的认识","NeRF 让我理解连续场和体积渲染。\n\n3DGS 让我理解显式高斯和实时光栅化。\n\n3D Photography 让我看到单图浅 3D 的核心是显露区域补全。\n\nSLIDE 让我认识到细结构需要软可见性。",CYAN,17)
    add_card(s,7.0,1.25,5.55,4.9,"下一步要验证的问题","单目深度在本课题图片上的边界质量如何？\n\nLDI、两层 Mesh、TMPI、轻量 3DGS 哪个最适合移动端？\n\n颜色补全和深度补全如何保持一致？\n\n复杂材质和大视角变化的失败边界在哪里？",ORANGE,17)

    s=inner(prs,"结束：把“单张图”变成“可轻微移动的空间照片”","目标不是一次性解决完整三维重建，而是先做稳定、可解释、可部署的浅 3D")
    add_text(s,"普通照片\n↓\n可靠的深度层次\n↓\n合理的遮挡补全\n↓\n轻量显式表达\n↓\n移动端运动视差",2.0,1.35,4.6,4.7,25,RGBColor(8, 83, 155),True,PP_ALIGN.CENTER)
    add_text(s,"谢谢！",8.0,2.55,3.2,.7,34,CYAN,True,PP_ALIGN.CENTER)
    add_text(s,"问题与讨论",8.0,3.5,3.2,.4,20,MUTED,False,PP_ALIGN.CENTER)

    prs.save(str(OUT))
    print(OUT)

if __name__ == "__main__": main()
