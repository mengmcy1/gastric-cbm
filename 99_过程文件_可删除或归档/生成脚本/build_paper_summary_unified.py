# -*- coding: utf-8 -*-
"""生成统一《论文总结.docx》
以带读笔记详细内容为主体，按统一结构（动机/方法/概念/创新/公式/实验/优劣/与课题关系）组织。
首版收录：NeRF、3D Gaussian Splatting。后续读完一篇追加一篇。
"""
import os
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "论文总结.docx")

BLUE = RGBColor(46, 116, 181)
DARK = RGBColor(31, 77, 120)
GRAY = RGBColor(95, 99, 104)
GREEN = RGBColor(56, 118, 29)
ORANGE = RGBColor(191, 100, 20)
LIGHT = "E8EEF5"


def font(run, name="Microsoft YaHei", size=None, bold=None, italic=None, color=None):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), name)
    if size: run.font.size = Pt(size)
    if bold is not None: run.bold = bold
    if italic is not None: run.italic = italic
    if color: run.font.color.rgb = color


def h1(doc, text):
    p = doc.add_paragraph()
    p.space_before = Pt(6)
    r = p.add_run(text)
    font(r, size=16, bold=True, color=BLUE)
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    return p


def h2(doc, text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    font(r, size=12.5, bold=True, color=DARK)
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    return p


def para(doc, text, size=10.5, color=None):
    p = doc.add_paragraph()
    r = p.add_run(text)
    font(r, size=size, color=color)
    p.paragraph_format.space_after = Pt(4)
    return p


def bullet(doc, text, size=10.5):
    p = doc.add_paragraph(style=None)
    p.paragraph_format.left_indent = Pt(14)
    r = p.add_run("• ")
    font(r, size=size, bold=True, color=BLUE)
    r2 = p.add_run(text)
    font(r2, size=size)
    p.paragraph_format.space_after = Pt(2)
    return p


def term(doc, name, expl, size=10.5):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Pt(14)
    r = p.add_run("• " + name + "：")
    font(r, size=size, bold=True, color=DARK)
    r2 = p.add_run(expl)
    font(r2, size=size)
    p.paragraph_format.space_after = Pt(2)
    return p


def mono(doc, text, size=9.5):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Pt(14)
    for line in text.split("\n"):
        r = p.add_run(line + "\n")
        font(r, name="Consolas", size=size, color=GRAY)
    p.paragraph_format.space_after = Pt(4)
    return p


def info_line(doc, text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    font(r, size=9.5, color=GRAY, italic=True)
    p.paragraph_format.space_after = Pt(6)
    return p


def make_table(doc, headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    for i, hh in enumerate(headers):
        c = t.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(hh)
        font(r, size=9.5, bold=True, color=DARK)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(val)
            font(r, size=9.5)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


# ============ 生成文档 ============
doc = Document()
doc.styles["Normal"].font.name = "Microsoft YaHei"
doc.styles["Normal"]._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")

# 封面标题
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("论文总结")
font(r, size=24, bold=True, color=BLUE)
p2 = doc.add_paragraph()
p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p2.add_run("2D 转浅 3D 项目 · 文献学习记录")
font(r, size=12, color=GRAY)

para(doc, "")
para(doc, "阅读顺序：基础必读 → 深度估计 → 分层表达 → 前馈 3DGS → 渲染优化 → 稀疏视图", color=GRAY)
para(doc, "研究方向：单张 2D 图像 → 浅 3D 表达 → 新视角合成", color=GRAY)
para(doc, "本文件为读完整篇论文后的详细总结，每篇按统一结构：动机与问题 / 方法核心 / 关键概念 / 技术创新 / 核心公式与设计 / 关键实验结论 / 优势与局限 / 与本课题的关系。", color=GRAY)

# ==================== 1. NeRF ====================
h1(doc, "1. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis")
info_line(doc, "作者：Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng  |  UC Berkeley + Google Research  |  ECCV 2020  |  重要性 ★★★★★ 领域基石  |  难度：中等偏高  |  开源：github.com/bmild/nerf")

h2(doc, "1.1 动机与核心问题")
para(doc, "核心问题：给定一组从不同角度拍摄的照片 + 每张照片对应的相机位姿，能否生成任意新视角的照片（新视角合成）？")
para(doc, "此前方法的两个极端：")
bullet(doc, "体素网格方法：3D 空间切方块存储，分辨率翻倍 → 内存 ×8，无法表示高精度场景。")
bullet(doc, "神经隐式方法（SRN、NV）：用 MLP 隐式表示场景，但渲染质量远远不够。")
para(doc, "NeRF 的目标：用神经网络实现连续、高精度的场景表示，同时克服体素网格的存储灾难。")

h2(doc, "1.2 方法核心")
para(doc, "一句话：用 MLP 记住空间中每个点的「颜色」和「实在程度（密度）」，渲染时沿每根光线采点、按体积渲染公式叠加。整个过程可微分，用梯度下降直接优化。")
para(doc, "网络结构：")
mono(doc, "(x,y,z) → 位置编码(L=10, 60维) → [8层FC, 256维, ReLU] → 密度 σ + 256维特征向量\n特征向量 + 观察方向 d → 位置编码(L=4, 24维) → [1层FC, 128维] → RGB颜色 c")
para(doc, "关键设计：密度 σ 只取决于位置（一个点是不是固体跟你怎么看无关），颜色 c 取决于位置 + 方向（镜面反射随视角变化）——这种分离保证了多视图一致性。")

h2(doc, "1.3 关键概念")
term(doc, "新视角合成（Novel View Synthesis）", "已知若干视角的照片，合成任意新相机位置看到的图像。")
term(doc, "辐射场（Radiance Field）", "空间中每个点、每个方向上的颜色与密度分布。NeRF 用 MLP 来拟合它。")
term(doc, "体积渲染（Volume Rendering）", "沿光线把途经各点的颜色按密度和透射率加权累加，得到像素颜色。")
term(doc, "谱偏置（Spectral Bias）", "神经网络天然偏向学习低频（平滑）函数，导致直接输入坐标会渲染模糊。")

h2(doc, "1.4 技术创新")
bullet(doc, "① 连续场景表示：场景编码进神经网络权重（~5MB），替代体素网格（~15GB），约 3000× 压缩，且可查询任意精度坐标。")
bullet(doc, "② 位置编码（PE）：把坐标经 sin/cos 映射到高维——γ(p)=[sin(p),cos(p),sin(2p),cos(2p),...]。本质是把不同频率的傅里叶基底提前摆好，MLP 只需学加权组合。类比：给网络配了一副高清眼镜。空间坐标 L=10，方向坐标 L=4。")
bullet(doc, "③ 分层采样（粗/细网络）：粗网络均匀采 64 点找到物体大致位置，细网络在高密度区附近再采 128 点精细渲染。类比：先扫一眼房间找人，再盯着仔细看。")

h2(doc, "1.5 核心公式与设计")
para(doc, "体积渲染积分（连续形式）：")
mono(doc, "C(r) = ∫[tn→tf] T(t)·σ(r(t))·c(r(t),d) dt\nT(t) = exp(−∫[tn→t] σ(r(s)) ds)   ← 透射率：光线走到 t 还剩多少没被吸收")
para(doc, "离散近似（实际代码使用）：")
mono(doc, "Ĉ(r) = Σ Tᵢ·αᵢ·cᵢ\nαᵢ = 1 − exp(−σᵢ·δᵢ)     ← 第 i 段不透明度\nTᵢ = Π[j<i] (1−αⱼ)        ← 累积透射率，T₁=1\nδᵢ = t_{i+1} − t_i         ← 采样间距")
para(doc, "物理直觉：透过 N 层半透明彩色玻璃看东西，每层贡献一点颜色，越后面的层被前面削弱越多。最前面贡献最大（T₁=1）。")

h2(doc, "1.6 关键实验结论")
bullet(doc, "位置编码最重要：去掉 PE → PSNR 从 31.01 降到 28.77（降 2.24），画面明显模糊。")
bullet(doc, "视角依赖次之：去掉视角依赖 → PSNR 降 3.35，无法表示镜面反射。")
bullet(doc, "分层采样：去掉 → PSNR 降约 0.95，主要影响效率。")
bullet(doc, "极致空间效率：单场景 ~5MB vs 对比方法 ~15GB，约 3000×。")
bullet(doc, "仅用 25 张输入图，效果已超过其他方法用 100 张。")

h2(doc, "1.7 优势与局限")
para(doc, "✅ 优势：", color=GREEN)
bullet(doc, "首次实现连续神经场景表示，能以极高保真度渲染复杂真实场景的新视角。")
bullet(doc, "位置编码巧妙解决 MLP 低频偏置，使网络能学到高频细节。")
bullet(doc, "极致空间效率（5MB 存一个场景）；可微分体积渲染管线，端到端训练，无需 3D 真值。")
para(doc, "⚠️ 局限：", color=ORANGE)
bullet(doc, "每个场景需单独训练 1-2 天（V100），无法泛化到新场景。")
bullet(doc, "需要几十张输入照片 + 精确相机位姿（依赖 COLMAP SfM）。")
bullet(doc, "渲染慢（~0.07 fps），每帧约 1.2 亿次 MLP 查询；对位姿精度敏感。")

h2(doc, "1.8 与本课题的关系")
para(doc, "NeRF 是「新视角合成」的奠基工作，提供了整个领域的数学框架和渲染范式。")
bullet(doc, "体积渲染公式：后续所有神经渲染方法（含 3DGS）都沿用这套公式或其变体，必掌握。")
bullet(doc, "密度与颜色分离：为后续深度估计驱动的 3D 重建提供理论基础。")
bullet(doc, "局限即启示：需多图 + 位姿 + 慢 → 本项目关注的前馈方法（pixelSplat、AnySplat）和单目深度估计正是为解决这些问题。")

# ==================== 2. 3DGS ====================
h1(doc, "2. 3D Gaussian Splatting for Real-Time Radiance Field Rendering")
info_line(doc, "作者：Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis  |  Inria（法国）+ Max-Planck-Institut  |  SIGGRAPH 2023  |  重要性 ★★★★★ 与 NeRF 并列基石  |  难度：中等  |  开源：repo-sam.inria.fr/fungraph/3d-gaussian-splatting/")

h2(doc, "2.1 动机与核心问题")
para(doc, "NeRF 的核心矛盾：质量好但太慢，快速方法快但质量差。在 1080p 下没有方法能实时渲染（≥30 fps）。3DGS 的目标：兼得高质量与实时渲染。")

h2(doc, "2.2 方法核心")
para(doc, "NeRF 用 MLP 隐式存场景，每次查询都要过网络 → 慢。3DGS 用几百万个彩色椭球（3D 高斯）显式存场景，直接投影到屏幕上 α 混合叠加 → 极快（135+ fps）。两者图像形成模型数学等价（都是 α 混合），实现方式完全不同。")
para(doc, "每个 3D 高斯球（软椭球，中心最密、越远越稀薄）存约 59 个参数：")
term(doc, "位置 μ（3 个浮点）", "球中心在哪。")
term(doc, "缩放 s（3 个浮点）", "三个轴方向各拉伸多长。")
term(doc, "旋转 q（四元数，4 个浮点）", "椭球朝向。")
term(doc, "不透明度 α（1 个浮点）", "0=透明，1=不透明。")
term(doc, "球谐系数 SH（48 个浮点）", "从不同方向看的颜色，替代 NeRF 的视角相关 MLP。")

h2(doc, "2.3 关键概念")
term(doc, "3D 高斯（3D Gaussian）", "一个没有硬边界的软椭球，与 NeRF 的体积密度精神一致：软、连续、可微。")
term(doc, "泼溅（Splatting）", "把 3D 椭球投影到 2D 屏幕变成椭圆斑（splat）再叠加，与光线步进相反的正向渲染思路。")
term(doc, "SfM（Structure from Motion）", "从多张图恢复稀疏 3D 点云和相机位姿，3DGS 用它做初始化。")
term(doc, "球谐函数（Spherical Harmonics）", "用一组基函数表示「颜色随方向的变化」，紧凑地表达视角相关外观。")

h2(doc, "2.4 技术创新")
bullet(doc, "① 3D 高斯作为场景表示：协方差 Σ = R S Sᵀ Rᵀ 分解为缩放与旋转分别优化，保证 Σ 永远合法。投影到屏幕：Σ' = J W Σ Wᵀ Jᵀ。")
bullet(doc, "② 自适应密度控制：欠重建（小球+大梯度）→ 克隆；过重建（大球+大梯度）→ 分裂（÷1.6）；近透明球 → 删除；每 3000 次迭代把 α 压到接近 0，有用的靠梯度涨回来。")
bullet(doc, "③ 瓦片式快速光栅化器：视锥裁剪 → 屏幕分 16×16 瓦片 → GPU 按深度+瓦片 ID 排序 → 每瓦片并行从远到近叠加、α 饱和即停 → 反向传播倒序遍历算梯度。")

h2(doc, "2.5 核心公式与设计")
para(doc, "渲染叠加公式（与 NeRF 数学等价）：")
mono(doc, "Color = Σ Tᵢ·αᵢ·cᵢ    （α 混合，从远到近）")
para(doc, "协方差分解：Σ = R S Sᵀ Rᵀ（S 缩放、R 旋转）；投影：Σ' = J W Σ Wᵀ Jᵀ（W 视图变换，J 投影雅可比近似）。")

h2(doc, "2.6 关键实验结论")
make_table(doc, ["维度", "NeRF", "3DGS"], [
    ["场景表示", "MLP 隐式函数", "百万级显式椭球"],
    ["数学公式", "体积渲染积分", "α 混合（数学等价）"],
    ["渲染方式", "光线步进（串行）", "瓦片光栅化（GPU并行）"],
    ["训练时间", "1-2 天", "6 分钟 - 51 分钟"],
    ["渲染速度", "~0.07 fps", "135+ fps"],
    ["存储大小", "~5 MB", "~几百 MB"],
    ["需要初始化？", "不需要", "需要 SfM 稀疏点云"],
])

h2(doc, "2.7 优势与局限")
para(doc, "✅ 优势：", color=GREEN)
bullet(doc, "实现神经渲染的实时化突破（135+ fps 高质量渲染），训练也远快于 NeRF。")
bullet(doc, "显式表示便于编辑、与传统图形管线结合。")
para(doc, "⚠️ 局限：", color=ORANGE)
bullet(doc, "仍需几十张输入图 + SfM 位姿 + 逐场景训练。")
bullet(doc, "存储较大（几百 MB）；对初始化点云质量有依赖。")

h2(doc, "2.8 与本课题的关系")
para(doc, "3DGS 是当前主流的显式神经渲染表示，但与 NeRF 一样需要多图 + 位姿 + 逐场景训练。本项目关注的论文正是要解决这些限制：")
bullet(doc, "前馈式 3DGS（pixelSplat / AnySplat，04 类）：输入 1-2 张图直接预测高斯参数，免逐场景重训。")
bullet(doc, "单图深度估计（Depth Anything / Depth Pro，03 类）：为 3DGS 提供单图几何初始化，替代多视图 SfM。")
bullet(doc, "分层深度表达（3D Photo / SLIDE / TMPI，02 类）：3DGS 之外处理单图的另一条技术路线。")
bullet(doc, "渲染稳定与效率（05 类）、训练与稀疏视图（06 类）：分别优化 3DGS 的质量、速度、位姿依赖。")

# ==================== 3. 3D Photography (Context-aware LDI Inpainting) ====================
h1(doc, "3. 3D Photography using Context-aware Layered Depth Inpainting")
info_line(doc, "作者：Meng-Li Shih, Shih-Yang Su, Johannes Kopf, Jia-Bin Huang  |  Virginia Tech + Facebook  |  CVPR 2020  |  重要性 ★★★★★ 与本课题相关度最高  |  难度：中等  |  开源：shihmengli.github.io/3D-Photo-Inpainting  |  本地：papers/02_单图3D照片与分层表达/03_3D_Photography_Context_aware_Layered_Depth_Inpainting.pdf")

h2(doc, "3.1 动机与核心问题")
para(doc, "输入一张 RGB-D 图像（普通照片 + 深度图），生成一张能小范围晃动视角、看到立体视差的「3D 照片」。若只有普通 2D 照片，则需先用单目深度估计得到深度图。")
para(doc, "核心难点是去遮挡（disocclusion）：视角移动后，原本被前景挡住的背景区域会露出来，但这部分内容原图不存在。此前方案各有缺陷：")
bullet(doc, "朴素深度变形（直接拽深度图）：产生黑洞或橡皮筋式拉伸。")
bullet(doc, "Facebook 3D Photo 的扩散填充：太模糊。")
bullet(doc, "MPI 多平面图像：斜面上有台阶伪影，且存储冗余。")
para(doc, "本文思路：LDI + 沿深度边迭代局部补全 → 转 Mesh。")

h2(doc, "3.2 方法核心")
para(doc, "整体管线（重活在生成阶段做完，渲染阶段极轻）：")
mono(doc, "RGB-D 输入\n → 3.1 预处理：修深度边界，找出深度边\n → 3.2 切开分区：沿深度边剪开前景/背景，圈出要补的洞(synthesis)+参考(context)\n → 3.3 补全：三网络接力(边缘→颜色→深度)，共享边缘保对齐，迭代到无新边\n → 3.4 转 Mesh：融合回 LDI → 带纹理三角网格 → 边缘设备零推理渲染")
para(doc, "预处理（3.1）：深度转视差并归一化 → 双边中值滤波(7×7, spatial=4.0, intensity=0.5)锐化保边 → 视差阈值检测不连续 → 连通分量连成 linked depth edges → 删短边(<10 像素) → RGB-D 初始化为 LDI(每像素一层, 4-连通)。关键结论：浅 3D 补全应围绕深度断层展开——那是视角移动后最易空洞/拉伸/撕裂的位置。")
para(doc, "切开分区（3.2）：一次处理一条深度边 → 断开跨边连接，形成前景轮廓(绿,不补)与背景轮廓(红,需补) → 从背景轮廓向被遮挡方向洪水填充 40 步 = synthesis region(待填空像素)；沿 LDI 连接向已知方向走 100 步 = context region(给 CNN 看的已知像素) → synthesis 朝遮挡边界方向膨胀 5 像素，弥补深度边检测不精确。补全只被「直接相连的同表面像素」约束，网络只看 context region。")
para(doc, "补全（3.3）：虽在 LDI 上，但局部区域像普通 2D 图，可套标准图像补全网络。三个 U-Net 接力——①边缘网络：由上下文边缘预测合成区深度边(结构骨架)；②颜色网络：输入(补全边缘+上下文颜色)→输出合成区 RGB；③深度网络同理。颜色与深度共享同一份预测边缘 → 天然对齐，解决「独立补全对不齐」。")
para(doc, "转 Mesh（3.4）：补全结果整合回 LDI → 转带纹理三角网格 → 无需逐视角推理，标准图形引擎即可在边缘设备渲染。")

h2(doc, "3.3 关键概念")
term(doc, "RGB-D", "彩色图 + 深度图(每像素距相机距离)。双摄手机可直接出，也可用单目深度估计(MegaDepth 等)从普通照片估算。")
term(doc, "LDI（Layered Depth Image，分层深度图像）", "每个像素位置可存多层颜色+深度，像一摞玻璃板。像素间存显式上下左右连接指针——同层表面相连、跨深度边不连。紧凑、自适应深度复杂度、可转 Mesh。")
term(doc, "MPI（Multiplane Image，多平面图像）", "把空间沿深度切成 N 层等距玻璃板，每层存 RGB+透明度 α，渲染时叠加。规整好套 CNN，但斜面有台阶伪影、存储冗余。")
term(doc, "Mesh（三角网格）", "图形学最基础的 3D 表示：顶点+三角面+纹理贴图。所有设备原生渲染，无需网络推理。")
term(doc, "去遮挡（Disocclusion）", "相机偏移时原被前景挡住的背景暴露出来。dis-occlusion = 解除遮挡。本方法的核心待解问题。")
term(doc, "运动视差（Motion Parallax）", "近处物体看起来移动多、远处移动少，大脑借此感知深度。3D 照片的核心体验来源。")
term(doc, "深度边（Depth Edge）", "深度图中深度值突变处 = 前景/背景边界。是补全算法的基本工作单元。")
term(doc, "伪真值（Pseudo Ground Truth）", "用另一个预训练模型(如 MegaDepth)预测的值当训练标准答案。非传感器实测但够用。")

h2(doc, "3.4 技术创新")
bullet(doc, "① 弹性 LDI + 显式连接指针：不同于固定层数 MPI 或刚性 LDI，每像素存任意多层、层间存连接指针。补全时只看「同层物理相连」的已知像素，不被其他层干扰。")
bullet(doc, "② 上下文感知的迭代局部补全：不一次性全局处理，而是一条深度边一条边独立处理，各自提取上下文和合成区域。把 LDI 复杂拓扑问题降维成局部 2D 补全，可直接套标准 CNN。")
bullet(doc, "③ 边缘引导的两阶段补全：先预测深度边结构(骨架)，再在骨架约束下同时补颜色和深度。解决「颜色深度各补各的对不齐」。")
bullet(doc, "④ 无需标注的自监督训练：COCO 图 + MegaDepth 伪深度 → 随机模拟遮挡 → 自动获得 ground truth，不依赖人工标注。")
bullet(doc, "⑤ 输出 Mesh、渲染零推理：重活前置到生成阶段，渲染退化成纯图形学，命中移动端「不依赖 GPU 网络推理」目标。")

h2(doc, "3.5 核心公式与设计（关键参数）")
bullet(doc, "双边中值滤波：7×7 窗口，spatial=4.0，intensity=0.5——锐化深度边同时保边。")
bullet(doc, "合成区域扩展：40 步——覆盖实际去遮挡暴露的区域。")
bullet(doc, "上下文区域扩展：100 步——给 CNN 足够已知信息。")
bullet(doc, "合成区域膨胀：5 像素——补偿深度边检测不精确。")
bullet(doc, "短边删除阈值：<10 像素——5 折交叉验证以 LPIPS 为指标选出。")
bullet(doc, "训练数据：COCO 2017 训练集 11.8 万张，每张最多取 3 对区域。")

h2(doc, "3.6 关键实验结论")
bullet(doc, "消融(4.4)：边缘引导补全在数值指标上仅小幅提升，主要收益在感知质量/视觉不撕裂——因它解决的是颜色深度对齐，不是 PSNR。评估须单独看去遮挡区域，否则被大片已知区域稀释。")
bullet(doc, "不同深度源(4.5)：MegaDepth / MiDaS / Kinect 三种深度都能处理——证明深度估计模块可插拔、与后段管线解耦。")
bullet(doc, "失败案例：复杂细小结构(头发/树枝/栅栏)和反射/透明表面(镜子/玻璃/水面)会失败。根因是单目深度估计难 + 显式深度表示对无单一真实深度的表面失效。")
bullet(doc, "结论：核心贡献是「补全的 LDI + 上下文感知补全」，相比同类方法视觉伪影显著更少。")

h2(doc, "3.7 优势与局限")
para(doc, "✅ 优势：", color=GREEN)
bullet(doc, "与课题目标(单图→移动端浅 3D)匹配度最高：单图输入、小幅晃动、Mesh 输出可移动端渲染。")
bullet(doc, "生成期重、渲染期轻，渲染零网络推理。")
bullet(doc, "深度源可插拔，训练自监督无需标注。")
para(doc, "⚠️ 局限：", color=ORANGE)
bullet(doc, "本质是图层补全而非真 3D 重建，转动角度小。")
bullet(doc, "CNN 补全靠「猜」，语义不合理时可能补出奇怪内容。")
bullet(doc, "依赖深度边检测精度；硬分层会损伤头发/细线等细结构；反射/透明表面失效。")

h2(doc, "3.8 与本课题的关系")
para(doc, "这是目前读到的论文中与课题(单张普通照片 → 移动端浅 3D 表达)匹配度最高的一篇。")
bullet(doc, "可拆解框架：把「单图转浅 3D」拆成深度估计 → 分层表示 → 遮挡补全 → 实时渲染四个模块，比「端到端生成新视角」更清晰、更契合移动端部署。")
bullet(doc, "深度是天花板：深度图质量不只看整体误差，更要看前景背景边界——边界错位直接导致边缘撕裂、背景拉伸。")
bullet(doc, "补全须颜色+深度联合：只补 RGB 有纹理没几何，只补深度没画面。")
bullet(doc, "可换更强深度模型：论文用 MegaDepth/MiDaS，本课题可直接换成 Depth Anything V2(07)或 Depth Pro(08)，后段不变。")
bullet(doc, "后续论文定位：SLIDE、TMPI、单图前馈 3DGS 是对本文「硬分层、冗余存储、细结构边界、表达能力」的进一步改进。")
para(doc, "与本课题方案的对应关系：")
mono(doc, "论文输入 RGB-D        → 本课题输入普通 RGB，先用 Depth Anything/Depth Pro 估深度\n论文表示 LDI+mesh     → 本课题可选 LDI/简化 mesh/TMPI/轻量 3DGS\n论文补全 局部颜色+深度 → 本课题借鉴：对新视角显露区域做深度感知修复\n论文渲染 图形引擎实时  → 本课题目标：移动端小幅晃动/滑动产生运动视差")

h2(doc, "3.9 可写入论文/开题报告的参考表述")
para(doc, "单张图像生成浅 3D 表达的关键难点在于视角变化引起的显露区域(disocclusion)。传统基于深度的重投影方法虽能产生一定运动视差，但在前景与背景交界处容易出现空洞或纹理拉伸。Shih 等人提出的 3D Photography 方法以 RGB-D 图像为输入，构建具有显式像素连接关系的分层深度图(LDI)，并沿深度不连续边界进行局部颜色与深度的联合补全，从而在被遮挡区域生成合理的背景结构。该方法表明，单图浅 3D 表达可被拆解为深度估计、分层表示、遮挡区域补全和轻量渲染等模块，为移动端浅 3D 照片生成提供了清晰的工程路径。")

doc.save(OUT)
print("saved:", OUT)
