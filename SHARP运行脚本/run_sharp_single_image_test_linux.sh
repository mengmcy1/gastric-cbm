#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname -- "$SCRIPT_DIR")"
SHARP_REPO="$PROJECT_ROOT/源码/SHARP_APPLE注释"
CHECKPOINT="$SHARP_REPO/checkpoints/sharp_2572gikvuh.pt"
CONDA_SH="/home/mcy/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="sharp"
GPU_INDEX="${SHARP_GPU_INDEX:-1}"
INPUT_ROOT="$PROJECT_ROOT/SHARP真实图片测试/输入图片"
TEST_ROOT="$PROJECT_ROOT/SHARP真实图片测试/输出结果"

usage() {
    printf '%s\n' \
        "用法：" \
        "  bash run_sharp_single_image_test_linux.sh <图片路径> [测试名称]" \
        "" \
        "示例：" \
        "  先把图片放到：$INPUT_ROOT" \
        "  bash run_sharp_single_image_test_linux.sh \"$INPUT_ROOT/room.jpg\" 房间人物" \
        "" \
        "功能：" \
        "  1. 固定使用物理 GPU 1（第二张显卡）" \
        "  2. 运行 SHARP 预测并保存 PLY" \
        "  3. 分别运行首次渲染和热启动渲染" \
        "  4. 保存环境、预测和渲染日志" \
        "  5. 从彩色视频和深度视频提取 0/0.5/1.0/1.5 秒关键帧" \
        "  6. 生成问题分析记录模板" \
        "" \
        "可选环境变量：" \
        "  SHARP_GPU_INDEX=1  指定物理 GPU，默认固定为 1"
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage
    exit 2
fi

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi

INPUT_PATH="$(readlink -f -- "$1")"
if [[ ! -f "$INPUT_PATH" ]]; then
    printf '错误：输入图片不存在：%s\n' "$INPUT_PATH" >&2
    exit 1
fi

INPUT_NAME="$(basename -- "$INPUT_PATH")"
if [[ "$INPUT_NAME" != *.* ]]; then
    printf '错误：输入文件没有扩展名：%s\n' "$INPUT_NAME" >&2
    exit 1
fi

INPUT_EXT="${INPUT_NAME##*.}"
INPUT_EXT="${INPUT_EXT,,}"
case "$INPUT_EXT" in
    jpg|jpeg|png|heic|heif|webp|bmp|tif|tiff) ;;
    *)
        printf '警告：扩展名 .%s 可能不在 SHARP 的支持列表中。\n' "$INPUT_EXT" >&2
        ;;
esac

if [[ ! -d "$SHARP_REPO" ]]; then
    printf '错误：找不到 SHARP 仓库：%s\n' "$SHARP_REPO" >&2
    exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
    printf '错误：找不到 SHARP 权重：%s\n' "$CHECKPOINT" >&2
    exit 1
fi
if [[ ! -f "$CONDA_SH" ]]; then
    printf '错误：找不到 Conda 初始化脚本：%s\n' "$CONDA_SH" >&2
    exit 1
fi

mkdir -p "$INPUT_ROOT" "$TEST_ROOT"

if [[ $# -eq 2 ]]; then
    TEST_LABEL="$2"
else
    TEST_LABEL="${INPUT_NAME%.*}"
fi
TEST_LABEL="${TEST_LABEL//\//_}"
TEST_LABEL="${TEST_LABEL// /_}"
RUN_ID="$(date '+%Y%m%d_%H%M%S')_${TEST_LABEL}"
RUN_DIR="$TEST_ROOT/$RUN_ID"
INPUT_DIR="$RUN_DIR/输入图片"
PREDICT_DIR="$RUN_DIR/预测输出"
RENDER_FIRST_DIR="$RUN_DIR/渲染输出_首次"
RENDER_WARM_DIR="$RUN_DIR/渲染输出_热启动"
LOG_DIR="$RUN_DIR/日志"
FRAME_DIR="$RUN_DIR/关键帧"

mkdir -p \
    "$INPUT_DIR" \
    "$PREDICT_DIR" \
    "$RENDER_FIRST_DIR" \
    "$RENDER_WARM_DIR" \
    "$LOG_DIR" \
    "$FRAME_DIR/彩色" \
    "$FRAME_DIR/深度"

COPIED_INPUT="$INPUT_DIR/input.$INPUT_EXT"
cp -p -- "$INPUT_PATH" "$COPIED_INPUT"

on_error() {
    local exit_code=$?
    printf '\n测试失败，退出码：%s，出错行：%s\n' "$exit_code" "${BASH_LINENO[0]}" >&2
    printf '已生成的文件和日志保留在：%s\n' "$RUN_DIR" >&2
    exit "$exit_code"
}
trap on_error ERR

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export PYTHONUNBUFFERED=1

if ! command -v sharp >/dev/null 2>&1; then
    printf '错误：当前 Conda 环境中找不到 sharp 命令。\n' >&2
    exit 1
fi
if [[ ! -x /usr/bin/time ]]; then
    printf '错误：找不到 /usr/bin/time，无法记录完整运行信息。\n' >&2
    exit 1
fi

{
    printf '测试开始时间：%s\n' "$(date --iso-8601=seconds)"
    printf '测试目录：%s\n' "$RUN_DIR"
    printf '原始输入：%s\n' "$INPUT_PATH"
    printf '测试副本：%s\n' "$COPIED_INPUT"
    printf 'SHARP 仓库：%s\n' "$SHARP_REPO"
    printf '权重：%s\n' "$CHECKPOINT"
    printf 'Conda 环境：%s\n' "$CONDA_ENV"
    printf 'CUDA_VISIBLE_DEVICES：%s\n' "$CUDA_VISIBLE_DEVICES"
    python -c 'import torch; print("PyTorch：", torch.__version__); print("CUDA可用：", torch.cuda.is_available()); print("进程可见GPU数量：", torch.cuda.device_count()); print("进程cuda:0：", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "不可用")'
    python -c 'import sys; from PIL import Image; p=sys.argv[1]; im=Image.open(p); exif=im.getexif(); print("图片格式：", im.format); print("原始分辨率：", im.size); print("EXIF项目数：", len(exif)); print("35mm等效焦距：", exif.get(41989, "缺失，将由SHARP尝试其他字段或使用默认30mm"))' "$COPIED_INPUT"
} 2>&1 | tee "$LOG_DIR/environment.log"

python -c 'import torch,sys; ok=torch.cuda.is_available() and torch.cuda.device_count()==1; sys.exit(0 if ok else 1)'

cd "$SHARP_REPO"

printf '\n[1/3] 开始预测高斯。\n'
/usr/bin/time -v sharp predict \
    -i "$COPIED_INPUT" \
    -o "$PREDICT_DIR" \
    -c "$CHECKPOINT" \
    --device cuda \
    --no-render \
    -v \
    2>&1 | tee "$LOG_DIR/predict.log"

PLY_PATH="$PREDICT_DIR/input.ply"
if [[ ! -s "$PLY_PATH" ]]; then
    printf '错误：预测结束后没有生成有效 PLY：%s\n' "$PLY_PATH" >&2
    exit 1
fi

printf '\n[2/3] 开始首次渲染。\n'
/usr/bin/time -v sharp render \
    -i "$PLY_PATH" \
    -o "$RENDER_FIRST_DIR" \
    -v \
    2>&1 | tee "$LOG_DIR/render_first.log"

printf '\n[3/3] 开始热启动渲染。\n'
/usr/bin/time -v sharp render \
    -i "$PLY_PATH" \
    -o "$RENDER_WARM_DIR" \
    -v \
    2>&1 | tee "$LOG_DIR/render_warm.log"

COLOR_VIDEO="$RENDER_WARM_DIR/input.mp4"
DEPTH_VIDEO="$RENDER_WARM_DIR/input.depth.mp4"
if [[ ! -s "$COLOR_VIDEO" || ! -s "$DEPTH_VIDEO" ]]; then
    printf '错误：没有生成完整的彩色/深度视频。\n' >&2
    exit 1
fi

FFMPEG_BIN=""
if command -v ffmpeg >/dev/null 2>&1; then
    FFMPEG_BIN="$(command -v ffmpeg)"
else
    FFMPEG_BIN="$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())' 2>/dev/null || true)"
fi

extract_frames() {
    local video_path="$1"
    local output_dir="$2"
    local prefix="$3"
    local timestamps=("0.0" "0.5" "1.0" "1.5")
    local labels=("00_source" "05_side_a" "10_forward" "15_side_b")
    local i

    for i in "${!timestamps[@]}"; do
        "$FFMPEG_BIN" \
            -hide_banner \
            -loglevel error \
            -y \
            -ss "${timestamps[$i]}" \
            -i "$video_path" \
            -frames:v 1 \
            "$output_dir/${prefix}_${labels[$i]}.png"
    done
}

if [[ -n "$FFMPEG_BIN" && -x "$FFMPEG_BIN" ]]; then
    extract_frames "$COLOR_VIDEO" "$FRAME_DIR/彩色" "color"
    extract_frames "$DEPTH_VIDEO" "$FRAME_DIR/深度" "depth"
else
    printf '警告：找不到 ffmpeg，跳过关键帧提取。\n' | tee -a "$LOG_DIR/environment.log"
fi

if command -v ffprobe >/dev/null 2>&1; then
    {
        printf '===== 彩色视频 =====\n'
        ffprobe -v error -select_streams v:0 \
            -show_entries stream=width,height,r_frame_rate,nb_frames,duration \
            -of default=noprint_wrappers=1 "$COLOR_VIDEO"
        printf '\n===== 深度视频 =====\n'
        ffprobe -v error -select_streams v:0 \
            -show_entries stream=width,height,r_frame_rate,nb_frames,duration \
            -of default=noprint_wrappers=1 "$DEPTH_VIDEO"
    } > "$LOG_DIR/video_info.txt"
fi

PLY_SIZE="$(du -h "$PLY_PATH" | cut -f1)"
INPUT_SIZE="$(du -h "$COPIED_INPUT" | cut -f1)"
COLOR_VIDEO_SIZE="$(du -h "$COLOR_VIDEO" | cut -f1)"
DEPTH_VIDEO_SIZE="$(du -h "$DEPTH_VIDEO" | cut -f1)"
REPORT_PATH="$RUN_DIR/问题分析记录.md"

python - "$REPORT_PATH" "$RUN_ID" "$INPUT_PATH" "$INPUT_SIZE" "$PLY_SIZE" "$COLOR_VIDEO_SIZE" "$DEPTH_VIDEO_SIZE" "$GPU_INDEX" <<'PY'
from pathlib import Path
import sys

report_path = Path(sys.argv[1])
run_id, input_path, input_size, ply_size, color_size, depth_size, gpu_index = sys.argv[2:]

text = f"""# SHARP 单张真实图片测试记录

## 1. 测试基本信息

- 测试编号：{run_id}
- 原始输入：`{input_path}`
- 输入文件大小：{input_size}
- PLY 大小：{ply_size}
- 彩色视频大小：{color_size}
- 深度视频大小：{depth_size}
- 默认轨迹：`rotate_forward`
- 默认帧数：60
- 默认帧率：30 FPS
- 物理 GPU：GPU {gpu_index}（进程内显示为 `cuda:0`）

运行环境、焦距信息和分辨率见：`日志/environment.log`。
运行耗时和资源占用见：`日志/predict.log`、`日志/render_first.log`、`日志/render_warm.log`。

## 2. 场景内容

- 场景类型：待填写
- 主要前景：待填写
- 主要背景：待填写
- 是否包含细结构：待填写
- 是否包含反射/透明区域：待填写
- 是否包含明显景深或运动模糊：待填写
- EXIF 焦距是否存在：查看 `日志/environment.log`

## 3. 画质检查

严重度：0=没有，1=轻微，2=明显，3=严重影响观看。

| 检查项 | 严重度 0–3 | 发生位置 | 具体表现 |
|---|---:|---|---|
| 原始视角重建 |  |  |  |
| 整体深度结构 |  |  |  |
| 前景边缘 |  |  |  |
| 新显露区域 |  |  |  |
| 细长结构 |  |  |  |
| 漂浮高斯 |  |  |  |
| 重影或双边 |  |  |  |
| 反射/透明区域 |  |  |  |
| 连续帧稳定性 |  |  |  |
| 清晰度和纹理 |  |  |  |
| 全局视差幅度 |  |  |  |

## 4. 分阶段判断

### 原始相机位置

比较原图与 `关键帧/彩色/color_00_source.png`：

- 是否已经模糊或失真：待填写
- 如果原视角就错误，优先检查颜色、透明度、焦距、输入缩放和高斯覆盖。

### 附近新视角

比较另外三张彩色关键帧：

- 是否只在前景边缘出现问题：待填写
- 是否出现空洞、拖抹或背景复制：待填写
- 如果原视角正常、新视角错误，优先检查深度、第二层高斯和遮挡补全能力。

### 深度结构

查看 `关键帧/深度/`：

- 前景是否比背景更近：待填写
- 墙面、天空和地面是否异常弯曲：待填写
- 细结构是否与背景粘连：待填写
- 反射、透明或虚化区域是否出现错误深度：待填写

## 5. 初步归因

- Depth Pro 深度问题：待填写
- 双层深度分工问题：待填写
- Gaussian Decoder 问题：待填写
- 3DGS 覆盖/表示问题：待填写
- EXIF 或焦距问题：待填写
- CLI 工程性能问题：待填写

## 6. 最值得优先解决的问题

1. 待填写
2. 待填写
3. 待填写

## 7. 后续对照实验

- [ ] 再测试一张普通清晰场景
- [ ] 测试头发、栏杆、树枝等细结构
- [ ] 测试反射、透明、夜景或明显景深
- [ ] 比较存在 EXIF 与缺少 EXIF 的输入
- [ ] 比较不同相机运动幅度
- [ ] 记录高斯数量、PLY 大小和热启动渲染速度
"""

report_path.write_text(text, encoding="utf-8")
PY

{
    printf '\n测试完成。\n'
    printf '测试目录：%s\n' "$RUN_DIR"
    printf '输入副本：%s\n' "$COPIED_INPUT"
    printf '高斯 PLY：%s（%s）\n' "$PLY_PATH" "$PLY_SIZE"
    printf '热启动彩色视频：%s\n' "$COLOR_VIDEO"
    printf '热启动深度视频：%s\n' "$DEPTH_VIDEO"
    printf '关键帧目录：%s\n' "$FRAME_DIR"
    printf '问题分析记录：%s\n' "$REPORT_PATH"
    printf '日志目录：%s\n' "$LOG_DIR"
} | tee "$LOG_DIR/completion.txt"
