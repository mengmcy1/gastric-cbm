#!/usr/bin/env python3
"""将冻结M0 Keep清单与癌图bbox合并为M1辅助定位训练清单。"""

import argparse
import ast
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT / "数据整理记录/图像裁剪"
    / "胃早癌概念提取训练集0804_预处理_v1"
    / "07_M0来源内平衡清单_20260805"
    / "m0_balanced_keep_primary_1to1p3_split_seed42.csv"
)
DEFAULT_BBOX = PROJECT_ROOT / "数据/标注与汇总/bbox_manifest_20260807.csv"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "数据整理记录/图像裁剪"
    / "胃早癌概念提取训练集0804_预处理_v1"
    / "08_M1辅助定位清单_20260810"
)
SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}


# 配置与解析：固定M0主清单、临床bbox清单和全新输出目录。
def parse_args():
    """定义M1清单输入、bbox来源、数据角色和不可覆盖输出目录。

    Returns:
        argparse.Namespace: 路径参数为``Path``，``dataset_role``和
        ``output_name``为``str``。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--bbox", type=Path, default=DEFAULT_BBOX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dataset-role",
        choices=["balanced_primary", "full_diagnostic"],
        default="balanced_primary",
        help="决定输出命名和报告结论；不改变合并与校验逻辑。",
    )
    parser.add_argument(
        "--output-name",
        default="",
        help="可选CSV文件名；留空时根据dataset-role使用正式名称。",
    )
    return parser.parse_args()


def file_sha256(path):
    """计算输入与输出文件SHA256，用于冻结清单追溯。

    Args:
        path (Path): 待读取的本地文件路径。

    Returns:
        str: 64位小写十六进制SHA256摘要。
    """
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def parse_bbox(value):
    """安全解析四元素XYXY字符串并统一转换为浮点坐标。

    Args:
        value (object): CSV中的bbox值，通常为``"[x1,y1,x2,y2]"``字符串。

    Returns:
        tuple[float, float, float, float]: 原处理图像素坐标系中的XYXY框。

    Raises:
        ValueError: 内容不能安全解析或不是四元素序列。
    """
    try:
        box = ast.literal_eval(str(value))
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"无法解析bbox: {value}") from error
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError(f"bbox必须为4个坐标: {value}")
    return tuple(float(number) for number in box)


def validate_base_manifest(frame):
    """校验M0 Keep清单标签、split、患者身份、路径主键和跨患者重复。

    Args:
        frame (pandas.DataFrame): 平衡主实验或全量诊断的M0 Keep图片清单。

    Returns:
        None: 校验通过时不修改输入表。

    Raises:
        ValueError: 字段、类别、split、患者归属、路径或跨患者SHA不合法。
    """
    required = {
        "relative_path", "image_relpath", "patient_id", "label", "split",
        "width", "height", "sha256", "branch",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"M0清单缺少字段: {sorted(missing)}")
    if set(frame.label.unique()) != {0, 1}:
        raise ValueError("M0清单必须同时包含癌与非癌")
    if set(frame.split.unique()) != {"train", "val", "test"}:
        raise ValueError("M0清单必须包含train/val/test")
    if not frame.branch.eq("keep").all():
        raise ValueError("M1主清单只允许使用Keep分支")
    if frame.relative_path.duplicated().any():
        raise ValueError("M0清单存在重复图片路径")
    if frame.groupby("patient_id").split.nunique().gt(1).any():
        raise ValueError("检测到患者跨split")
    if frame.groupby("patient_id").label.nunique().gt(1).any():
        raise ValueError("检测到患者跨标签")
    cross_patient_sha = frame.groupby("sha256").patient_id.nunique()
    if cross_patient_sha.gt(1).any():
        raise ValueError(
            f"M0清单存在{int(cross_patient_sha.gt(1).sum())}组跨患者完全重复图"
        )


def validate_bbox_manifest(frame):
    """校验临床bbox字段、癌标签限定及处理图路径唯一性。

    Args:
        frame (pandas.DataFrame): 医学生交付并整合后的癌图bbox清单。

    Returns:
        None: 校验通过时不修改输入表。

    Raises:
        ValueError: 缺字段、出现非癌标注或处理图路径重复。
    """
    required = {
        "processed_relative_path", "patient_id", "label", "bbox_xyxy",
        "pip_present", "notch_intersect",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"bbox清单缺少字段: {sorted(missing)}")
    if not frame.label.eq(1).all():
        raise ValueError("bbox清单中出现非癌标签")
    if frame.processed_relative_path.duplicated().any():
        duplicates = int(frame.processed_relative_path.duplicated(False).sum())
        raise ValueError(f"bbox清单存在{duplicates}条重复处理路径")


def attach_bbox(frame, bbox_frame):
    """按处理图相对路径连接癌框，并生成展开及归一化bbox训练字段。

    Args:
        frame (pandas.DataFrame): 已校验M0图片清单，含尺寸和患者字段。
        bbox_frame (pandas.DataFrame): 已校验临床癌图bbox清单。

    Returns:
        pandas.DataFrame: 保留M0字段并增加像素/归一化XYXY、中心、宽高、
        面积比例、``bbox_valid``和``localization_supervision``字段的M1清单。

    Raises:
        ValueError: 癌图漏框、非癌误匹配、患者/标签不一致或bbox越界。
    """
    bbox_columns = [
        "processed_relative_path", "patient_id", "label", "bbox_xyxy",
        "pip_present", "notch_intersect",
    ]
    renamed = bbox_frame[bbox_columns].rename(columns={
        "patient_id": "bbox_patient_id",
        "label": "bbox_label",
        "pip_present": "bbox_pip_present",
        "notch_intersect": "bbox_notch_intersect",
    })
    result = frame.merge(
        renamed,
        how="left",
        left_on="relative_path",
        right_on="processed_relative_path",
        validate="one_to_one",
    )
    cancer = result.label.eq(1)
    if result.loc[cancer, "bbox_xyxy"].isna().any():
        missing = result.loc[cancer & result.bbox_xyxy.isna(), "relative_path"]
        raise ValueError(f"{len(missing)}张癌图缺少bbox，示例: {missing.iloc[0]}")
    if result.loc[~cancer, "bbox_xyxy"].notna().any():
        raise ValueError("非癌图意外匹配到bbox")
    patient_mismatch = cancer & result.bbox_patient_id.ne(result.patient_id)
    if patient_mismatch.any():
        row = result.loc[patient_mismatch].iloc[0]
        raise ValueError(
            f"bbox患者不一致: {row.relative_path} / "
            f"{row.patient_id} != {row.bbox_patient_id}"
        )
    if not result.loc[cancer, "bbox_label"].eq(1).all():
        raise ValueError("癌图匹配到错误bbox标签")

    result["localization_supervision"] = cancer.astype(int)
    for column in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]:
        result[column] = float("nan")
    result["bbox_valid"] = False
    for index in result.index[cancer]:
        x1, y1, x2, y2 = parse_bbox(result.at[index, "bbox_xyxy"])
        width = float(result.at[index, "width"])
        height = float(result.at[index, "height"])
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(
                f"bbox越界: {result.at[index, 'relative_path']} / "
                f"{(x1, y1, x2, y2)} vs {(width, height)}"
            )
        result.loc[index, ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]] = [
            x1, y1, x2, y2
        ]
        result.at[index, "bbox_valid"] = True

    result["bbox_x1_norm"] = result.bbox_x1 / result.width
    result["bbox_y1_norm"] = result.bbox_y1 / result.height
    result["bbox_x2_norm"] = result.bbox_x2 / result.width
    result["bbox_y2_norm"] = result.bbox_y2 / result.height
    result["bbox_center_x_norm"] = (
        result.bbox_x1_norm + result.bbox_x2_norm
    ) / 2
    result["bbox_center_y_norm"] = (
        result.bbox_y1_norm + result.bbox_y2_norm
    ) / 2
    result["bbox_width_norm"] = result.bbox_x2_norm - result.bbox_x1_norm
    result["bbox_height_norm"] = result.bbox_y2_norm - result.bbox_y1_norm
    result["bbox_area_fraction"] = (
        result.bbox_width_norm * result.bbox_height_norm
    )
    result.loc[~cancer, "bbox_xyxy"] = ""
    result.loc[~cancer, "bbox_pip_present"] = ""
    result.loc[~cancer, "bbox_notch_intersect"] = ""
    return result.drop(columns=[
        "processed_relative_path", "bbox_patient_id", "bbox_label",
    ])


def validate_images(frame):
    """逐图确认处理文件存在，且实际尺寸与bbox坐标系完全一致。

    Args:
        frame (pandas.DataFrame): 已连接bbox的候选M1清单。

    Returns:
        None: 校验通过时不修改清单或图像。

    Raises:
        FileNotFoundError: 处理图缺失。
        ValueError: 实际``(width,height)``与清单尺寸不一致。
    """
    missing = []
    dimension_mismatch = []
    for row in frame.itertuples(index=False):
        path = PROJECT_ROOT / row.image_relpath
        if not path.is_file():
            missing.append(str(path))
            continue
        with Image.open(path) as image:
            actual = image.size
        expected = (int(row.width), int(row.height))
        if actual != expected:
            dimension_mismatch.append((row.relative_path, expected, actual))
    if missing:
        raise FileNotFoundError(f"缺失{len(missing)}张处理图，示例: {missing[0]}")
    if dimension_mismatch:
        raise ValueError(
            f"{len(dimension_mismatch)}张图尺寸不一致，示例: {dimension_mismatch[0]}"
        )


def build_audit(frame):
    """按split汇总图片、患者、标签及癌图bbox覆盖率。

    Args:
        frame (pandas.DataFrame): 最终候选M1图片清单。

    Returns:
        pandas.DataFrame: ``train/val/test/all``四行的数据规模和bbox覆盖统计。
    """
    patient_frame = frame.drop_duplicates("patient_id")
    rows = []
    for split in ["train", "val", "test", "all"]:
        images = frame if split == "all" else frame.loc[frame.split.eq(split)]
        patients = patient_frame if split == "all" else patient_frame.loc[
            patient_frame.split.eq(split)
        ]
        rows.append({
            "split": split,
            "images": int(len(images)),
            "patients": int(len(patients)),
            "cancer_images": int(images.label.eq(1).sum()),
            "control_images": int(images.label.eq(0).sum()),
            "cancer_patients": int(patients.label.eq(1).sum()),
            "control_patients": int(patients.label.eq(0).sum()),
            "bbox_supervised_images": int(images.localization_supervision.sum()),
            "bbox_coverage_cancer": float(
                images.loc[images.label.eq(1), "bbox_valid"].mean()
            ),
        })
    return pd.DataFrame(rows)


def dataframe_to_markdown(frame):
    """无额外依赖地把小型审计表转换成Markdown表格。

    Args:
        frame (pandas.DataFrame): 需要展示的小型二维表。

    Returns:
        str: 含表头、分隔行和数据行的Markdown文本。
    """
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    output, audit, checks, source_manifest, bbox_manifest, dataset_role
):
    """写出便于人工阅读的M1清单验收报告和结论边界。

    Args:
        output (Path): 已创建的本次清单输出目录。
        audit (pandas.DataFrame): ``build_audit``生成的分层统计表。
        checks (dict[str, object]): 输入摘要、策略和自动检查结果。
        source_manifest (Path): M0 Keep来源清单路径。
        bbox_manifest (Path): 临床bbox来源清单路径。
        dataset_role (str): ``balanced_primary``或``full_diagnostic``。

    Returns:
        None: 写出UTF-8编码的``M1辅助定位清单验收.md``。
    """
    all_row = audit.loc[audit.split.eq("all")].iloc[0]
    role_text = {
        "balanced_primary": "来源内1:1.3平衡主实验",
        "full_diagnostic": "全量诊断对照实验",
    }[dataset_role]
    lines = [
        "# M1辅助定位清单验收",
        "",
        f"- 生成时间：{checks['created_at']}",
        f"- M0来源清单：`{source_manifest}`",
        f"- bbox来源清单：`{bbox_manifest}`",
        f"- 数据角色：{role_text}。",
        "- 输入分支：仅Keep；不读取带可视化矩形框的图片。",
        f"- 总规模：{all_row.patients}人/{all_row.images}张。",
        f"- 癌图bbox覆盖：{all_row.bbox_supervised_images}/{all_row.cancer_images}（100%）。",
        "- 非癌图经过定位头，但M1中`localization_supervision=0`，不计算定位损失。",
        "",
        "## 自动检查",
        "",
    ]
    lines.extend(f"- {key}: `{value}`" for key, value in checks["checks"].items())
    lines.extend([
        "",
        "## 分层规模",
        "",
        dataframe_to_markdown(audit),
        "",
        "## 结论",
        "",
        f"该清单可作为M1{role_text}的冻结输入。Notch相交标记仅随表保留",
        "用于旁路审计，不影响Keep主线定位监督。内部test行已冻结，但模型选择阶段继续",
        "保持默认不传`--evaluate-test`，不得读取test结果调参。",
        "",
    ])
    (output / "M1辅助定位清单验收.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    """编排清单合并、硬校验、审计、协议记录和校验和生成。

    Args:
        无。输入、数据角色和输出目录均来自命令行参数。

    Returns:
        None: 生成冻结CSV、审计CSV、协议JSON、验收Markdown和checksums。
    """
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if not args.bbox.is_file():
        raise FileNotFoundError(args.bbox)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"输出目录非空，拒绝覆盖: {args.output}")
    default_names = {
        "balanced_primary": "m1_balanced_keep_primary_1to1p3_split_seed42.csv",
        "full_diagnostic": "m1_full_keep_split_seed42.csv",
    }
    output_name = args.output_name or default_names[args.dataset_role]
    if Path(output_name).name != output_name or not output_name.lower().endswith(".csv"):
        raise ValueError("--output-name必须是当前输出目录下的CSV文件名")

    frame = pd.read_csv(args.input, encoding="utf-8-sig", dtype={"patient_id": str})
    bbox_frame = pd.read_csv(
        args.bbox, encoding="utf-8-sig", dtype={"patient_id": str}
    )
    validate_base_manifest(frame)
    validate_bbox_manifest(bbox_frame)
    result = attach_bbox(frame, bbox_frame)
    validate_images(result)
    sort_columns = ["split", "source", "label", "patient_id"]
    if "selection_rank" in result:
        sort_columns.append("selection_rank")
    else:
        sort_columns.append("relative_path")
    result = result.sort_values(
        sort_columns,
        key=lambda column: column.map(SPLIT_ORDER) if column.name == "split" else column,
        kind="stable",
    ).reset_index(drop=True)
    audit = build_audit(result)

    checks = {
        "created_at": datetime.now().astimezone().isoformat(),
        "input_manifest": str(args.input.resolve()),
        "input_manifest_sha256": file_sha256(args.input),
        "bbox_manifest": str(args.bbox.resolve()),
        "bbox_manifest_sha256": file_sha256(args.bbox),
        "policy": {
            "dataset_role": args.dataset_role,
            "image_branch": "keep_only",
            "positive_localization": "all cancer images with valid bbox",
            "negative_localization": "forward_only; localization loss masked in M1",
            "duplicate_policy": "reject cross-patient processed SHA duplicates",
            "test_policy": "frozen in manifest; deferred during M1 model selection",
        },
        "checks": {
            "all_paths_exist": True,
            "all_image_dimensions_match_manifest": True,
            "patients_do_not_cross_split": True,
            "patients_do_not_cross_label": True,
            "cross_patient_processed_sha_groups": 0,
            "cancer_bbox_coverage": 1.0,
            "all_bbox_inside_image": True,
            "noncancer_bbox_count": 0,
        },
    }

    args.output.mkdir(parents=True, exist_ok=False)
    manifest_path = args.output / output_name
    result.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    audit.to_csv(args.output / "m1_manifest_audit.csv", index=False, encoding="utf-8-sig")
    (args.output / "m1_pre_registered_manifest_protocol.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(
        args.output, audit, checks, args.input, args.bbox, args.dataset_role
    )
    checksum_lines = []
    for path in sorted(args.output.iterdir()):
        if path.name != "checksums.sha256":
            checksum_lines.append(f"{file_sha256(path)}  {path.name}")
    (args.output / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    print(audit.to_string(index=False))
    print(f"M1清单: {manifest_path}")
    print(f"输出目录: {args.output}")


if __name__ == "__main__":
    main()
