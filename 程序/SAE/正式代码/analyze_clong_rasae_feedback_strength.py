#!/usr/bin/env python3
"""将首批医学自由反馈对应到图1—8，比较主要描述与疑似反例的激活强度。

本分析只读首批既有64张选例及其peak/Q99，不重新选图、不重新forward、
不请求医学侧查看新图。“疑似反例”仅表示与该Feature当前主要候选描述
不一致，不意味着幽门、皱襞、反光或病灶外响应本身必然错误。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
BATCH_ROOT = ROOT / "结果/SAE/RA_SAE_Pilot_20260908/functional_first_medical_batch_20260915"


FEEDBACK = {
    219: {
        "raw": "图1、2、5、6、7中的高激活区域特点类似，红白相间的凹陷区域；但图3主要在胃体皱襞上，图4在边框处；图8在反光处，不在病灶处",
        "main": [1, 2, 5, 6, 7], "counter": [3, 4, 8], "other": [],
        "descriptions": {
            "main": "红白相间的凹陷区域",
            "counter": "图3胃体皱襞；图4边框；图8反光且不在病灶",
        },
    },
    601: {
        "raw": "图2、3、4、5、8的高激活区域主要在幽门口；图1、7在病灶处，但相同点较少，可能表示病灶边界？不是很确定；图6主要在边框处，可能和透明帽有关",
        "main": [2, 3, 4, 5, 8], "counter": [6], "other": [1, 7],
        "descriptions": {
            "main": "主要在幽门口",
            "counter": "主要在边框，可能与透明帽有关",
            "other": "在病灶处，与幽门口模式相同点较少；可能是病灶边界，不确定",
        },
    },
    868: {
        "raw": "图1主要在正常黏膜处；图2-7在病灶处，可能表示凹陷或糜烂；图8在病灶旁中断的皱襞处",
        "main": [2, 3, 4, 5, 6, 7], "counter": [1], "other": [8],
        "descriptions": {
            "main": "在病灶处，可能是凹陷或糜烂",
            "counter": "主要在正常黏膜处",
            "other": "在病灶旁中断的皱襞处",
        },
    },
    1089: {
        "raw": "图1、2、4、5、8都是幽门附近的病灶，热图高响应主要在幽门口处，有的会包含部分病灶；图3、7在隆起病灶处，可表示为隆起，但无幽门；图6在正常黏膜处，可能表示正常黏膜或边框",
        "main": [1, 2, 4, 5, 8], "counter": [6], "other": [3, 7],
        "descriptions": {
            "main": "幽门附近病灶，高响应主要在幽门口，有时包含部分病灶",
            "counter": "在正常黏膜处，可能是正常黏膜或边框",
            "other": "在隆起病灶处，可能表示隆起，但没有幽门",
        },
    },
    1385: {
        "raw": "可能表示病灶的边界",
        "main": [], "counter": [], "other": list(range(1, 9)),
        "descriptions": {"other": "未逐图指定；整体候选描述为病灶边界"},
    },
    2388: {
        "raw": "可能表示病灶发红与凹陷，但图3中高激活区域和病灶关系不大，其他几张图都在病灶处",
        "main": [1, 2, 4, 5, 6, 7, 8], "counter": [3], "other": [],
        "descriptions": {
            "main": "在病灶处，可能表示发红与凹陷",
            "counter": "高激活区域和病灶关系不大",
        },
    },
    2444: {
        "raw": "图1、2、5、6、7高激活区域都在病灶处，可能表示凹陷；图4不在病灶处，但形态与凹陷接近；但图3在透明帽处，图8的高激活区域太小，看不太出来是什么",
        "main": [1, 2, 5, 6, 7], "counter": [3], "other": [4, 8],
        "descriptions": {
            "main": "在病灶处，可能表示凹陷",
            "counter": "在透明帽处",
            "other": "图4不在病灶但形态接近凹陷；图8高激活区域太小，无法判断",
        },
    },
    2464: {
        "raw": "图1、3、4、7、8可能表示隆起或病灶边界；图2、5、6主要在边框处，和病灶关系不大",
        "main": [1, 3, 4, 7, 8], "counter": [2, 5, 6], "other": [],
        "descriptions": {
            "main": "可能表示隆起或病灶边界",
            "counter": "主要在边框，和病灶关系不大",
        },
    },
}


def map_feedback(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """将医学生的图1—8描述对应到现有图片代号。

    Args:
        selected (pd.DataFrame): 首批64张选例，保持成图顺序。

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: 8行原始反馈和64行逐图对应。
    """
    frame = selected.copy()
    frame["image_number"] = frame.groupby("feature_id", sort=False).cumcount() + 1
    if set(frame.feature_id) != set(FEEDBACK) or not frame.groupby("feature_id").size().eq(8).all():
        raise RuntimeError("反馈Feature或图1—8与首批选例不一致")
    raw_rows, mapped_rows = [], []
    for feature, feedback in FEEDBACK.items():
        assigned = feedback["main"] + feedback["counter"] + feedback["other"]
        if sorted(assigned) != list(range(1, 9)) or len(set(assigned)) != 8:
            raise RuntimeError(f"RA-F{feature:04d}的图1—8未恰好分配一次")
        raw_rows.append({
            "feature_id": feature, "feature": f"RA-F{feature:04d}",
            "raw_medical_feedback": feedback["raw"],
            "scope": "首批8张图的初步医学生反馈；不是全量标注或已确认命名",
        })
        for record in frame[frame.feature_id.eq(feature)].itertuples():
            number = int(record.image_number)
            if number in feedback["main"]:
                relation = "main_candidate_description"
                description = feedback["descriptions"]["main"]
            elif number in feedback["counter"]:
                relation = "possible_counterexample"
                description = feedback["descriptions"]["counter"]
            else:
                relation = "other_or_uncertain"
                description = feedback["descriptions"]["other"]
            mapped_rows.append({
                "feature_id": feature, "feature": f"RA-F{feature:04d}",
                "image_number": number, "image_code": record.image_code,
                "relation_to_current_candidate_description": relation,
                "medical_description": description,
                "peak_q99": float(record.peak_q99),
                "interpretation_boundary": "反例仅指与当前主要候选描述不一致；不判定该解剖/成像内容本身错误",
            })
    return pd.DataFrame(raw_rows), pd.DataFrame(mapped_rows)


def summarize_strength(mapped: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """汇总各Feature不同反馈关系的peak/Q99，不作显著性推断。

    Args:
        mapped (pd.DataFrame): 64行逐图反馈对应。

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: 分类描述统计和主要/疑似反例对照表。
    """
    rows = []
    for (feature, relation), frame in mapped.groupby(
        ["feature", "relation_to_current_candidate_description"], sort=False
    ):
        rows.append({
            "feature": feature, "relation": relation, "images": len(frame),
            "mean_peak_q99": float(frame.peak_q99.mean()),
            "median_peak_q99": float(frame.peak_q99.median()),
            "min_peak_q99": float(frame.peak_q99.min()),
            "max_peak_q99": float(frame.peak_q99.max()),
        })
    category = pd.DataFrame(rows)
    comparisons = []
    for feature in sorted(mapped.feature.unique()):
        frame = mapped[mapped.feature.eq(feature)]
        main = frame[frame.relation_to_current_candidate_description.eq("main_candidate_description")].peak_q99
        counter = frame[frame.relation_to_current_candidate_description.eq("possible_counterexample")].peak_q99
        record = {"feature": feature, "main_images": len(main), "possible_counterexample_images": len(counter)}
        if len(main) and len(counter):
            record.update({
                "main_mean_peak_q99": float(main.mean()),
                "main_median_peak_q99": float(main.median()),
                "counter_mean_peak_q99": float(counter.mean()),
                "counter_median_peak_q99": float(counter.median()),
                "counter_to_main_mean_ratio": float(counter.mean() / main.mean()),
                "all_counters_below_all_main": bool(counter.max() < main.min()),
            })
        comparisons.append(record)
    return category, pd.DataFrame(comparisons)


def main() -> None:
    """读取首批选例，写入独立反馈、强度对照和结果说明。

    Args:
        None: 输出目录由CLI提供。

    Returns:
        None: 不修改原选例或医学图片。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    selected = pd.read_csv(BATCH_ROOT / "selected_images_internal.csv")
    raw, mapped = map_feedback(selected)
    category, comparison = summarize_strength(mapped)
    raw.to_csv(args.output / "medical_feedback_raw.csv", index=False, encoding="utf-8-sig")
    mapped.to_csv(args.output / "medical_feedback_image_mapping.csv", index=False, encoding="utf-8-sig")
    category.to_csv(args.output / "activation_strength_by_feedback_relation.csv", index=False)
    comparison.to_csv(args.output / "main_vs_possible_counterexample.csv", index=False)
    summary = {
        "features": len(raw), "images": len(mapped),
        "feedback_scope": "existing first medical batch only",
        "activation": "peak/Q99 already used to select these high-response examples",
        "comparison": "descriptive mean/median/range only; very small and unequal group sizes",
        "new_images": 0, "new_forward": False, "new_training": False,
        "test_read": False, "external_read": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    explanation = """# 首批医学反馈与激活强度小范围核对（2026-09-15）

## 范围

本轮只把医学生所说的“图1—8”对应到首批已有图片代号，并比较主要候选描述与疑似反例的peak/Q99。
没有新选图、新forward、重训或重分组。“疑似反例”只表示与当前拟议的共同模式不一致。

## 结果

- RA-F0868的疑似反例强度为2.132×Q99，高于6张主要描述图的中位数0.952；不能靠提高激活阈值排除。
- RA-F2388的疑似反例为1.719×Q99，与主要描述图的中位数1.704相当。
- RA-F2444的疑似反例为1.037×Q99，高于主要描述图的中位数0.924。
- RA-F0219的3张疑似反例中位数为1.098×Q99，也不低于5张主要描述图的中位数0.987。
- RA-F1089的单个疑似反例为1.127×Q99，仍在主要描述图1.094—1.803的范围内。
- RA-F0601的单个疑似反例为0.804×Q99，低于主要描述图中位数1.029，但只有1张反例，不足以定阈值。
- RA-F2464的3张疑似反例为0.715—0.936×Q99，都低于5张主要描述图的0.979—4.891×Q99。这批图中存在强度分离迹象，但因图片本就按高响应选取且样本很小，只能在以后用少量未参与本次描述的图复核，不能直接设阈值或删除反例。
- RA-F1385没有逐图指定主要描述与反例，因此本轮不做强度对照。

## 结论

对RA-F0219、RA-F0868、RA-F1089、RA-F2388和RA-F2444，疑似反例并未普遍集中在弱响应区间，因此不能靠简单提高激活阈值解决混杂。
RA-F0601证据不足；RA-F2464在当前8张中有“反例较弱”迹象，但需要后续少量独立图像才能核对。
这些结果支持继续保留混杂和不确定性，不支持立即重训、重聚类、删除Feature或扩大医学阅片。
"""
    (args.output / "结果说明.md").write_text(explanation, encoding="utf-8")
    verification = {
        "features": len(raw), "images": len(mapped),
        "each_feature_has_images_1_to_8_once": True,
        "source_selection_unchanged": True,
        "new_forward": False, "new_images": False,
        "test_read": False, "external_read": False,
    }
    (args.output / "verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(verification, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
