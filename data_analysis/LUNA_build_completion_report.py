"""Build a Chinese report for LUNA16 generalisation and complete-CT detection."""

from __future__ import print_function

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTERNAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_reports"
    / "lidc_luna16_generalization"
)
DEFAULT_ROI_SUMMARY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "luna16_labeled_external_rois"
    / "luna16_labeled_external_roi_summary.json"
)
DEFAULT_CANDIDATE_METRICS = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_results"
    / "luna16_candidate_detector"
    / "subset9_metrics.json"
)
DEFAULT_DETECTION_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_reports"
    / "luna16_full_ct_detection_subset9"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_reports"
    / "luna16_validation_and_full_ct_report.md"
)


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def metric(value):
    if value in (None, ""):
        return ""
    return "{:.4f}".format(float(value))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build the LUNA16 validation and complete-CT report."
    )
    parser.add_argument("--external-dir", type=Path, default=DEFAULT_EXTERNAL_DIR)
    parser.add_argument("--roi-summary", type=Path, default=DEFAULT_ROI_SUMMARY)
    parser.add_argument(
        "--candidate-metrics",
        type=Path,
        default=DEFAULT_CANDIDATE_METRICS,
    )
    parser.add_argument("--detection-dir", type=Path, default=DEFAULT_DETECTION_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    roi_summary = read_json(args.roi_summary)
    generalisation = read_csv_rows(
        args.external_dir / "luna16_generalization_summary.csv"
    )
    candidate_payload = read_json(args.candidate_metrics)
    candidate_metrics = candidate_payload.get("metrics", candidate_payload)
    detection = read_json(
        args.detection_dir / "luna16_full_ct_proposal_summary.json"
    )
    failure_summary = read_json(
        args.detection_dir / "luna16_full_ct_failure_summary.json"
    )

    lines = [
        "# LUNA16 泛化验证与完整 CT 自动结节检测报告",
        "",
        "## 1. 验证定位",
        "",
        (
            "LUNA16 来源于 LIDC-IDRI，因此本次验证用于检验不同预处理、坐标重采样"
            "和 ROI 导出流程下的模型稳定性，不等同于独立医院或独立患者队列验证。"
        ),
        "",
        "## 2. 标签匹配与外部 ROI",
        "",
        "- LUNA16 标注结节数：{}".format(roi_summary["luna_annotation_count"]),
        "- 一对一坐标匹配成功：{}".format(roi_summary["matched_annotation_count"]),
        "- 未匹配：{}".format(roi_summary["unmatched_annotation_count"]),
        "- 匹配距离中位数：{:.3f} mm".format(
            roi_summary["median_match_distance_mm"]
        ),
        "- 匹配距离 P95：{:.3f} mm".format(
            roi_summary["p95_match_distance_mm"]
        ),
        "- 测试集导出 ROI：{}".format(roi_summary["exported_roi_count"]),
        "- 导出失败：{}".format(roi_summary["export_failure_count"]),
        "",
        "## 3. LUNA16 域模型泛化结果",
        "",
        "| 输入 | 模型 | 任务 | 样本数 | Accuracy | F1 | AUC-ROC | 内部测试 F1 | F1 差值 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in generalisation:
        lines.append(
            "| {input_dim} | {model} | {task} | {sample_count} | {accuracy} | "
            "{f1} | {auc} | {internal_f1} | {gap} |".format(
                input_dim=row["input_dim"],
                model=row["model"],
                task=row["task"],
                sample_count=row["sample_count"],
                accuracy=metric(row["accuracy"]),
                f1=metric(row["f1"]),
                auc=metric(row["auc_roc"]),
                internal_f1=metric(row.get("internal_f1")),
                gap=metric(row.get("f1_gap_external_minus_internal")),
            )
        )

    lines.extend([
        "",
        "## 4. LUNA16 候选结节假阳性抑制模型",
        "",
        "- 架构：轻量 3D CNN + 深层 3D CBAM。",
        "- 数据划分：subset0-7 训练、subset8 验证、subset9 测试。",
        "- 测试样本数：{}".format(candidate_metrics["sample_count"]),
        "- Accuracy：{}".format(metric(candidate_metrics["accuracy"])),
        "- Precision：{}".format(metric(candidate_metrics["precision"])),
        "- Recall：{}".format(metric(candidate_metrics["recall"])),
        "- F1：{}".format(metric(candidate_metrics["f1"])),
        "- AUC-ROC：{}".format(metric(candidate_metrics["auc_roc"])),
        "",
        "## 5. 完整 CT 自动结节检测",
        "",
        "- 评估 CT 数：{}".format(detection["scan_count"]),
        "- 标注结节数：{}".format(detection["annotation_count"]),
        "- 候选生成召回率：{}".format(metric(detection["proposal_sensitivity"])),
        "- 默认阈值：{}".format(detection["nodule_threshold"]),
        "- 最终检测召回率：{}".format(metric(detection["detection_sensitivity"])),
        "- 平均检测数/例：{}".format(
            metric(detection["average_detections_per_scan"])
        ),
        "- 平均假阳性/例：{}".format(
            metric(detection["average_false_positive_detections_per_scan"])
        ),
        "- 候选生成阶段漏检：{}".format(
            failure_summary["proposal_miss_count"]
        ),
        "- CBAM 候选筛选阶段漏检：{}".format(
            failure_summary["candidate_filter_miss_count"]
        ),
        "",
        "### 检测工作点",
        "",
        "| 结节概率阈值 | 敏感度 | 平均检测数/例 | 平均假阳性/例 |",
        "|---:|---:|---:|---:|",
    ])
    for point in detection["operating_points"]:
        lines.append(
            "| {threshold:.2f} | {sensitivity} | {detections} | {false_positives} |".format(
                threshold=float(point["nodule_threshold"]),
                sensitivity=metric(point["sensitivity"]),
                detections=metric(point["average_detections_per_scan"]),
                false_positives=metric(
                    point["average_false_positives_per_scan"]
                ),
            )
        )

    lines.extend([
        "",
        "### 结节大小亚组",
        "",
        "| 直径组 | 标注数 | 候选召回率 | 最终检测召回率 |",
        "|---|---:|---:|---:|",
    ])
    for group in failure_summary["diameter_groups"]:
        lines.append(
            "| {group} | {count} | {proposal} | {detection} |".format(
                group=group["diameter_group"],
                count=group["annotation_count"],
                proposal=metric(group["proposal_sensitivity"]),
                detection=metric(group["detection_sensitivity"]),
            )
        )

    lines.extend([
        "",
        "## 6. 完整处理链路",
        "",
        "1. 读取完整 DICOM/MHD/NIfTI CT，并重采样到 1 mm 各向同性空间。",
        "2. 分割肺实质，在肺内生成 3D LoG 与轴向连通域高召回候选。",
        "3. 使用 LUNA16 训练的 3D CBAM 模型进行候选结节评分和假阳性抑制。",
        "4. 对保留候选裁剪 64×64×64 ROI，执行良恶性分类和三级风险分层。",
        "5. 导出世界坐标、概率、风险标签、ROI、总览图和结构化元数据。",
        "",
        "## 7. 局限性与下一步",
        "",
        "- LUNA16 不是独立临床中心，仍需外部医院队列进行真正的跨中心验证。",
        "- 完整 CT 检测仍需根据目标临床场景选择敏感度与假阳性之间的工作点。",
        "- 对漏检和高置信假阳性应继续进行结节大小、钙化、胸膜旁位置和血管交叉等失败模式分析。",
        "- 当前系统属于研究原型，不能直接用于临床诊断。",
        "",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
