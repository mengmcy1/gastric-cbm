"""整理完整MOCE K敏感性自动分析，生成可直接发送的医学生目录。"""

import argparse
import shutil
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = PROJECT_DIR / "结果" / "MOCE聚类数量分析"


def parse_args():
    parser = argparse.ArgumentParser(description="整理MOCE聚类数量分析医学生提交版")
    parser.add_argument("--analysis-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--output-name", default="医学生提交版_完整K_v1")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.analysis_root).resolve()
    source = root / "自动分析"
    guide = root / "医学生整理概念集筛选与MOCE严格平衡集构建及聚类数量分析说明.docx"
    output = root / args.output_name
    if output.exists():
        raise FileExistsError(f"输出目录已存在，请更换--output-name: {output}")
    if not source.exists() or not guide.exists():
        raise FileNotFoundError("缺少自动分析目录或Word阅读指南")

    output.mkdir(parents=True)
    shutil.copy2(guide, output / "00_MOCE聚类数量分析阅读指南_医学生版.docx")
    shutil.copytree(source, output / "01_全部K自动分析")
    (output / "README.md").write_text(
        """# MOCE聚类数量分析医学生提交版

本目录保留K=10、15、20、25、30、35、40、45、50的全部自动分析材料。

## 阅读顺序

1. 先阅读根目录Word指南。
2. 查看`01_全部K自动分析/K敏感性汇总.xlsx`和四张跨K汇总图。
3. 第一轮优先比较K=20、25、30；K=35用于观察过度拆分。
4. 如需完整敏感性核查，再查看K=10、15、40、45、50。
5. 每个K均按模型和class_0/1分开；class_0为非癌，class_1为癌/高级别。

## 注意

- 不同K、模型和类别的概念编号不能直接对应。
- 先看概念聚类清晰版和重要概念扩展图，再查看重要性、来源和SSC/SDC。
- 本提交版不含K-Means模型和特征缓存；服务器上的原始结果保持不变。
""",
        encoding="utf-8",
    )
    print(f"MOCE医学生提交版已生成: {output}")


if __name__ == "__main__":
    main()
