# 分析旋转不变性测试结果

import pickle
import numpy as np
import os


def load_rotation_results(result_file):
    """加载旋转评估结果"""
    if not os.path.exists(result_file):
        print(f"❌ 错误: 找不到结果文件: {result_file}")
        return None

    with open(result_file, 'rb') as f:
        all_stats = pickle.load(f)

    return all_stats


def print_rotation_analysis(all_stats):
    """打印旋转不变性分析结果"""
    if all_stats is None:
        return

    print("\n" + "=" * 80)
    print("Chilean数据集旋转不变性测试结果")
    print("=" * 80)

    # 提取角度列表并排序
    angles = sorted(all_stats.keys())

    # 打印表头
    print(f"\n{'旋转角度':>10} | {'Top 1% Recall':>14} | {'Recall@1':>10} | "
          f"{'Recall@5':>10} | {'Recall@10':>11} | {'Recall@25':>11}")
    print("-" * 80)

    # 存储基准性能（0度）
    baseline_top1p = all_stats[0]['ave_one_percent_recall']
    baseline_recall1 = all_stats[0]['ave_recall'][0]

    # 打印每个角度的结果
    for angle in angles:
        stats = all_stats[angle]
        top1p_recall = stats['ave_one_percent_recall']
        recall = stats['ave_recall']

        print(f"{angle:>10}° | {top1p_recall:>13.2f}% | {recall[0]:>9.2f}% | "
              f"{recall[4]:>9.2f}% | {recall[9]:>10.2f}% | {recall[24]:>10.2f}%")

    print("-" * 80)

    # 计算统计信息
    print("\n" + "=" * 80)
    print("统计分析")
    print("=" * 80)

    # 基准性能（0度）
    print(f"\n基准性能 (0°):")
    print(f"  Top 1% Recall: {baseline_top1p:.2f}%")
    print(f"  Recall@1:      {baseline_recall1:.2f}%")

    # 平均性能（所有角度）
    all_top1p = [all_stats[angle]['ave_one_percent_recall'] for angle in angles]
    all_recall1 = [all_stats[angle]['ave_recall'][0] for angle in angles]

    mean_top1p = np.mean(all_top1p)
    mean_recall1 = np.mean(all_recall1)

    print(f"\n平均性能 (所有角度):")
    print(f"  Top 1% Recall: {mean_top1p:.2f}%")
    print(f"  Recall@1:      {mean_recall1:.2f}%")

    # 最差性能
    min_top1p = np.min(all_top1p)
    min_recall1 = np.min(all_recall1)
    worst_angle_top1p = angles[np.argmin(all_top1p)]
    worst_angle_recall1 = angles[np.argmin(all_recall1)]

    print(f"\n最差性能:")
    print(f"  Top 1% Recall: {min_top1p:.2f}% (at {worst_angle_top1p}°)")
    print(f"  Recall@1:      {min_recall1:.2f}% (at {worst_angle_recall1}°)")

    # 性能下降
    drop_top1p = baseline_top1p - mean_top1p
    drop_recall1 = baseline_recall1 - mean_recall1
    drop_top1p_pct = (drop_top1p / baseline_top1p) * 100
    drop_recall1_pct = (drop_recall1 / baseline_recall1) * 100

    max_drop_top1p = baseline_top1p - min_top1p
    max_drop_recall1 = baseline_recall1 - min_recall1
    max_drop_top1p_pct = (max_drop_top1p / baseline_top1p) * 100
    max_drop_recall1_pct = (max_drop_recall1 / baseline_recall1) * 100

    print(f"\n相比基准的平均性能下降:")
    print(f"  Top 1% Recall: {drop_top1p:.2f}% (相对下降 {drop_top1p_pct:.1f}%)")
    print(f"  Recall@1:      {drop_recall1:.2f}% (相对下降 {drop_recall1_pct:.1f}%)")

    print(f"\n相比基准的最大性能下降:")
    print(f"  Top 1% Recall: {max_drop_top1p:.2f}% (相对下降 {max_drop_top1p_pct:.1f}%)")
    print(f"  Recall@1:      {max_drop_recall1:.2f}% (相对下降 {max_drop_recall1_pct:.1f}%)")

    # 旋转不变性评估
    print(f"\n" + "=" * 80)
    print("旋转不变性评估")
    print("=" * 80)

    if drop_recall1_pct < 5:
        rating = "优秀 ✓✓✓"
    elif drop_recall1_pct < 10:
        rating = "良好 ✓✓"
    elif drop_recall1_pct < 20:
        rating = "一般 ✓"
    else:
        rating = "较差 ✗"

    print(f"\n基于Recall@1的平均相对下降 ({drop_recall1_pct:.1f}%): {rating}")

    if max_drop_recall1_pct < 10:
        rating_worst = "优秀 ✓✓✓"
    elif max_drop_recall1_pct < 20:
        rating_worst = "良好 ✓✓"
    elif max_drop_recall1_pct < 30:
        rating_worst = "一般 ✓"
    else:
        rating_worst = "较差 ✗"

    print(f"基于Recall@1的最大相对下降 ({max_drop_recall1_pct:.1f}%): {rating_worst}")

    print("\n" + "=" * 80)

    # 保存文本报告
    save_text_report(all_stats, angles, baseline_top1p, baseline_recall1,
                     mean_top1p, mean_recall1, drop_top1p_pct, drop_recall1_pct,
                     max_drop_top1p_pct, max_drop_recall1_pct)


def save_text_report(all_stats, angles, baseline_top1p, baseline_recall1,
                     mean_top1p, mean_recall1, drop_top1p_pct, drop_recall1_pct,
                     max_drop_top1p_pct, max_drop_recall1_pct):
    """保存文本报告"""
    report_file = "rotation_results.txt"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Chilean数据集旋转不变性测试结果\n")
        f.write("=" * 80 + "\n\n")

        # 详细结果表格
        f.write(f"{'旋转角度':>10} | {'Top 1% Recall':>14} | {'Recall@1':>10} | "
                f"{'Recall@5':>10} | {'Recall@10':>11} | {'Recall@25':>11}\n")
        f.write("-" * 80 + "\n")

        for angle in angles:
            stats = all_stats[angle]
            top1p_recall = stats['ave_one_percent_recall']
            recall = stats['ave_recall']

            f.write(f"{angle:>10}° | {top1p_recall:>13.2f}% | {recall[0]:>9.2f}% | "
                    f"{recall[4]:>9.2f}% | {recall[9]:>10.2f}% | {recall[24]:>10.2f}%\n")

        f.write("-" * 80 + "\n\n")

        # 统计摘要
        f.write("=" * 80 + "\n")
        f.write("统计摘要\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"基准性能 (0°):\n")
        f.write(f"  Top 1% Recall: {baseline_top1p:.2f}%\n")
        f.write(f"  Recall@1:      {baseline_recall1:.2f}%\n\n")

        f.write(f"平均性能 (所有角度):\n")
        f.write(f"  Top 1% Recall: {mean_top1p:.2f}%\n")
        f.write(f"  Recall@1:      {mean_recall1:.2f}%\n\n")

        f.write(f"相比基准的平均性能下降:\n")
        f.write(f"  Top 1% Recall: 相对下降 {drop_top1p_pct:.1f}%\n")
        f.write(f"  Recall@1:      相对下降 {drop_recall1_pct:.1f}%\n\n")

        f.write(f"相比基准的最大性能下降:\n")
        f.write(f"  Top 1% Recall: 相对下降 {max_drop_top1p_pct:.1f}%\n")
        f.write(f"  Recall@1:      相对下降 {max_drop_recall1_pct:.1f}%\n")

    print(f"\n✓ 文本报告已保存到: {report_file}")


if __name__ == "__main__":
    # 加载结果
    result_file = "/home/wzj/pan1/MinkLocBev_Chilean_原始点云/training/rotation_results_model_MinkLocBEV_20251221_1346_final.pkl"
    all_stats = load_rotation_results(result_file)

    # 分析并打印结果
    if all_stats is not None:
        print_rotation_analysis(all_stats)