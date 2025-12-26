# 为Chilean数据集生成训练和测试查询字典
# 基于时间段划分：训练集 session 100-159，测试集 session 160-209

import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KDTree
import pickle
import random

from datasets.base_datasets import TrainingTuple

# 数据集路径配置
BASE_PATH = "/home/wzj/pan2/Chilean_Underground_Mine_Dataset_Many_Times/"
RUNS_FOLDER = "chilean_NoRot_NoScale/"
FILENAME = "pointcloud_locations_20m_10overlap.csv"
POINTCLOUD_FOLS = "/pointcloud_20m_10overlap/"

# 基于时间/session的划分策略
TRAIN_SESSION_START = 100
TRAIN_SESSION_END = 159
TEST_SESSION_START = 160
TEST_SESSION_END = 179

# 正负样本距离阈值
POSITIVE_THRESHOLD = 7  # 7米内为正样本
NEGATIVE_THRESHOLD = 35  # 35米外为负样本


def check_in_test_set_by_session(session_id):
    """基于session ID判断是否属于测试集"""
    try:
        session_num = int(session_id)
        return TEST_SESSION_START <= session_num <= TEST_SESSION_END
    except ValueError:
        return False


def check_in_train_set_by_session(session_id):
    """基于session ID判断是否属于训练集"""
    try:
        session_num = int(session_id)
        return TRAIN_SESSION_START <= session_num <= TRAIN_SESSION_END
    except ValueError:
        return False


def construct_query_dict(df_centroids, base_path, filename, ind_nn_r=POSITIVE_THRESHOLD, ind_r_r=NEGATIVE_THRESHOLD):
    """
    构建查询字典，使用TrainingTuple格式

    Args:
        df_centroids: 包含文件路径和坐标的DataFrame
        base_path: 数据集根路径
        filename: 输出pickle文件名
        ind_nn_r: 正样本距离阈值（米）
        ind_r_r: 负样本距离阈值（米）
    """
    # 构建KDTree用于邻域查询
    tree = KDTree(df_centroids[['northing', 'easting']])

    # 查询正样本（距离小于ind_nn_r）和非负样本（距离小于ind_r_r）
    ind_nn = tree.query_radius(df_centroids[['northing', 'easting']], r=ind_nn_r)
    ind_r = tree.query_radius(df_centroids[['northing', 'easting']], r=ind_r_r)

    queries = {}
    for anchor_ndx in range(len(ind_nn)):
        anchor_pos = np.array(df_centroids.iloc[anchor_ndx][['northing', 'easting']])
        query = df_centroids.iloc[anchor_ndx]["file"]

        # 从文件路径提取timestamp
        scan_filename = os.path.split(query)[1]
        assert os.path.splitext(scan_filename)[1] == '.bin', f"期望.bin文件: {scan_filename}"
        timestamp = int(os.path.splitext(scan_filename)[0])

        # 获取正样本索引（排除自身）
        positives = ind_nn[anchor_ndx]
        positives = positives[positives != anchor_ndx]
        positives = np.sort(positives)

        # 获取非负样本索引（距离小于ind_r_r的所有样本）
        non_negatives = ind_r[anchor_ndx]
        non_negatives = np.sort(non_negatives)

        # 使用TrainingTuple格式
        queries[anchor_ndx] = TrainingTuple(
            id=anchor_ndx,
            timestamp=timestamp,
            rel_scan_filepath=query,
            positives=positives,
            non_negatives=non_negatives,
            position=anchor_pos
        )

    # 保存为pickle文件
    file_path = os.path.join(os.path.dirname(__file__), filename)
    with open(file_path, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"完成: {filename}")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("生成Chilean数据集训练和测试查询字典")
    print("=" * 60)

    # 验证基础路径
    full_path = os.path.join(BASE_PATH, RUNS_FOLDER)
    print(f"数据集路径: {full_path}")

    if not os.path.exists(full_path):
        print(f"❌ 错误: 数据集路径不存在: {full_path}")
        exit(1)

    # 获取所有session文件夹
    all_folders = sorted(os.listdir(full_path))
    print(f"找到 {len(all_folders)} 个文件夹")

    # 筛选出有效的session folders（数字命名）
    valid_folders = []
    for folder in all_folders:
        if not folder.startswith('.') and folder.isdigit():
            valid_folders.append(folder)

    valid_folders.sort(key=int)
    print(f"有效session文件夹: {len(valid_folders)}")

    if len(valid_folders) > 0:
        print(f"Session范围: {valid_folders[0]} - {valid_folders[-1]}")

    # 划分训练和测试sessions
    train_folders = []
    test_folders = []

    for folder in valid_folders:
        session_num = int(folder)
        if TRAIN_SESSION_START <= session_num <= TRAIN_SESSION_END:
            train_folders.append(folder)
        elif TEST_SESSION_START <= session_num <= TEST_SESSION_END:
            test_folders.append(folder)

    print(f"\n{'=' * 60}")
    print(f"Session划分")
    print(f"{'=' * 60}")
    print(f"训练sessions ({TRAIN_SESSION_START}-{TRAIN_SESSION_END}): {len(train_folders)} sessions")
    print(f"测试sessions ({TEST_SESSION_START}-{TEST_SESSION_END}): {len(test_folders)} sessions")

    # 初始化DataFrame
    df_train = pd.DataFrame(columns=['file', 'northing', 'easting'], dtype=object)
    df_test = pd.DataFrame(columns=['file', 'northing', 'easting'], dtype=object)

    processed_folders = 0
    total_processed_files = 0
    train_files_count = 0
    test_files_count = 0

    # 处理所有相关的sessions（训练+测试）
    all_relevant_folders = train_folders + test_folders

    for folder in all_relevant_folders:
        csv_path = os.path.join(BASE_PATH, RUNS_FOLDER, folder, FILENAME)

        if not os.path.exists(csv_path):
            print(f"跳过 {folder}: CSV文件不存在")
            continue

        # 检查对应的pointcloud文件夹是否存在
        pointcloud_folder_path = os.path.join(BASE_PATH, RUNS_FOLDER, folder, POINTCLOUD_FOLS.strip('/'))
        if not os.path.exists(pointcloud_folder_path):
            print(f"跳过 {folder}: 点云文件夹不存在")
            continue

        print(f"处理session {folder}...")

        df_locations = pd.read_csv(csv_path, sep=',')

        # 构建文件路径
        df_locations['timestamp'] = RUNS_FOLDER + folder + POINTCLOUD_FOLS + df_locations['timestamp'].astype(
            str) + '.bin'
        df_locations = df_locations.rename(columns={'timestamp': 'file'})

        folder_processed_files = 0
        session_num = int(folder)

        for index, row in df_locations.iterrows():
            # 验证文件是否真实存在
            full_file_path = os.path.join(BASE_PATH, row['file'])
            if not os.path.exists(full_file_path):
                print(f"警告: 文件不存在: {full_file_path}")
                continue

            # 根据session ID分配到训练集或测试集
            if check_in_test_set_by_session(session_num):
                df_test = pd.concat([df_test, pd.DataFrame([row])], ignore_index=True)
                test_files_count += 1
            elif check_in_train_set_by_session(session_num):
                df_train = pd.concat([df_train, pd.DataFrame([row])], ignore_index=True)
                train_files_count += 1

            folder_processed_files += 1
            total_processed_files += 1

        print(f"  处理了 {folder_processed_files} 个文件")
        processed_folders += 1

    print(f"\n{'=' * 60}")
    print(f"处理摘要")
    print(f"{'=' * 60}")
    print(f"处理的sessions: {processed_folders}")
    print(f"总处理文件数: {total_processed_files}")
    print(f"训练文件: {train_files_count} (sessions {TRAIN_SESSION_START}-{TRAIN_SESSION_END})")
    print(f"测试文件: {test_files_count} (sessions {TEST_SESSION_START}-{TEST_SESSION_END})")

    # 验证生成的文件路径
    if len(df_train) > 0:
        sample_file = df_train.iloc[0]['file']
        print(f"\n示例训练文件路径: {sample_file}")

        # 验证文件可以正确加载
        try:
            sample_full_path = os.path.join(BASE_PATH, sample_file)
            pc_data = np.fromfile(sample_full_path, dtype=np.float64)
            print(f"示例点云数据大小: {pc_data.shape}")
            print("✓ 点云文件验证成功!")
        except Exception as e:
            print(f"✗ 加载示例点云文件出错: {e}")

    # 生成查询字典文件
    print(f"\n{'=' * 60}")
    print(f"生成查询字典")
    print(f"{'=' * 60}")

    output_dir = os.path.dirname(__file__)
    if len(df_train) > 0:
        construct_query_dict(df_train, output_dir, "training_queries_chilean.pickle")
        print(f"生成训练查询: {len(df_train)} 个样本")
    else:
        print("警告: 未找到训练数据!")

    if len(df_test) > 0:
        construct_query_dict(df_test, output_dir, "test_queries_chilean.pickle")
        print(f"生成测试查询: {len(df_test)} 个样本")
    else:
        print("警告: 未找到测试数据!")

    print(f"\n{'=' * 60}")
    print(f"生成的文件")
    print(f"{'=' * 60}")
    print(f"训练查询: training_queries_chilean.pickle")
    print(f"测试查询: test_queries_chilean.pickle")
    print(f"\n{'=' * 60}")
    print(f"最终数据划分摘要")
    print(f"{'=' * 60}")
    print(f"训练数据: {len(df_train)} 个点云，来自 {len(train_folders)} 个sessions")
    print(f"测试数据: {len(df_test)} 个点云，来自 {len(test_folders)} 个sessions")
    print(
        f"无数据泄漏: 训练sessions ({TRAIN_SESSION_START}-{TRAIN_SESSION_END}) 和测试sessions ({TEST_SESSION_START}-{TEST_SESSION_END}) 完全分离")