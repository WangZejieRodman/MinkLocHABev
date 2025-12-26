"""
生成用于评估的查询集和数据库集
跨时间段拆分（模拟真实场景）

数据库: Session 160-189 (历史地图)
查询:   Session 190-209 (当前观测)
"""

import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KDTree
import pickle
from pathlib import Path

# 基于时间/session的划分策略
DATABASE_SESSION_START = 180  # 数据库使用的session范围（历史数据）
DATABASE_SESSION_END = 194
QUERY_SESSION_START = 195  # 查询使用的session范围（当前数据）
QUERY_SESSION_END = 209

# 正样本距离阈值（评估时使用）
POSITIVE_THRESHOLD = 7  # 7米内为正样本


def output_to_file(output, filename):
    """保存pickle文件"""
    with open(filename, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ 保存: {filename}")


def construct_query_and_database_sets(base_path, runs_folder,
                                     database_folders, query_folders,
                                     pointcloud_fols, filename, output_name):
    """
    分别构建数据库集和查询集

    Args:
        base_path: 数据集根路径
        runs_folder: 运行文件夹名称
        database_folders: 数据库session列表 (160-189)
        query_folders: 查询session列表 (190-209)
        pointcloud_fols: 点云文件夹名称
        filename: CSV文件名
        output_name: 输出文件前缀

    Returns:
        database_sets: List[Dict], 每个数据库session的点云字典
        query_sets: List[Dict], 每个查询session的点云字典
    """

    print(f"\n{'='*60}")
    print(f"构建数据库集和查询集")
    print(f"{'='*60}")
    print(f"点云文件夹: {pointcloud_fols}")
    print(f"数据库sessions: {database_folders}")
    print(f"查询sessions: {query_folders}")

    # ==================== 第一步：构建数据库集 ====================
    print(f"\n[1/4] 构建数据库集...")
    database_sets = []
    database_coordinates_list = []  # 存储每个数据库session的有效坐标

    for folder in database_folders:
        database = {}
        valid_coordinates = []

        csv_path = os.path.join(base_path, runs_folder, folder, filename)
        if not os.path.exists(csv_path):
            print(f"  ⚠️  跳过数据库 {folder}: CSV文件不存在")
            database_sets.append(database)
            database_coordinates_list.append(np.array([]).reshape(0, 2))
            continue

        folder_path = os.path.join(base_path, runs_folder, folder, pointcloud_fols.strip('/'))
        if not os.path.exists(folder_path):
            print(f"  ⚠️  跳过数据库 {folder}: 点云文件夹不存在")
            database_sets.append(database)
            database_coordinates_list.append(np.array([]).reshape(0, 2))
            continue

        df_locations = pd.read_csv(csv_path, sep=',')
        df_locations['timestamp'] = runs_folder + folder + pointcloud_fols + df_locations['timestamp'].astype(str) + '.bin'
        df_locations = df_locations.rename(columns={'timestamp': 'file'})

        for index, row in df_locations.iterrows():
            full_path = os.path.join(base_path, row['file'])
            if not os.path.exists(full_path):
                continue

            database[len(database.keys())] = {
                'query': row['file'],
                'northing': row['northing'],
                'easting': row['easting']
            }

            # 同时记录有效坐标，确保索引一致
            valid_coordinates.append([row['northing'], row['easting']])

        database_sets.append(database)

        # 转换为numpy数组
        if valid_coordinates:
            database_coordinates_list.append(np.array(valid_coordinates))
        else:
            database_coordinates_list.append(np.array([]).reshape(0, 2))

        print(f"  ✓ 数据库 {folder}: {len(database)} 条目")

    # ==================== 第二步：构建KDTree ====================
    print(f"\n[2/4] 构建KDTree索引...")
    database_trees = []
    for i, coords in enumerate(database_coordinates_list):
        if len(coords) > 0:
            database_tree = KDTree(coords)
            database_trees.append(database_tree)
            print(f"  ✓ 数据库 {database_folders[i]}: KDTree含 {len(coords)} 个点")
        else:
            database_trees.append(None)
            print(f"  ⚠️  数据库 {database_folders[i]}: 空")

    # ==================== 第三步：构建查询集 ====================
    print(f"\n[3/4] 构建查询集...")
    query_sets = []
    for folder in query_folders:
        queries = {}

        csv_path = os.path.join(base_path, runs_folder, folder, filename)
        if not os.path.exists(csv_path):
            print(f"  ⚠️  跳过查询 {folder}: CSV文件不存在")
            query_sets.append(queries)
            continue

        folder_path = os.path.join(base_path, runs_folder, folder, pointcloud_fols.strip('/'))
        if not os.path.exists(folder_path):
            print(f"  ⚠️  跳过查询 {folder}: 点云文件夹不存在")
            query_sets.append(queries)
            continue

        df_locations = pd.read_csv(csv_path, sep=',')
        df_locations['timestamp'] = runs_folder + folder + pointcloud_fols + df_locations['timestamp'].astype(str) + '.bin'
        df_locations = df_locations.rename(columns={'timestamp': 'file'})

        for index, row in df_locations.iterrows():
            full_path = os.path.join(base_path, row['file'])
            if not os.path.exists(full_path):
                continue

            queries[len(queries.keys())] = {
                'query': row['file'],
                'northing': row['northing'],
                'easting': row['easting']
            }

        query_sets.append(queries)
        print(f"  ✓ 查询 {folder}: {len(queries)} 条目")

    # ==================== 第四步：计算正样本匹配 ====================
    print(f"\n[4/4] 计算正样本匹配...")

    # 为每个查询添加positives字段（存储在所有数据库中的正样本）
    for j, query_set in enumerate(query_sets):
        for key in query_set.keys():
            # 初始化positives为字典，key是数据库session索引
            query_set[key]['positives'] = {}

    total_positive_pairs = 0
    positive_distribution = {}  # 统计每个数据库session的正样本分布

    # 统计正样本数量（用于计算平均值和比例）
    query_positive_counts = []  # 每个query的正样本数量

    for i, (database_tree, database_set) in enumerate(zip(database_trees, database_sets)):
        if database_tree is None or len(database_set) == 0:
            print(f"  ⚠️  数据库session {database_folders[i]}: 空，跳过")
            continue

        session_positive_count = 0

        for j, query_set in enumerate(query_sets):
            for key in query_set.keys():
                query_coord = np.array([[query_set[key]["northing"], query_set[key]["easting"]]])

                # 在数据库session i中找到距离15米内的正样本
                positive_indices = database_tree.query_radius(query_coord, r=POSITIVE_THRESHOLD)[0].tolist()

                # 存储到查询集中（使用数据库session索引作为key）
                query_set[key]['positives'][i] = positive_indices

                session_positive_count += len(positive_indices)
                total_positive_pairs += len(positive_indices)

        positive_distribution[database_folders[i]] = session_positive_count
        print(f"  ✓ 数据库 {database_folders[i]}: {session_positive_count} 个正样本匹配")

    print(f"\n  总正样本对数: {total_positive_pairs}")

    # ==================== 统计每个query的正样本数量 ====================
    total_database_size = sum(len(db_set) for db_set in database_sets)

    for j, query_set in enumerate(query_sets):
        for key in query_set.keys():
            # 统计该query在所有database sessions中的正样本总数
            query_positive_count = 0
            for i in query_set[key]['positives']:
                query_positive_count += len(query_set[key]['positives'][i])
            query_positive_counts.append(query_positive_count)

    # 计算统计信息
    if len(query_positive_counts) > 0:
        avg_positives_per_query = np.mean(query_positive_counts)
        median_positives_per_query = np.median(query_positive_counts)
        min_positives_per_query = np.min(query_positive_counts)
        max_positives_per_query = np.max(query_positive_counts)
        avg_positive_ratio = avg_positives_per_query / total_database_size * 100

        print(f"\n{'='*60}")
        print(f"正样本统计分析")
        print(f"{'='*60}")
        print(f"总数据库大小: {total_database_size} 个点云")
        print(f"总查询数量: {len(query_positive_counts)} 个查询")
        print(f"平均每个query的正样本数: {avg_positives_per_query:.2f}")
        print(f"中位数正样本数: {median_positives_per_query:.2f}")
        print(f"最小正样本数: {min_positives_per_query}")
        print(f"最大正样本数: {max_positives_per_query}")
        print(f"平均正样本占总数据库比例: {avg_positive_ratio:.2f}%")
        print(f"{'='*60}")

    # ==================== 输出文件 ====================
    output_dir = os.path.dirname(__file__)
    database_filename = os.path.join(output_dir, f'{output_name}_evaluation_database_{DATABASE_SESSION_START}_{DATABASE_SESSION_END}.pickle')
    query_filename = os.path.join(output_dir, f'{output_name}_evaluation_query_{QUERY_SESSION_START}_{QUERY_SESSION_END}.pickle')

    output_to_file(database_sets, database_filename)
    output_to_file(query_sets, query_filename)

    # ==================== 验证索引一致性 ====================
    print(f"\n{'='*60}")
    print(f"验证索引一致性")
    print(f"{'='*60}")

    validation_passed = True
    queries_with_positives = 0
    queries_without_positives = 0

    for j, query_set in enumerate(query_sets):
        for key in query_set.keys():
            has_positive = False

            for i, database_set in enumerate(database_sets):
                if i not in query_set[key]['positives']:
                    continue

                positive_indices = query_set[key]['positives'][i]

                if len(positive_indices) > 0:
                    has_positive = True

                # 验证索引有效性
                for pos_idx in positive_indices:
                    if pos_idx not in database_set:
                        print(f"❌ 错误：查询 {key} 在数据库session {i} 的正样本索引 {pos_idx} 无效")
                        validation_passed = False

            if has_positive:
                queries_with_positives += 1
            else:
                queries_without_positives += 1

    if validation_passed:
        print(f"✅ 所有索引验证通过!")
    else:
        print(f"❌ 存在无效索引!")

    print(f"\n有正样本的查询数: {queries_with_positives}")
    print(f"无正样本的查询数: {queries_without_positives}")

    # ==================== 统计摘要 ====================
    print(f"\n{'='*60}")
    print(f"最终统计")
    print(f"{'='*60}")

    total_db_entries = sum(len(db_set) for db_set in database_sets)
    total_query_entries = sum(len(query_set) for query_set in query_sets)

    print(f"数据库集: {len(database_sets)} sessions, {total_db_entries} 条目")
    print(f"查询集: {len(query_sets)} sessions, {total_query_entries} 条目")
    print(f"总正样本对数: {total_positive_pairs}")
    print(f"平均每查询正样本数: {total_positive_pairs / max(total_query_entries, 1):.1f}")

    print(f"\n正样本分布（按数据库session）:")
    for db_session, count in positive_distribution.items():
        print(f"  Session {db_session}: {count} 匹配")

    print(f"\n{'='*60}")
    print(f"应用场景")
    print(f"{'='*60}")
    print(f"数据库 ({DATABASE_SESSION_START}-{DATABASE_SESSION_END}): 历史点云作为参考地图")
    print(f"查询 ({QUERY_SESSION_START}-{QUERY_SESSION_END}): 当前观测用于定位")
    print(f"任务: 在历史地图中为当前观测找到匹配位置")

    return database_sets, query_sets


# ==================== 主执行部分 ====================
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"生成评估数据集：跨时间段拆分")
    print(f"{'='*60}")

    # 路径配置
    base_path = "/home/wzj/pan2/Chilean_Underground_Mine_Dataset_Many_Times/"
    runs_folder = "chilean_NoRot_NoScale/"
    pointcloud_fols = "/pointcloud_20m_10overlap/"
    filename = "pointcloud_locations_20m_10overlap.csv"
    output_name = "chilean"

    # 验证基础路径
    path = os.path.join(base_path, runs_folder)
    print(f"基础路径: {path}")

    if not os.path.exists(path):
        print(f"❌ 错误: 基础路径不存在: {path}!")
        exit(1)

    # 获取所有session folders
    all_folders = sorted(os.listdir(path))
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

    # 划分数据库和查询sessions
    database_folders = []
    query_folders = []

    for folder in valid_folders:
        session_num = int(folder)
        if DATABASE_SESSION_START <= session_num <= DATABASE_SESSION_END:
            database_folders.append(folder)
        elif QUERY_SESSION_START <= session_num <= QUERY_SESSION_END:
            query_folders.append(folder)

    print(f"\n{'='*60}")
    print(f"Session划分")
    print(f"{'='*60}")
    print(f"数据库sessions ({DATABASE_SESSION_START}-{DATABASE_SESSION_END}): {len(database_folders)} sessions")
    if database_folders:
        print(f"  Sessions: {', '.join(database_folders)}")
    print(f"查询sessions ({QUERY_SESSION_START}-{QUERY_SESSION_END}): {len(query_folders)} sessions")
    if query_folders:
        print(f"  Sessions: {', '.join(query_folders)}")

    # 执行构建
    database_sets, query_sets = construct_query_and_database_sets(
        base_path,
        runs_folder,
        database_folders,
        query_folders,
        pointcloud_fols,
        filename,
        output_name
    )

    print(f"\n{'='*60}")
    print(f"✓ 完成!")
    print(f"{'='*60}")
    print(f"生成的文件:")
    print(f"  1. {output_name}_evaluation_database_{DATABASE_SESSION_START}_{DATABASE_SESSION_END}.pickle")
    print(f"  2. {output_name}_evaluation_query_{QUERY_SESSION_START}_{QUERY_SESSION_END}.pickle")