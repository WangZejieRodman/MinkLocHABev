# Chilean数据集旋转不变性评估脚本
# 修改版: 适配 MinkLocBEV (32维特征)

import sys
import os

# 添加项目根目录到path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from sklearn.neighbors import KDTree
import numpy as np
import pickle
import torch
import MinkowskiEngine as ME
import tqdm

from models.model_factory import model_factory
from misc.utils import TrainingParams
from datasets.pointnetvlad.pnv_raw import PNVPointCloudLoader
# 确保 datasets/rotation_utils.py 存在
from datasets.rotation_utils import rotate_point_cloud_z


def evaluate_chilean_with_rotation(model, device, params: TrainingParams, rotation_angles,
                                   log: bool = False, show_progress: bool = False):
    """
    使用多个旋转角度评估Chilean数据集

    Args:
        model: 训练好的模型
        device: 计算设备
        params: 训练参数
        rotation_angles: list - 测试的旋转角度列表（度）
        log: 是否记录详细日志
        show_progress: 是否显示进度条

    Returns:
        all_stats: dict - 每个角度的评估统计
    """
    # 加载Chilean评估数据集
    eval_database_file = 'chilean_evaluation_database_180_194.pickle'
    eval_query_file = 'chilean_evaluation_query_195_209.pickle'

    dataset_chilean_path = os.path.join(project_root, 'datasets', 'chilean')
    database_path = os.path.join(dataset_chilean_path, eval_database_file)
    query_path = os.path.join(dataset_chilean_path, eval_query_file)

    print(f"\n加载Chilean评估数据集:")
    print(f"  Database: {database_path}")
    print(f"  Query: {query_path}")

    if not os.path.exists(database_path):
        print(f"❌ 错误: 找不到database文件: {database_path}")
        return None

    if not os.path.exists(query_path):
        print(f"❌ 错误: 找不到query文件: {query_path}")
        return None

    with open(database_path, 'rb') as f:
        database_sets = pickle.load(f)

    with open(query_path, 'rb') as f:
        query_sets = pickle.load(f)

    print(f"✓ Database sessions: {len(database_sets)}")
    print(f"✓ Query sessions: {len(query_sets)}")

    # 对每个旋转角度进行评估
    all_stats = {}

    for angle in rotation_angles:
        print(f"\n{'=' * 60}")
        print(f"测试旋转角度: {angle}°")
        print(f"{'=' * 60}")

        stats = evaluate_dataset_chilean_rotation(
            model, device, params, database_sets, query_sets,
            rotation_angle=angle, log=log, show_progress=show_progress
        )

        all_stats[angle] = stats

        if stats is not None:
            print(f"旋转 {angle}° - Top 1% Recall: {stats['ave_one_percent_recall']:.2f}%")
            print(f"旋转 {angle}° - Recall@1: {stats['ave_recall'][0]:.2f}%")

    return all_stats


def evaluate_dataset_chilean_rotation(model, device, params: TrainingParams,
                                      database_sets, query_sets, rotation_angle=0,
                                      log: bool = False, show_progress: bool = False):
    """
    在Chilean数据集上运行评估（支持旋转query点云）

    Args:
        rotation_angle: float - query点云的旋转角度（度），database不旋转
    """
    model.eval()

    # ==================== 第一步：计算所有embeddings ====================
    print(f"\n{'=' * 60}")
    print(f"计算Embeddings (rotation={rotation_angle}°)")
    print(f"{'=' * 60}")

    # 计算所有database embeddings（不旋转）
    database_embeddings = []
    for i, db_set in enumerate(
            tqdm.tqdm(database_sets, disable=not show_progress, desc='Computing database embeddings')):
        if len(db_set) > 0:
            database_embeddings.append(get_latent_vectors(model, db_set, device, params, rotation_angle=0))
        else:
            database_embeddings.append(np.array([]).reshape(0, 256))

    # 计算所有query embeddings（旋转指定角度）
    query_embeddings = []
    for i, query_set in enumerate(tqdm.tqdm(query_sets, disable=not show_progress,
                                            desc=f'Computing query embeddings (rot={rotation_angle}°)')):
        if len(query_set) > 0:
            query_embeddings.append(get_latent_vectors(model, query_set, device, params,
                                                       rotation_angle=rotation_angle))
        else:
            query_embeddings.append(np.array([]).reshape(0, 256))

    # ==================== 第二步：合并所有database embeddings ====================
    all_database_embeddings = []
    database_to_session_map = []

    for i, db_emb in enumerate(database_embeddings):
        if len(db_emb) > 0:
            all_database_embeddings.append(db_emb)
            for local_idx in range(len(db_emb)):
                database_to_session_map.append((i, local_idx))

    if len(all_database_embeddings) == 0:
        print("❌ 错误：没有database embeddings")
        return None

    database_output = np.vstack(all_database_embeddings)
    print(f"\n合并后的数据库大小: {len(database_output)} 个embeddings")

    # 构建KDTree用于最近邻搜索
    database_nbrs = KDTree(database_output)

    # ==================== 第三步：对每个query进行检索和评估 ====================
    print(f"\n{'=' * 60}")
    print(f"评估检索性能 (rotation={rotation_angle}°)")
    print(f"{'=' * 60}")

    num_neighbors = 25
    recall = np.zeros(num_neighbors)
    one_percent_recall_list = []
    num_evaluated = 0

    threshold = max(int(round(len(database_output) / 100.0)), 1)

    # 遍历所有query sessions
    for j, query_set in enumerate(query_sets):
        if len(query_set) == 0:
            continue

        queries_output = query_embeddings[j]

        for query_idx in range(len(queries_output)):
            query_details = query_set[query_idx]

            if 'positives' not in query_details:
                continue

            # 构建全局正样本索引列表
            true_neighbors_global = []

            for db_session_idx in query_details['positives']:
                local_positive_indices = query_details['positives'][db_session_idx]

                for local_idx in local_positive_indices:
                    for global_idx, (sess_idx, loc_idx) in enumerate(database_to_session_map):
                        if sess_idx == db_session_idx and loc_idx == local_idx:
                            true_neighbors_global.append(global_idx)
                            break

            if len(true_neighbors_global) == 0:
                continue

            num_evaluated += 1

            # 查找最近的num_neighbors个邻居
            distances, indices = database_nbrs.query(np.array([queries_output[query_idx]]), k=num_neighbors)

            # Recall计算
            for k in range(len(indices[0])):
                if indices[0][k] in true_neighbors_global:
                    recall[k:] += 1
                    break

            # 计算top 1% recall
            top_1_percent = indices[0][0:threshold]
            if len(set(top_1_percent).intersection(set(true_neighbors_global))) > 0:
                one_percent_recall_list.append(1.0)
            else:
                one_percent_recall_list.append(0.0)

    # ==================== 第四步：计算最终统计 ====================
    if num_evaluated == 0:
        print("❌ 错误：没有有效的查询被评估")
        return None

    ave_recall = (recall / float(num_evaluated)) * 100.0
    ave_one_percent_recall = np.mean(one_percent_recall_list) * 100.0

    stats = {
        'ave_one_percent_recall': ave_one_percent_recall,
        'ave_recall': ave_recall,
        'num_evaluated': num_evaluated,
        'rotation_angle': rotation_angle
    }

    return stats


def get_latent_vectors(model, point_cloud_set, device, params: TrainingParams, rotation_angle=0):
    """
    计算点云集合的embeddings（支持旋转）
    """
    if params.debug:
        embeddings = np.random.rand(len(point_cloud_set), 256)
        return embeddings

    pc_loader = PNVPointCloudLoader()
    model.eval()
    embeddings = None

    for i, elem_ndx in enumerate(point_cloud_set):
        pc_file_path = os.path.join(params.dataset_folder, point_cloud_set[elem_ndx]["query"])
        pc = pc_loader(pc_file_path)

        # 应用旋转
        if rotation_angle != 0:
            pc = rotate_point_cloud_z(pc, rotation_angle)

        pc = torch.tensor(pc)
        embedding = compute_embedding(model, pc, device, params)

        if embeddings is None:
            embeddings = np.zeros((len(point_cloud_set), embedding.shape[1]), dtype=embedding.dtype)
        embeddings[i] = embedding

        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    return embeddings


def compute_embedding(model, pc, device, params: TrainingParams):
    """计算单个点云的embedding（BEV专用）"""
    coords, feats = params.model_params.quantizer(pc)

    with torch.no_grad():
        bcoords = ME.utils.batched_coordinates([coords])
        batch = {'coords': bcoords.to(device), 'features': feats.to(device)}

        y = model(batch)
        embedding = y['global'].detach().cpu().numpy()

    return embedding


def save_rotation_results(all_stats, output_file):
    """保存旋转评估结果"""
    with open(output_file, 'wb') as f:
        pickle.dump(all_stats, f)
    print(f"\n✓ 结果已保存到: {output_file}")


if __name__ == "__main__":
    # 设置参数
    class Args:
        def __init__(self):
            # 这里指向我们刚刚创建的 BEV 配置文件
            self.config = '../config/config_chilean_bev.txt'
            self.model_config = '../models/minkloc_bev.txt'

            # 权重文件: 请修改为你刚刚训练生成的 .pth 文件路径
            # 例如:
            self.weights = '/home/wzj/pan1/MinkLocHABev/weights/model_MinkLocBEV_20251221_1045_final.pth'
            self.debug = False


    args = Args()

    print('=' * 60)
    print('Chilean数据集旋转不变性测试')
    print('=' * 60)
    print(f'配置文件: {args.config}')
    print(f'模型配置: {args.model_config}')
    print(f'权重文件: {args.weights}')
    print(f'Debug模式: {args.debug}')
    print('')

    # 检查权重文件是否存在
    if args.weights and not os.path.exists(args.weights):
        print(f"⚠️ 警告: 找不到权重文件 {args.weights}")
        print("请在代码中修改 args.weights 指向正确的 .pth 文件")
        exit(1)

    params = TrainingParams(args.config, args.model_config, debug=args.debug)
    params.print()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f'设备: {device}\n')

    # 创建模型
    model = model_factory(params.model_params)

    if args.weights is not None:
        print(f'加载权重: {args.weights}')
        model.load_state_dict(torch.load(args.weights, map_location=device))
    else:
        print("⚠️ 警告: 未加载权重，正在使用随机初始化模型进行测试！")

    model.to(device)

    # 定义测试角度
    # 测试一组角度来验证旋转不变性
    rotation_angles = [0, 5, 10, 15, 30, 45, 60, 90, 135, 180]

    print(f"\n测试角度: {rotation_angles}")

    # 运行旋转评估
    all_stats = evaluate_chilean_with_rotation(
        model, device, params, rotation_angles,
        log=False, show_progress=True
    )

    # 保存结果
    if all_stats is not None:
        output_file = "rotation_results.pkl"
        save_rotation_results(all_stats, output_file)