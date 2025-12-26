# Chilean数据集评估脚本
# 评估协议：跨session划分（database: 160-189, query: 190-209）

from sklearn.neighbors import KDTree
import numpy as np
import pickle
import os
import torch
import MinkowskiEngine as ME
import tqdm

from models.model_factory import model_factory
from misc.utils import TrainingParams
from datasets.pointnetvlad.pnv_raw import PNVPointCloudLoader


def evaluate_chilean(model, device, params: TrainingParams, log: bool = False, show_progress: bool = False):
    """
    在Chilean数据集上运行评估

    Args:
        model: 训练好的模型
        device: 计算设备
        params: 训练参数
        log: 是否记录详细日志
        show_progress: 是否显示进度条

    Returns:
        stats: 评估统计信息字典
    """

    # Chilean数据集的评估文件
    eval_database_file = 'chilean_evaluation_database_180_194.pickle'
    eval_query_file = 'chilean_evaluation_query_195_209.pickle'

    # 文件路径（在datasets/chilean目录下）
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

    # 运行评估
    stats = evaluate_dataset_chilean(model, device, params, database_sets, query_sets,
                                     log=log, show_progress=show_progress)

    return stats


def evaluate_dataset_chilean(model, device, params: TrainingParams, database_sets, query_sets,
                             log: bool = False, show_progress: bool = False):
    """
    在Chilean数据集上运行评估
    """

    model.eval()

    # ==================== 第一步：计算所有embeddings ====================
    print(f"\n{'=' * 60}")
    print(f"计算Embeddings")
    print(f"{'=' * 60}")

    # 计算所有database embeddings
    database_embeddings = []
    for i, db_set in enumerate(
            tqdm.tqdm(database_sets, disable=not show_progress, desc='Computing database embeddings')):
        if len(db_set) > 0:
            database_embeddings.append(get_latent_vectors(model, db_set, device, params))
        else:
            database_embeddings.append(np.array([]).reshape(0, 256))

    # 计算所有query embeddings
    query_embeddings = []
    for i, query_set in enumerate(tqdm.tqdm(query_sets, disable=not show_progress, desc='Computing query embeddings')):
        if len(query_set) > 0:
            query_embeddings.append(get_latent_vectors(model, query_set, device, params))
        else:
            query_embeddings.append(np.array([]).reshape(0, 256))

    # ==================== 第二步：合并所有database embeddings ====================
    # 将所有database sessions的embeddings合并成一个大的数据库
    all_database_embeddings = []
    database_to_session_map = []  # 记录每个embedding属于哪个session和session内索引

    for i, db_emb in enumerate(database_embeddings):
        if len(db_emb) > 0:
            all_database_embeddings.append(db_emb)
            # 记录：(session_idx, local_idx_in_session)
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
    print(f"评估检索性能")
    print(f"{'=' * 60}")

    num_neighbors = 25
    recall = np.zeros(num_neighbors)
    one_percent_recall_list = []
    num_evaluated = 0

    # Top 1%的阈值
    threshold = max(int(round(len(database_output) / 100.0)), 1)
    print(f"Top 1% 阈值: {threshold} (数据库大小的1%)")

    # 遍历所有query sessions
    for j, query_set in enumerate(query_sets):
        if len(query_set) == 0:
            continue

        queries_output = query_embeddings[j]

        # 对当前query session中的每个query进行评估
        for query_idx in range(len(queries_output)):
            query_details = query_set[query_idx]

            # 获取该query的所有正样本（在全局database中的位置）
            if 'positives' not in query_details:
                continue

            # 构建全局正样本索引列表
            true_neighbors_global = []

            # 遍历该query在各个database session中的正样本
            for db_session_idx in query_details['positives']:
                local_positive_indices = query_details['positives'][db_session_idx]

                # 将局部索引转换为全局索引
                for local_idx in local_positive_indices:
                    # 在database_to_session_map中找到对应的全局索引
                    for global_idx, (sess_idx, loc_idx) in enumerate(database_to_session_map):
                        if sess_idx == db_session_idx and loc_idx == local_idx:
                            true_neighbors_global.append(global_idx)
                            break

            if len(true_neighbors_global) == 0:
                continue

            num_evaluated += 1

            # 查找最近的num_neighbors个邻居
            distances, indices = database_nbrs.query(np.array([queries_output[query_idx]]), k=num_neighbors)

            # ========== Recall计算逻辑 ==========
            # 检查返回的top-k中是否有正样本
            for k in range(len(indices[0])):
                if indices[0][k] in true_neighbors_global:
                    # 从位置k开始到最后都应该+1（因为top-k包含了这个正样本）
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

    # 计算平均recall
    ave_recall = (recall / float(num_evaluated)) * 100.0
    ave_one_percent_recall = np.mean(one_percent_recall_list) * 100.0

    stats = {
        'ave_one_percent_recall': ave_one_percent_recall,
        'ave_recall': ave_recall,
        'num_evaluated': num_evaluated
    }

    return stats


def get_latent_vectors(model, point_cloud_set, device, params: TrainingParams):
    """
    计算点云集合的embeddings
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
        pc = torch.tensor(pc)

        embedding = compute_embedding(model, pc, device, params)

        if embeddings is None:
            embeddings = np.zeros((len(point_cloud_set), embedding.shape[1]), dtype=embedding.dtype)
        embeddings[i] = embedding

        # 定期清理GPU缓存
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


def print_eval_stats(stats):
    """打印评估统计信息"""
    if stats is None:
        print("❌ 评估失败")
        return

    print('\nDataset: chilean')
    print(f"Avg. top 1% recall: {stats['ave_one_percent_recall']:.2f}   Avg. recall @N:")
    print(stats['ave_recall'])


def chilean_write_eval_stats(file_name, prefix, stats):
    """将评估结果写入文件"""
    if stats is None:
        return

    s = prefix
    ave_1p_recall = stats['ave_one_percent_recall']
    ave_recall_1 = stats['ave_recall'][0]
    s += f", {ave_1p_recall:.2f}, {ave_recall_1:.2f}\n"

    with open(file_name, "a") as f:
        f.write(s)


if __name__ == "__main__":
    # 直接设置参数 (用于单独运行此脚本调试)
    class Args:
        def __init__(self):
            self.config = '../config/config_chilean_bev.txt'
            self.model_config = '../models/minkloc_bev.txt'
            self.weights = '/home/wzj/pan1/MinkLocHABev/weights/model_MinkLocBEV_20251221_1045_final.pth'
            # self.weights = None
            self.debug = False


    args = Args()

    print('=' * 60)
    print('Chilean数据集评估')
    print('=' * 60)
    print(f'配置文件: {args.config}')
    print(f'模型配置: {args.model_config}')

    if args.weights is None:
        print('权重文件: 随机初始化权重（测试模式）')
    else:
        print(f'权重文件: {args.weights}')

    print(f'Debug模式: {args.debug}')
    print('')

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
        if not os.path.exists(args.weights):
            print(f'❌ 错误: 找不到权重文件: {args.weights}')
            exit(1)
        print(f'加载权重: {args.weights}')
        model.load_state_dict(torch.load(args.weights, map_location=device))

    model.to(device)

    # 运行评估
    stats = evaluate_chilean(model, device, params, log=False, show_progress=True)

    # 打印结果（格式与Oxford一致）
    print_eval_stats(stats)

    # 保存结果到文件
    if stats is not None:
        model_params_name = os.path.split(params.model_params.model_params_path)[1]
        config_name = os.path.split(params.params_path)[1]

        if args.weights is not None:
            model_name = os.path.split(args.weights)[1]
            model_name = os.path.splitext(model_name)[0]
        else:
            model_name = "random_weights"

        prefix = f"{model_params_name}, {config_name}, {model_name}"
        chilean_write_eval_stats("chilean_experiment_results.txt", prefix, stats)
        print(f"\n结果已保存到: chilean_experiment_results.txt")