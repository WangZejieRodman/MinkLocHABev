# Chilean BEV模型训练脚本 (PyCharm调试专用)
# 集成旋转不变性评估
# Warsaw University of Technology

import os
import sys
import torch

# =========================================================
# 路径设置技巧:
# 确保在 PyCharm 的 Run/Debug Configurations 中:
# "Working directory" 设置为你的项目根目录 (minkloc3dv2_org 文件夹)
# =========================================================

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from training.trainer import do_train
from misc.utils import TrainingParams
from models.model_factory import model_factory

# =========================================================
# 修改点 1: 导入旋转评估模块
# =========================================================
from eval.evaluate_chilean_rotation import evaluate_chilean_with_rotation, save_rotation_results


def do_train_chilean(params: TrainingParams):
    """
    Chilean数据集训练主函数
    训练完成后自动运行旋转鲁棒性评估
    """
    print(f"开始训练: {params.model_params.model}")

    # 1. 执行训练
    model, model_pathname = do_train(params, skip_final_eval=True)

    # 2. 训练完成后在Chilean数据集上评估
    print('\n' + '=' * 60)
    print('训练结束，开始在Chilean数据集上进行【旋转鲁棒性评估】...')
    print('=' * 60)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # 加载最终模型权重
    final_model_path = model_pathname + '_final.pth'
    if os.path.exists(final_model_path):
        print(f'加载最终模型权重: {final_model_path}')
        model.load_state_dict(torch.load(final_model_path, map_location=device))
    else:
        print('警告: 未找到最终模型权重文件 (_final.pth)，使用当前内存中的模型状态进行评估')

    model.to(device)

    # =========================================================
    # 修改点 2: 运行旋转评估
    # =========================================================
    # 定义想要测试的旋转角度
    rotation_angles = [0, 5, 10, 15, 30, 45, 60, 90, 135, 180]
    print(f"测试角度列表: {rotation_angles}")

    # 调用旋转评估函数
    all_stats = evaluate_chilean_with_rotation(
        model, device, params, rotation_angles,
        log=False, show_progress=True
    )

    # 4. 保存评估结果
    if all_stats is not None:
        # 获取文件名用于日志记录
        model_name = os.path.splitext(os.path.split(final_model_path)[1])[0]

        # 结果将保存在项目根目录下，文件名为 "rotation_results_<模型名>.pkl"
        output_file = f"rotation_results_{model_name}.pkl"

        save_rotation_results(all_stats, output_file)

        # 简要打印 0度和180度的结果供参考
        if 0 in all_stats:
            print(f"\n[摘要] 0度 (原始) Recall@1: {all_stats[0]['ave_recall'][0]:.2f}%")
        if 180 in all_stats:
            print(f"[摘要] 180度 Recall@1: {all_stats[180]['ave_recall'][0]:.2f}%")


if __name__ == '__main__':
    # =========================================================
    # PyCharm 调试配置区域
    # =========================================================
    class Args:
        def __init__(self):
            # 1. 训练参数配置文件
            self.config = '../config/config_chilean_bev.txt'

            # 2. 模型结构配置文件
            self.model_config = '../models/minkloc_bev.txt'

            # 3. 调试模式
            self.debug = False


    args = Args()

    # 路径检查
    if not os.path.exists(args.config):
        print(f"错误: 找不到配置文件: {os.path.abspath(args.config)}")
        print("请检查 PyCharm 的 'Working directory' 是否设置为项目根目录！")
        exit(1)

    if not os.path.exists(args.model_config):
        print(f"错误: 找不到模型配置文件: {os.path.abspath(args.model_config)}")
        exit(1)

    print('=' * 60)
    print('启动 Chilean BEV 训练 + 旋转评估 (PyCharm Mode)')
    print('=' * 60)
    print(f'Training Config : {args.config}')
    print(f'Model Config    : {args.model_config}')
    print(f'Debug Mode      : {args.debug}')
    print('')

    # 初始化参数
    params = TrainingParams(args.config, args.model_config, debug=args.debug)
    params.print()

    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    # 开始流程
    do_train_chilean(params)