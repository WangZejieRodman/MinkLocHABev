import numpy as np
import os
from datetime import datetime

from datasets.base_datasets import PointCloudLoader


class PNVPointCloudLoader(PointCloudLoader):
    def set_properties(self):
        # 点云已经预处理过，移除了地面
        self.remove_zero_points = False
        self.remove_ground_plane = False
        self.ground_plane_level = None

        # 添加：点数统计
        self.point_count_stats = []

        # 日志文件路径
        self.log_file = "/home/wzj/pan1/MinkLoc3dv2_Chilean_原始点云/training/pnv_raw.log"

        # 确保日志目录存在
        log_dir = os.path.dirname(self.log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 初始化日志文件（写入开始时间）
        with open(self.log_file, 'a') as f:
            f.write(f"\n{'=' * 60}\n")
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始加载点云\n")
            f.write(f"{'=' * 60}\n")

    def read_pc(self, file_pathname: str) -> np.ndarray:
        # 读取点云，不进行预处理
        # 返回 Nx3 的ndarray
        file_path = os.path.join(file_pathname)
        pc = np.fromfile(file_path, dtype=np.float64)
        pc = np.float32(pc)
        # 坐标在每个维度都在 -1..1 范围内
        pc = np.reshape(pc, (pc.shape[0] // 3, 3))

        MAX_POINTS = 20000

        # 如果点云过密，下采样到固定点数
        if pc.shape[0] > MAX_POINTS:
            indices = np.random.choice(pc.shape[0], MAX_POINTS, replace=False)
            pc = pc[indices]

        # 如果点云过稀（<1000点），记录警告但保留
        elif pc.shape[0] < 1000:
            # 可选：在日志里记录这些异常样本
            pass

        # 监控：记录点云点数
        num_points = pc.shape[0]
        self.point_count_stats.append(num_points)



        # 每1000个点云写一次统计信息到日志
        if len(self.point_count_stats) % 1000 == 0:
            stats = np.array(self.point_count_stats)
            with open(self.log_file, 'a') as f:
                f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 点云统计 - 已加载 {len(stats)} 个点云:\n")
                f.write(f"  点数范围: {stats.min()} - {stats.max()}\n")
                f.write(f"  平均点数: {stats.mean():.1f}\n")
                f.write(f"  中位数点数: {np.median(stats):.1f}\n")
                f.write(f"  稀疏(<1000点): {(stats < 1000).sum()} 个\n")
                f.write(f"  密集(>100000点): {(stats > 100000).sum()} 个\n")

        return pc