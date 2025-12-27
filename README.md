# MinkLocHABev - 基于BEV表达的地下矿井点云场景识别

## 项目概述

MinkLocHABev 是一个专门针对地下矿井环境的激光雷达点云场景识别与定位系统。该项目改编自 MinkLoc3D，核心创新在于采用 **BEV (Bird's Eye View, 鸟瞰图)** 表达方式替代传统的3D体素表达，更适合捕捉地下巷道环境中的几何连续性、走向和曲折等特征。

### 核心设计理念

**从3D到BEV的范式转换：**
- **MinkLoc3D**: 保留 (X, Y, Z) 三维坐标，将点云视为3D空间中的稀疏体素，捕捉物体的三维几何结构
- **MinkLocHABev**: 将 Z 轴高度信息转化为特征通道，生成"多通道的2D稀疏图像"，更适合表达巷道壁的刚性连续性

这种设计特别适合地下矿井场景，因为：
- 巷道具有明显的水平方向连续性
- 侧壁结构是主要的识别特征
- 顶底板往往是噪声源
- 走向和曲折是关键的空间信息

---

## 数据集

### Chilean Underground Mine Dataset

项目使用智利地下矿井数据集，这是一个大规模的矿井激光雷达扫描数据集。

**数据组织结构：**
```
Chilean_Underground_Mine_Dataset_Many_Times/
└── chilean_NoRot_NoScale/
    ├── 100/  (Session 100)
    │   ├── pointcloud_20m_10overlap/
    │   │   ├── 0.bin
    │   │   ├── 1.bin
    │   │   └── ...
    │   └── pointcloud_locations_20m_10overlap.csv
    ├── 101/  (Session 101)
    ├── ...
    └── 209/  (Session 209)
```

**数据划分策略（基于时间/Session）：**

| 用途 | Session范围 | 数量 | 说明 |
|------|------------|------|------|
| **训练集** | 100-159 | 60 sessions | 用于模型训练 |
| **测试集** | 160-179 | 20 sessions | 用于测试评估 |
| **数据库集** | 180-194 | 15 sessions | 历史点云作为参考地图 |
| **查询集** | 195-209 | 15 sessions | 当前观测用于定位 |

**数据特点：**
- **点云格式**: `.bin` 文件，双精度浮点数 (float64)，存储为 (X, Y, Z) 坐标
- **坐标归一化**: 每个维度都在 -1 到 1 范围内
- **采样策略**: 20米间隔采样，10%重叠
- **点云密度**: 自动下采样到最多20,000点（避免过密点云）
- **正样本阈值**: 7米内为正样本
- **负样本阈值**: 35米外为负样本

---

## 核心技术架构

### 1. BEV量化器 (BEVQuantizer)

**功能**: 将3D点云转换为BEV表达

**工作流程：**
```python
输入: (N, 3) 点云坐标 [X, Y, Z]
  ↓
过滤范围: X∈[-10,10], Y∈[-10,10], Z∈[-4,8]
  ↓
网格划分: 256×256×32 (XY平面×Z轴)
  ↓
XY平面去重: 保留唯一的(X,Y)位置
  ↓
Z轴编码: 将32个高度层编码为32维特征向量
  ↓
输出: (M, 2)坐标 + (M, 32)特征
```

**关键参数：**
- `coords_range`: `[-10, -10, -4, 10, 10, 8]` (米)
- `div_n`: `[256, 256, 32]` (网格分辨率)
- **体素大小**: ~7.8cm × 7.8cm (水平) × 37.5cm (垂直)

**BEV表达的优势：**
- 降维：从3D稀疏张量降至2D稀疏张量
- 高度信息保留：通过32维occupancy向量保留Z轴信息
- 计算效率：2D卷积比3D卷积显著更快

---

### 2. MinkBEVBackbone - 创新型骨干网络

网络集成了三大创新模块，专门针对地下矿井巷道环境设计：

#### **创新点1: 动态全局层注意力 (Global Layer Attention)**

```python
class GlobalLayerAttention(nn.Module):
    """
    类似SE-Block，但作用于BEV的Z-layers (Channels)
    学习哪些高度层重要（如侧壁），哪些是噪声（如顶底板）
    """
```

**机制：**
1. 全局平均池化 → (Batch, 32)
2. MLP生成权重 → (Batch, 32)
3. 广播乘法 → 加权特征

**效果**: 自适应抑制顶底板噪声层，增强侧壁特征层

---

#### **创新点2: 垂直上下文模块 (Vertical Context Module)**

```python
class VerticalContextModule(nn.Module):
    """
    通过1D卷积在Channel (Z-axis)维度上进行上下文融合
    让Layer i能够感知Layer i-1和i+1的信息
    """
```

**机制：**
- 将 (N, 32) 特征视为 N 个长度为32的序列
- 应用 1D Conv (kernel_size=3) 进行垂直融合
- 恢复高度维度之间的相关性

**动机**: BEV量化过程丢失了Z轴的连续性，该模块用于恢复

---

#### **创新点3: 稀疏局部注意力 (Sparse Local Attention)**

```python
class SparseLocalAttention(nn.Module):
    """
    通过"Gated Local Context"机制模拟局部注意力
    如果局部几何特征与中心点特征匹配则激活，否则抑制
    """
```

**机制：**
- Query: 1×1卷积提取中心点特征
- Key/Context: 3×3卷积提取邻域特征
- Attention: Query ⊙ Key (哈达玛积)
- Value: 加权后的特征 + 残差连接

**优势**: 比全局Self-Attention更高效，利用稀疏性

---

#### **网络整体架构**

```
输入: (Batch, 32, H, W) BEV特征
  ↓
[Stage 1] 输入增强
  ├─ Vertical Context Module  (恢复Z轴相关性)
  └─ Global Layer Attention    (抑制噪声层)
  ↓
[Stage 2] 特征提取
  ├─ Block1: 32→64, /2 (downsample)
  ├─ Block2: 64→128, /4
  └─ Block3: 128→256, /8
  ↓
[Stage 3] 高级语义
  └─ Sparse Local Attention    (自适应几何建模)
  ↓
[Stage 4] 输出对齐
  └─ Conv: 256→256
  ↓
输出: (Batch, 256, H/8, W/8)
```

**Bottleneck模块**: 标准ResNet Bottleneck（1×1 → 3×3 → 1×1）

---

### 3. 全局描述子生成

**Pooling策略: GeM (Generalized Mean Pooling)**

```python
class GeM(nn.Module):
    """
    通过可学习的p参数平衡MaxPooling和AvgPooling
    """
    def forward(self, x):
        return (AvgPool(x^p))^(1/p)
```

**输出维度**: 256维全局描述子

---

## 训练策略

### 损失函数: TruncatedSmoothAP

一种针对大批量检索优化的损失函数：

**核心思想：**
- 只考虑每个query最近的K个正样本（K=2）
- 使用Sigmoid平滑化排序函数
- 优化Average Precision (AP)

**参数：**
- `tau1=0.01`: Sigmoid温度控制
- `positives_per_query=2`: 每个query考虑的正样本数
- `similarity='euclidean'`: 使用欧氏距离

---

### 优化器配置

```ini
[TRAIN]
batch_size=128           # 全局batch大小
batch_split_size=16      # 多阶段反向传播的mini-batch大小
epochs=100               # 训练轮数
lr=1e-3                  # 初始学习率
optimizer=Adam           # 优化器
scheduler=MultiStepLR    # 学习率调度
scheduler_milestones=30,60,80  # 在这些epoch降低学习率
weight_decay=1e-4        # L2正则化
```

**学习率衰减策略：**
- Epoch 0-29: lr = 1e-3
- Epoch 30-59: lr = 1e-4 (×0.1)
- Epoch 60-79: lr = 1e-5 (×0.1)
- Epoch 80-100: lr = 1e-6 (×0.1)

---

### 数据增强

**训练时增强 (TrainSetTransform):**
```python
aug_mode=1:
  - RandomRotation: 0-360度随机旋转（绕Z轴）
  - RandomFlip: X/Y轴随机翻转 (概率25%)
```

**PointNetVLAD风格增强 (TrainTransform):**
```python
  - JitterPoints: 高斯抖动 (σ=0.001)
  - RemoveRandomPoints: 随机移除0-10%的点
  - RandomTranslation: 随机平移 (max_delta=0.01)
  - RemoveRandomBlock: 随机遮挡块 (p=0.4)
```

---

## 评估协议

### 标准评估

**数据库 vs 查询:**
- Database: Session 180-194 (历史地图)
- Query: Session 195-209 (当前观测)

**评估指标:**
- **Top 1% Recall**: 在top 1%检索结果中找到正样本的查询比例
- **Recall@K**: 在top-K检索结果中找到正样本的查询比例 (K=1,5,10,25)

**检索流程:**
1. 计算所有database点云的256维描述子
2. 计算所有query点云的256维描述子
3. 使用KDTree进行最近邻搜索（欧氏距离）
4. 统计Recall指标

---

### 旋转不变性评估

专门测试模型对旋转的鲁棒性（这对矿井场景很重要，因为采集设备可能有不同的朝向）

**测试角度:**
```python
rotation_angles = [0, 5, 10, 15, 30, 45, 60, 90, 135, 180]  # 度
```

**评估策略:**
- Database保持原始朝向（0度）
- Query旋转指定角度
- 对比不同角度下的性能下降

**评估标准:**
- 平均相对下降 < 5%: 优秀 ✓✓✓
- 平均相对下降 < 10%: 良好 ✓✓
- 平均相对下降 < 20%: 一般 ✓
- 平均相对下降 ≥ 20%: 较差 ✗

---

## 使用方法

### 环境配置

**依赖库:**
```bash
torch >= 1.10
MinkowskiEngine >= 0.5
scikit-learn
pandas
tqdm
```

**安装MinkowskiEngine:**
```bash
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine
```

---

### 数据准备

1. **生成训练/测试查询字典:**
```bash
cd datasets/chilean
python generate_training_tuples_chilean.py
```
生成文件:
- `training_queries_chilean.pickle` (Sessions 100-159)
- `test_queries_chilean.pickle` (Sessions 160-179)

2. **生成评估数据集:**
```bash
python generate_test_sets_chilean.py
```
生成文件:
- `chilean_evaluation_database_180_194.pickle`
- `chilean_evaluation_query_195_209.pickle`

---

### 训练模型

**方式1: PyCharm调试模式**
```bash
python training/train_chilean_bev.py
```

**方式2: 命令行模式**
```bash
python training/trainer.py \
  --config config/config_chilean_bev.txt \
  --model_config models/minkloc_bev.txt
```

**训练输出:**
- 模型权重: `weights/model_MinkLocBEV_YYYYMMDD_HHMM_*.pth`
- 日志文件: `training/trainer.log`
- 旋转评估结果: `rotation_results_*.pkl`

---

### 评估模型

**标准评估:**
```bash
cd eval
python evaluate_chilean.py
```

**旋转不变性评估:**
```bash
python evaluate_chilean_rotation.py
```

**分析旋转结果:**
```bash
python analyze_rotation_results.py
```

---

## 项目结构

```
MinkLocHABev/
├── config/
│   └── config_chilean_bev.txt          # 训练配置
├── datasets/
│   ├── chilean/
│   │   ├── generate_training_tuples_chilean.py    # 生成训练数据
│   │   └── generate_test_sets_chilean.py          # 生成评估数据
│   ├── augmentation.py                 # 数据增强
│   ├── quantization.py                 # BEV量化器
│   └── rotation_utils.py               # 旋转工具
├── models/
│   ├── minkbev.py                      # BEV骨干网络
│   ├── minkloc.py                      # 主模型封装
│   ├── minkloc_bev.txt                 # 模型配置
│   ├── layers/
│   │   ├── bev_enhancers.py           # 三大创新模块
│   │   ├── pooling.py                 # GeM等池化
│   │   └── ...
│   └── losses/
│       └── truncated_smoothap.py      # TruncatedSmoothAP损失
├── training/
│   ├── train_chilean_bev.py           # Chilean训练脚本
│   └── trainer.py                      # 通用训练器
├── eval/
│   ├── evaluate_chilean.py            # 标准评估
│   ├── evaluate_chilean_rotation.py   # 旋转评估
│   └── analyze_rotation_results.py    # 结果分析
└── misc/
    └── utils.py                        # 工具函数
```

---

## 关键配置文件

### config_chilean_bev.txt
```ini
[DEFAULT]
dataset_folder=/path/to/Chilean_Underground_Mine_Dataset_Many_Times/

[TRAIN]
batch_size=128
batch_split_size=16
epochs=100
lr=1e-3
scheduler=MultiStepLR
scheduler_milestones=30,60,80
loss=TruncatedSmoothAP
similarity=euclidean
aug_mode=1          # 旋转+翻转增强
set_aug_mode=1      # 全局旋转增强
```

### minkloc_bev.txt
```ini
[MODEL]
model=MinkLocBEV
feature_size=256
output_dim=256
pooling=GeM
coordinates=bev
coords_range=-10., -10, -4, 10, 10, 8
div_n=256, 256, 32
in_channels=32
```

---

## 性能指标监控

训练过程中会记录以下指标：

**损失相关:**
- `loss`: TruncatedSmoothAP损失值
- `ap`: Average Precision
- `avg_embedding_norm`: 描述子的平均范数

**检索相关:**
- `positives_per_query`: 每个query的平均正样本数
- `best_positive_ranking`: 最佳正样本的平均排名
- `recall@1`: Recall at 1

**资源监控:**
- `avg_voxels`: 平均体素数（稀疏性指标）
- `max_voxels`: 峰值体素数
- `gpu_memory_mb`: GPU显存占用（MB）

---

## BEV vs 3D: 设计对比

| 维度 | MinkLoc3D | MinkLocHABev |
|------|-----------|--------------|
| **输入表达** | 3D稀疏体素 (X,Y,Z) | 2D BEV + 32通道 (X,Y,Features) |
| **卷积维度** | 3D卷积 | 2D卷积 |
| **计算复杂度** | O(N³) | O(N²) |
| **特征通道** | 1 (occupancy) | 32 (Z-axis encoding) |
| **适用场景** | 通用3D物体识别 | 结构化环境（巷道、道路） |
| **关键优势** | 完整3D几何 | 水平连续性、计算效率 |
| **信息保留** | 精确Z坐标 | Z-occupancy分布 |

**为什么BEV更适合矿井？**
1. **巷道的2D本质**: 巷道主要是水平延伸的，Z轴变化相对较小
2. **侧壁重要性**: 识别依赖侧壁纹理，而非精确的3D形状
3. **噪声抑制**: 顶底板噪声可以通过层注意力机制过滤
4. **计算效率**: 2D卷积比3D卷积快数倍，支持更大的batch size

---

## 创新总结

MinkLocHABev的核心贡献：

1. **范式创新**: 首次将BEV表达应用于地下矿井点云场景识别
2. **模块创新**: 
   - 垂直上下文模块恢复Z轴相关性
   - 全局层注意力抑制噪声层
   - 稀疏局部注意力自适应建模几何
3. **任务适配**: 针对矿井场景的数据增强和评估协议
4. **工程优化**: 多阶段反向传播支持大batch训练

---

## 引用

如果本项目对您的研究有帮助，请考虑引用：

```bibtex
@misc{minklochabev2024,
  title={MinkLocHABev: BEV-based Point Cloud Place Recognition for Underground Mines},
  author={Your Name},
  year={2024}
}
```

---

## 致谢

本项目改编自 [MinkLoc3D](https://github.com/jac99/MinkLoc3D)，感谢原作者的开源贡献。

Chilean Underground Mine Dataset 由相关研究团队提供，在此表示感谢。

---

## 许可证

本项目遵循 MIT License。
