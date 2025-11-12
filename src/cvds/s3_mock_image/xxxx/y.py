import numpy as np
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------- 1. 特征维度与权重配置 ----------------------
feature_config = [
    ('keypoints', 'array', 212, 0.3),
    ('skin_color', 'array', 3, 0.1),
    ('skin_var', 'array', 3, 0.05),
    ('lbp', 'array', 50, 0.1),
    ('contour', 'array', 5, 0.05),
    ('glasses', 'binary', 1, 0.05),
    ('mask', 'binary', 1, 0.05),
    ('hair', 'array', 8, 0.05),
    ('gender', 'binary', 1, 0.05),
    ('age', 'binary', 1, 0.05),
    ('expression', 'binary', 1, 0.05),
    ('pose', 'array', 3, 0.05),
    ('quality', 'array', 2, 0.025),
    ('occlusion', 'array', 3, 0.025)
]
TOTAL_DIM = sum(config[2] for config in feature_config)


# ---------------------- 2. 辅助函数 ----------------------

def _get_feature_indices(feature_name):
    """根据特征名获取其在向量中的起始和结束索引（[start, end)）"""
    start = 0
    for name, _, dim, _ in feature_config:
        if name == feature_name:
            return start, start + dim
        start += dim
    raise ValueError(f"Feature '{feature_name}' not found in config.")


def calculate_weighted_similarity(vec1, vec2):
    """计算两个向量的加权相似度"""
    total_sim = 0.0
    start = 0
    for _, dtype, dim, weight in feature_config:
        feat1, feat2 = vec1[start:start + dim], vec2[start:start + dim]
        start += dim

        if dtype == 'binary':
            sim = 1.0 if np.array_equal(feat1, feat2) else 0.0
        else:  # array
            sim = cosine_similarity([feat1], [feat2])[0][0] if dim > 0 else 0
        total_sim += sim * weight
    return total_sim


def generate_base_vector(seed=None):
    """生成一个随机的基准向量，代表一个独特的人（兼容旧版NumPy）"""
    if seed is not None:
        np.random.seed(seed)

    vec = np.zeros(TOTAL_DIM)
    current_idx = 0

    for name, dtype, dim, _ in feature_config:
        if dtype == 'binary':
            # 使用 randint 兼容旧版本
            part = np.random.randint(0, 2, size=dim)
        else:  # array
            if name == 'pose':
                # 姿态角度在 [-30, 30] 度
                part = np.random.uniform(-30, 30, size=dim)
            elif name == 'skin_var':
                # 肤色方差在 [0, 0.1]
                part = np.random.uniform(0, 0.1, size=dim)
            else:
                part = np.random.uniform(0, 1, size=dim)

        vec[current_idx:current_idx + dim] = part
        current_idx += dim

    return vec


# ---------------------- 3. 批量生成函数 ----------------------

def generate_batch(
        target_vector,
        num_same=50, p_same_high_sim=0.8, same_noise_scale_high=0.005, same_noise_scale_low=0.03,
        num_diff=100, p_diff_high_sim=0.1,
        num_confuser=20, confuser_noise_scale=0.02
):
    """
    批量生成三种类型的向量。
    """
    vectors = []
    labels = []
    similarities = []

    pose_start, pose_end = _get_feature_indices('pose')

    # 1. 生成“同一人”向量
    for _ in range(num_same):
        if np.random.random() < p_same_high_sim:
            noise_scale = same_noise_scale_high
            label = 'same_high'
        else:
            noise_scale = same_noise_scale_low
            label = 'same_low'

        noise = np.random.normal(0, noise_scale, size=TOTAL_DIM)
        vec = np.clip(target_vector + noise, 0, 1)

        # 单独处理姿态角度，确保其范围正确
        pose_noise = np.random.normal(0, noise_scale * 5, size=3)
        vec[pose_start:pose_end] = np.clip(target_vector[pose_start:pose_end] + pose_noise, -30, 30)

        vectors.append(vec)
        labels.append(label)
        similarities.append(calculate_weighted_similarity(target_vector, vec))

    # 2. 生成“不同人”向量
    for _ in range(num_diff):
        if np.random.random() < p_diff_high_sim:
            # 生成高相似度的不同人（混淆人）
            noise = np.random.normal(0, confuser_noise_scale * 1.2, size=TOTAL_DIM)
            vec = np.clip(target_vector + noise, 0, 1)
            pose_noise = np.random.normal(0, confuser_noise_scale * 6, size=3)
            vec[pose_start:pose_end] = np.clip(target_vector[pose_start:pose_end] + pose_noise, -30, 30)
            labels.append('diff_high')
        else:
            # 生成低相似度的不同人
            vec = generate_base_vector()
            labels.append('diff_low')

        vectors.append(vec)
        similarities.append(calculate_weighted_similarity(target_vector, vec))

    # 3. 生成“混淆人”向量
    for _ in range(num_confuser):
        noise = np.random.normal(0, confuser_noise_scale, size=TOTAL_DIM)
        vec = np.clip(target_vector + noise, 0, 1)

        pose_noise = np.random.normal(0, confuser_noise_scale * 5, size=3)
        vec[pose_start:pose_end] = np.clip(target_vector[pose_start:pose_end] + pose_noise, -30, 30)

        # 随机改变一个非关键二进制属性，增加混淆性
        if np.random.random() < 0.5:
            expr_start, expr_end = _get_feature_indices('expression')
            # 使用 randint 兼容旧版本
            vec[expr_start:expr_end] = np.random.randint(0, 2, size=expr_end - expr_start)

        vectors.append(vec)
        similarities.append(calculate_weighted_similarity(target_vector, vec))
        labels.append('confuser')

    return vectors, labels, similarities


# ---------------------- 4. 使用示例 ----------------------

if __name__ == '__main__':
    # 设置随机种子以保证结果可复现
    np.random.seed(42)

    # 步骤 1: 生成一个目标向量
    print("步骤 1: 生成目标向量...")
    target_user_vector = generate_base_vector(seed=123)
    print("目标向量生成完毕。")

    # 步骤 2: 调用批量生成函数
    # --- 核心参数调整 ---
    # same_noise_scale_low=0.08: 增大噪声，使同一人的低相似度样本相似度降低
    # confuser_noise_scale=0.04: 调整噪声，使混淆人的相似度保持在较高水平
    print("\n步骤 2: 批量生成向量...")
    all_vectors, all_labels, all_similarities = generate_batch(
        target_vector=target_user_vector,
        num_same=30, p_same_high_sim=0.7,
        same_noise_scale_high=0.005,
        same_noise_scale_low=0.08,  # <--- 调整此处
        num_diff=50, p_diff_high_sim=0.05,
        num_confuser=10,
        confuser_noise_scale=0.04  # <--- 调整此处
    )
    print(f"批量生成完毕，共 {len(all_vectors)} 个向量。")

    # 步骤 3: 分析和验证结果
    print("\n步骤 3: 结果分析...")

    # 创建一个字典来按标签分组相似度
    sim_by_label = {}
    for label, sim in zip(all_labels, all_similarities):
        sim_by_label.setdefault(label, []).append(sim)

    # 打印每种类型的统计信息
    print("各类型向量与目标向量的相似度统计：")
    for label, sims in sim_by_label.items():
        sims = np.array(sims)
        print(f"  - {label:<10}: 数量={len(sims):<3}  均值={sims.mean():.4f}  标准差={sims.std():.4f}  最小值={sims.min():.4f}  最大值={sims.max():.4f}")

    # 可视化结果（需要 matplotlib）
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 7))

        labels_for_plot = ['same_high', 'same_low', 'confuser', 'diff_high', 'diff_low']
        data_for_plot = [sim_by_label[label] for label in labels_for_plot if label in sim_by_label]
        labels_for_plot = [label for label in labels_for_plot if label in sim_by_label]

        plt.boxplot(data_for_plot, labels=labels_for_plot, patch_artist=True)
        plt.title('不同类型向量与目标向量的相似度分布（调整后）')
        plt.ylabel('加权余弦相似度')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(0, 1.05)
        plt.savefig("similarity_boxplot_adjusted.png")
    except ImportError:
        print("\n提示：未安装 matplotlib，跳过可视化。")