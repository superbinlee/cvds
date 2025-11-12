import numpy as np
import matplotlib

# 设置matplotlib使用非交互式后端，适用于服务器环境或无图形界面情况
matplotlib.use('Agg')

# 配置matplotlib字体以支持中文显示
plt_font_families = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 'Microsoft YaHei']
plt_font_set = False
for font in plt_font_families:
    try:
        matplotlib.rcParams['font.sans-serif'] = [font]
        matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        plt_font_set = True
        break
    except:
        continue

# 如果没有找到合适的中文字体，至少尝试设置默认字体
if not plt_font_set:
    matplotlib.rcParams['axes.unicode_minus'] = False

from sklearn.metrics.pairwise import cosine_similarity

# ---------------------- 1. 特征维度与权重配置 ----------------------
# 定义了人物特征向量的配置列表，每项包含：
# (特征名称, 数据类型, 维度大小, 权重)
feature_config = [
    ('keypoints', 'array', 212, 0.3),  # 关键点特征，212维，权重0.3
    ('skin_color', 'array', 3, 0.1),  # 肤色特征，3维，权重0.1
    ('skin_var', 'array', 3, 0.05),  # 肤色方差，3维，权重0.05
    ('lbp', 'array', 50, 0.1),  # LBP纹理特征，50维，权重0.1
    ('contour', 'array', 5, 0.05),  # 轮廓特征，5维，权重0.05
    ('glasses', 'binary', 1, 0.05),  # 是否戴眼镜，二值特征，权重0.05
    ('mask', 'binary', 1, 0.05),  # 是否戴口罩，二值特征，权重0.05
    ('hair', 'array', 8, 0.05),  # 头发特征，8维，权重0.05
    ('gender', 'binary', 1, 0.05),  # 性别，二值特征，权重0.05
    ('age', 'binary', 1, 0.05),  # 年龄，二值特征，权重0.05
    ('expression', 'binary', 1, 0.05),  # 表情，二值特征，权重0.05
    ('pose', 'array', 3, 0.05),  # 姿态，3维，权重0.05
    ('quality', 'array', 2, 0.025),  # 图像质量，2维，权重0.025
    ('occlusion', 'array', 3, 0.025)  # 遮挡程度，3维，权重0.025
]

# 计算总特征维度
TOTAL_DIM = sum(config[2] for config in feature_config)


# ---------------------- 2. 辅助函数 ----------------------

def _get_feature_indices(feature_name):
    """
    根据特征名获取其在向量中的起始和结束索引（[start, end)）

    参数：
        feature_name (str): 要查找的特征名称

    返回：
        tuple: (起始索引, 结束索引)，遵循左闭右开区间

    异常：
        ValueError: 当找不到指定的特征名时抛出
    """
    start = 0
    for name, _, dim, _ in feature_config:
        if name == feature_name:
            return start, start + dim
        start += dim
    raise ValueError(f"Feature '{feature_name}' not found in config.")


def calculate_weighted_similarity(vec1, vec2):
    """
    计算两个特征向量的加权相似度

    参数：
        vec1 (np.ndarray): 第一个特征向量
        vec2 (np.ndarray): 第二个特征向量

    返回：
        float: 加权相似度值，范围在[0,1]之间
    """
    total_sim = 0.0  # 总相似度初始化为0
    start = 0  # 当前特征在向量中的起始位置

    # 遍历每个特征配置
    for _, dtype, dim, weight in feature_config:
        # 提取当前特征的子向量
        feat1, feat2 = vec1[start:start + dim], vec2[start:start + dim]
        start += dim  # 更新起始位置到下一个特征

        # 根据特征类型选择不同的相似度计算方法
        if dtype == 'binary':
            # 对于二值特征，如果完全相同则相似度为1，否则为0
            sim = 1.0 if np.array_equal(feat1, feat2) else 0.0
        else:  # array类型特征
            # 对于数组类型特征，使用余弦相似度
            sim = cosine_similarity([feat1], [feat2])[0][0] if dim > 0 else 0

        # 累加加权相似度
        total_sim += sim * weight

    return total_sim


def generate_base_vector(seed=None):
    """
    生成一个随机的基准向量，代表一个独特的人
    注：代码兼容旧版NumPy

    参数：
        seed (int, optional): 随机种子，用于结果复现

    返回：
        np.ndarray: 生成的基准特征向量
    """
    # 设置随机种子以保证结果可复现
    if seed is not None:
        np.random.seed(seed)

    # 创建总维度大小的零向量
    vec = np.zeros(TOTAL_DIM)
    current_idx = 0

    # 遍历每个特征配置，为每个特征生成随机值
    for name, dtype, dim, _ in feature_config:
        if dtype == 'binary':
            # 使用 randint 生成二值特征（0或1），兼容旧版本
            part = np.random.randint(0, 2, size=dim)
        else:  # array类型特征
            if name == 'pose':
                # 姿态角度在 [-30, 30] 度范围内随机生成
                part = np.random.uniform(-30, 30, size=dim)
            elif name == 'skin_var':
                # 肤色方差在 [0, 0.1] 范围内随机生成
                part = np.random.uniform(0, 0.1, size=dim)
            else:
                # 其他数组特征在 [0, 1] 范围内随机生成
                part = np.random.uniform(0, 1, size=dim)

        # 将生成的特征部分赋值到向量的相应位置
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
    批量生成三种类型的特征向量：同一人、不同人、混淆人

    参数：
        target_vector (np.ndarray): 目标向量，用于生成其他相似或不同的向量
        num_same (int): "同一人"向量的生成数量
        p_same_high_sim (float): "同一人"中高相似度样本的概率
        same_noise_scale_high (float): 高相似度样本的噪声标准差
        same_noise_scale_low (float): 低相似度样本的噪声标准差
        num_diff (int): "不同人"向量的生成数量
        p_diff_high_sim (float): "不同人"中高相似度样本的概率
        num_confuser (int): "混淆人"向量的生成数量
        confuser_noise_scale (float): "混淆人"向量的噪声标准差

    返回：
        tuple: (vectors, labels, similarities)
            vectors: 生成的所有特征向量列表
            labels: 向量对应的标签列表
            similarities: 向量与目标向量的相似度列表
    """
    vectors = []  # 存储生成的所有向量
    labels = []  # 存储对应的标签
    similarities = []  # 存储相似度值

    # 获取姿态特征在向量中的位置
    pose_start, pose_end = _get_feature_indices('pose')

    # 1. 生成"同一人"向量
    for _ in range(num_same):
        # 根据概率决定是高相似度还是低相似度
        if np.random.random() < p_same_high_sim:
            noise_scale = same_noise_scale_high
            label = 'same_high'
        else:
            noise_scale = same_noise_scale_low
            label = 'same_low'

        # 添加高斯噪声并裁剪到[0,1]范围
        noise = np.random.normal(0, noise_scale, size=TOTAL_DIM)
        vec = np.clip(target_vector + noise, 0, 1)

        # 单独处理姿态角度，噪声更大，范围在[-30, 30]度
        pose_noise = np.random.normal(0, noise_scale * 5, size=3)
        vec[pose_start:pose_end] = np.clip(target_vector[pose_start:pose_end] + pose_noise, -30, 30)

        vectors.append(vec)
        labels.append(label)
        similarities.append(calculate_weighted_similarity(target_vector, vec))

    # 2. 生成"不同人"向量
    for _ in range(num_diff):
        if np.random.random() < p_diff_high_sim:
            # 生成高相似度的不同人（混淆人）
            noise = np.random.normal(0, confuser_noise_scale * 1.2, size=TOTAL_DIM)
            vec = np.clip(target_vector + noise, 0, 1)
            pose_noise = np.random.normal(0, confuser_noise_scale * 6, size=3)
            vec[pose_start:pose_end] = np.clip(target_vector[pose_start:pose_end] + pose_noise, -30, 30)
            labels.append('diff_high')
        else:
            # 生成低相似度的不同人，直接生成全新的随机向量
            vec = generate_base_vector()
            labels.append('diff_low')

        vectors.append(vec)
        similarities.append(calculate_weighted_similarity(target_vector, vec))

    # 3. 生成"混淆人"向量
    for _ in range(num_confuser):
        # 添加中等程度的噪声
        noise = np.random.normal(0, confuser_noise_scale, size=TOTAL_DIM)
        vec = np.clip(target_vector + noise, 0, 1)

        # 单独处理姿态角度
        pose_noise = np.random.normal(0, confuser_noise_scale * 5, size=3)
        vec[pose_start:pose_end] = np.clip(target_vector[pose_start:pose_end] + pose_noise, -30, 30)

        # 随机改变一个非关键二进制属性，增加混淆性
        if np.random.random() < 0.5:
            expr_start, expr_end = _get_feature_indices('expression')
            # 使用 randint 生成随机表情，兼容旧版本
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

        plt.boxplot(data_for_plot, tick_labels=labels_for_plot, patch_artist=True)
        plt.title('不同类型向量与目标向量的相似度分布（调整后）')
        plt.ylabel('加权余弦相似度')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(0, 1.05)
        plt.savefig("similarity_boxplot_adjusted.png")
    except ImportError:
        print("\n提示：未安装 matplotlib，跳过可视化。")
