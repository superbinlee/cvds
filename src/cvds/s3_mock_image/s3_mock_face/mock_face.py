import numpy as np
import matplotlib.pyplot as plt

# --- 1. 特征规格与权重定义 ---
FEATURE_DIMENSIONS = {
    'keypoints': 304,
    'skin_color': 32,
    'lbp': 64,
    'contour': 32,
    'glasses': 1,
    'mask': 1,
    'hair': 32,
    'gender': 1,
    'age': 1,
    'expression': 1,
    'pose': 16,
    'quality': 8,
    'occlusion': 8
}

WEIGHTS = {
    'keypoints': 0.3,
    'skin_color': 0.1,
    'lbp': 0.1,
    'contour': 0.05,
    'glasses': 0.05,
    'mask': 0.05,
    'hair': 0.05,
    'gender': 0.05,
    'age': 0.05,
    'expression': 0.05,
    'pose': 0.05,
    'quality': 0.05,
    'occlusion': 0.025
}

SKIN_COLOR_CATEGORIES = [f'skin_{i}' for i in range(FEATURE_DIMENSIONS['skin_color'])]
HAIR_TYPE_CATEGORIES = [f'hair_{i}' for i in range(FEATURE_DIMENSIONS['hair'])]


# --- 2. 特征生成、拆分与比对逻辑 ---
def _generate_onehot(categories):
    choice = np.random.choice(categories)
    vec = np.zeros(len(categories))
    vec[categories.index(choice)] = 1
    return vec


def generate_person_feature(person_id, noise_level=0.05):
    rng = np.random.default_rng(person_id)
    features = {}
    features['keypoints'] = rng.standard_normal(FEATURE_DIMENSIONS['keypoints'])
    features['skin_color'] = _generate_onehot(SKIN_COLOR_CATEGORIES)
    features['lbp'] = rng.standard_normal(FEATURE_DIMENSIONS['lbp'])
    features['contour'] = rng.standard_normal(FEATURE_DIMENSIONS['contour'])
    features['glasses'] = np.array([rng.integers(0, 2)])
    features['mask'] = np.array([rng.integers(0, 2)])
    features['hair'] = _generate_onehot(HAIR_TYPE_CATEGORIES)
    features['gender'] = np.array([rng.integers(0, 2)])
    features['age'] = np.array([rng.integers(0, 2)])
    features['expression'] = np.array([rng.integers(0, 2)])
    features['pose'] = rng.standard_normal(FEATURE_DIMENSIONS['pose'])
    features['quality'] = rng.standard_normal(FEATURE_DIMENSIONS['quality'])
    features['occlusion'] = rng.standard_normal(FEATURE_DIMENSIONS['occlusion'])

    combined_vector = np.concatenate([features[name] for name in FEATURE_DIMENSIONS.keys()])
    noise = np.random.standard_normal(combined_vector.shape) * noise_level
    final_combined_vector = combined_vector + noise
    return final_combined_vector


def split_features(combined_vector):
    split_features = {}
    start = 0
    for feature_name, dim in FEATURE_DIMENSIONS.items():
        end = start + dim
        split_features[feature_name] = combined_vector[start:end]
        start = end
    return split_features


def compare_features(vec1, vec2):
    features1 = split_features(vec1)
    features2 = split_features(vec2)
    total_similarity = 0.0
    for feature_name, weight in WEIGHTS.items():
        v1 = features1[feature_name]
        v2 = features2[feature_name]
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 < 1e-10 or norm_v2 < 1e-10:
            sim = 0.0 if (norm_v1 < 1e-10 and norm_v2 < 1e-10) else 0.5
        else:
            sim = np.dot(v1, v2) / (norm_v1 * norm_v2)
        total_similarity += sim * weight
    return total_similarity


# --- 3. 批量生成比对数据（进一步调整参数） ---
def generate_similarity_distribution():
    sim_by_label = {
        'same_high': [],
        'same_low': [],
        'confuser': [],
        'diff_high': [],
        'diff_low': []
    }

    # 1. 提升 same_high：使用更小的噪声
    for person_id in range(1000):
        vec1 = generate_person_feature(person_id, noise_level=0.005)  # 降低噪声
        vec2 = generate_person_feature(person_id, noise_level=0.01)  # 降低噪声
        sim_by_label['same_high'].append(compare_features(vec1, vec2))

    # 2. 降低 same_low：使用更大的噪声
    for person_id in range(1000):
        vec1 = generate_person_feature(person_id, noise_level=0.005)
        vec2 = generate_person_feature(person_id, noise_level=0.4)  # 显著提高噪声
        sim_by_label['same_low'].append(compare_features(vec1, vec2))

    # 生成 diff_high (保持不变)
    for i in range(1000):
        p1_id = i * 2
        p2_id = i * 2 + 1
        vec1 = generate_person_feature(p1_id, noise_level=0.05)
        vec2 = generate_person_feature(p2_id, noise_level=0.05)
        feat1 = split_features(vec1)
        feat2 = split_features(vec2)
        feat2['keypoints'] = feat1['keypoints'] * 0.6 + feat2['keypoints'] * 0.4
        feat2['lbp'] = feat1['lbp'] * 0.6 + feat2['lbp'] * 0.4
        vec2 = np.concatenate([feat2[name] for name in FEATURE_DIMENSIONS.keys()])
        sim = compare_features(vec1, vec2)
        sim_by_label['diff_high'].append(sim)

    # 生成 diff_low (保持不变)
    for i in range(1000):
        p1_id = i * 2 + 2000
        p2_id = i * 2 + 2001
        vec1 = generate_person_feature(p1_id, noise_level=0.05)
        vec2 = generate_person_feature(p2_id, noise_level=0.05)
        sim = compare_features(vec1, vec2)
        sim_by_label['diff_low'].append(sim)

    # 3. 提升 confuser：共享更多、比例更高的特征
    target_id = 9999
    target_vec = generate_person_feature(target_id, noise_level=0.005)
    for i in range(500):
        confuser_id = i + 3000
        confuser_vec = generate_person_feature(confuser_id, noise_level=0.05)
        feat_target = split_features(target_vec)
        feat_confuser = split_features(confuser_vec)
        # 提高共享比例并增加共享特征
        feat_confuser['keypoints'] = feat_target['keypoints'] * 0.9 + feat_confuser['keypoints'] * 0.1
        feat_confuser['lbp'] = feat_target['lbp'] * 0.9 + feat_confuser['lbp'] * 0.1
        feat_confuser['pose'] = feat_target['pose'] * 0.8 + feat_confuser['pose'] * 0.2  # 新增共享特征
        feat_confuser['skin_color'] = feat_target['skin_color']
        feat_confuser['hair'] = feat_target['hair']
        confuser_vec = np.concatenate([feat_confuser[name] for name in FEATURE_DIMENSIONS.keys()])
        sim = compare_features(target_vec, confuser_vec)
        sim_by_label['confuser'].append(sim)

    # 转换为NumPy数组并裁剪范围
    for label in sim_by_label:
        sim_by_label[label] = np.clip(np.array(sim_by_label[label]), 0, 1)

    return sim_by_label


# --- 4. 统计与可视化 ---
def print_statistics(sim_by_label):
    print("各类型特征对的相似度统计：")
    for label, sims in sim_by_label.items():
        if len(sims) > 0:
            print(f"  - {label:9}: 数量={len(sims):5d}  均值={np.mean(sims):.4f}  标准差={np.std(sims):.4f}  最小值={np.min(sims):.4f}  最大值={np.max(sims):.4f}")


def visualize_similarity_distribution(sim_by_label):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    labels = ['same_high', 'same_low', 'confuser', 'diff_high', 'diff_low']
    data = []
    plot_labels = []
    for label in labels:
        sims = sim_by_label[label]
        if len(sims) > 0:
            data.append(sims)
            plot_labels.append(label)

    colors = ['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336']

    plt.figure(figsize=(12, 8))
    violin = plt.violinplot(data, showmeans=False, showmedians=True)

    for i, pc in enumerate(violin['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.8)

    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        vp = violin[partname]
        vp.set_edgecolor('blue')
        vp.set_linewidth(2)

    plt.xticks(range(1, len(plot_labels) + 1), plot_labels)
    plt.title('不同类型向量的相似度分布（小提琴图）', fontsize=16)
    plt.ylabel('加权余弦相似度', fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7, color='gray')

    plt.tight_layout()
    plt.savefig("similarity_distribution_vector_based_v2.png", dpi=300, bbox_inches='tight')
    print("\n已生成调整后的相似度分布图：similarity_distribution_vector_based_v2.png")


# --- 5. 主函数 ---
def main():
    print("--- 开始生成基于特征向量合成的相似度数据（V2） ---")
    sim_distribution = generate_similarity_distribution()
    print_statistics(sim_distribution)
    visualize_similarity_distribution(sim_distribution)
    print("--- 任务完成 ---")


if __name__ == "__main__":
    main()