import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------- 1. 特征维度配置（含权重） ----------------------
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


# ---------------------- 2. 生成特征向量的辅助函数 ----------------------
def generate_person_vector(
        gender=1, age=1, expression=0, glasses=1, mask=0,
        pose_yaw=5.0, pose_pitch=2.0, pose_roll=1.0,
        noise_scale=0.01, base_skin_color=np.array([0.58, 0.45, 0.39])
):
    keypoints = np.clip(np.random.normal(0.5, 0.1, 212) + np.random.normal(0, noise_scale, 212), 0, 1)
    skin_color = np.clip(base_skin_color + np.random.normal(0, noise_scale, 3), 0, 1)
    skin_var = np.clip(np.array([0.02, 0.015, 0.01]) + np.random.normal(0, noise_scale / 10, 3), 0, 0.1)
    lbp = np.clip(np.random.normal(0.5, 0.1, 50) + np.random.normal(0, noise_scale, 50), 0, 1)
    contour = np.clip(np.array([0.6, 0.4, 0.5, 0.3, 0.7]) + np.random.normal(0, noise_scale, 5), 0, 1)
    glasses_vec = np.array([glasses])
    mask_vec = np.array([mask])
    hair = np.clip(np.random.normal(0.5, 0.1, 8) + np.random.normal(0, noise_scale, 8), 0, 1)
    gender_vec = np.array([gender])
    age_vec = np.array([age])
    expression_vec = np.array([expression])
    pose = np.clip(np.array([pose_yaw, pose_pitch, pose_roll]) + np.random.normal(0, noise_scale * 5, 3), -30, 30)
    quality = np.clip(np.array([0.85, 0.9]) + np.random.normal(0, noise_scale, 2), 0, 1)
    occlusion = np.clip(np.array([0.1, 0.05, 0.08]) + np.random.normal(0, noise_scale, 3), 0, 1)

    return np.hstack([
        keypoints, skin_color, skin_var, lbp, contour,
        glasses_vec, mask_vec, hair, gender_vec, age_vec,
        expression_vec, pose, quality, occlusion
    ])


# ---------------------- 3. 生成测试向量 ----------------------
np.random.seed(42)

# 同一个人 (Person A) 的两个样本
person_A1 = generate_person_vector(gender=1, age=1, expression=0, glasses=1, mask=0, noise_scale=0.005)
person_A2 = generate_person_vector(gender=1, age=1, expression=1, glasses=1, mask=0, noise_scale=0.005)

# 完全不同的人 (Person B)
person_B = generate_person_vector(gender=0, age=2, expression=2, glasses=0, mask=1, noise_scale=0.01)

# 高相似但不同的人 (Person D)
# 与 A 性别、年龄、是否戴眼镜相同，但肤色、发型、姿态略有不同
person_D = generate_person_vector(
    gender=1, age=1, expression=0, glasses=1, mask=0,
    pose_yaw=8.0, pose_pitch=4.0, pose_roll=3.0,  # 姿态有差异
    noise_scale=0.02,  # 增加噪声，模拟更大的个体差异
    base_skin_color=np.array([0.62, 0.48, 0.41])  # 肤色相似但不同
)


# ---------------------- 4. 相似度计算函数 ----------------------
def calculate_dimension_similarity(vec1, vec2, config):
    name, dtype, dim, weight = config
    start = 0
    for n, d, s, w in feature_config:
        if n == name: break
        start += s
    end = start + dim

    feat1, feat2 = vec1[start:end], vec2[start:end]

    if dtype == 'binary':
        return 1.0 if np.array_equal(feat1, feat2) else 0.0
    elif dtype == 'array':
        return cosine_similarity([feat1], [feat2])[0][0] if dim > 0 else 0


def calculate_weighted_similarity(vec1, vec2, configs):
    total_sim = 0.0
    dim_sims = []
    for config in configs:
        dim_sim = calculate_dimension_similarity(vec1, vec2, config)
        dim_sims.append((config[0], dim_sim, config[3]))
        total_sim += dim_sim * config[3]
    return total_sim, dim_sims


# ---------------------- 5. 执行比对并输出结果 ----------------------
sim_A1_A2, _ = calculate_weighted_similarity(person_A1, person_A2, feature_config)
sim_A1_B, _ = calculate_weighted_similarity(person_A1, person_B, feature_config)
sim_A1_D, dim_sims_A1_D = calculate_weighted_similarity(person_A1, person_D, feature_config)

print("=" * 60)
print("相似度比对结果汇总")
print("=" * 60)
print(f"1. 【同一人】A1 vs A2 的相似度: {sim_A1_A2:.4f}")
print(f"2. 【不同人】A1 vs B  的相似度: {sim_A1_B:.4f}")
print(f"3. 【高相似不同人】A1 vs D 的相似度: {sim_A1_D:.4f}")
print("\n" + "=" * 60)

# 详细打印 A1 vs D 的各维度相似度
print("\n【高相似不同人 (A1 vs D) 的维度贡献分析】")
print("-" * 60)
print(f"{'特征维度':<15} {'相似度':<10} {'权重':<10} {'加权贡献':<10}")
print("-" * 60)
for name, sim, weight in dim_sims_A1_D:
    print(f"{name:<15} {sim:.4f}      {weight:.3f}        {sim * weight:.4f}")