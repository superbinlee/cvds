import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


class PersonVectorGenerator:
    def __init__(self, vector_dim=256, same_person_noise=0.1, different_person_scale=1.0):
        """
        初始化向量生成器

        参数:
        vector_dim: 向量维度
        same_person_noise: 相同人向量的噪声水平，值越小向量越相似
        different_person_scale: 不同人向量的分布范围缩放因子
        """
        self.vector_dim = vector_dim
        self.same_person_noise = same_person_noise
        self.different_person_scale = different_person_scale

    def generate_base_vector(self):
        """生成一个基础向量，作为某个人的特征表示"""
        # 使用正态分布生成基础向量
        return np.random.randn(self.vector_dim)

    def generate_same_person_vectors(self, n_vectors=2, base_vector=None):
        """
        生成同一个人的多个向量（相似但不完全相同）

        参数:
        n_vectors: 要生成的向量数量
        base_vector: 可选的基础向量，如果不提供则自动生成

        返回:
        向量列表和对应的基础向量
        """
        if base_vector is None:
            base_vector = self.generate_base_vector()

        vectors = []
        for _ in range(n_vectors):
            # 在基础向量上添加小噪声
            noise = self.same_person_noise * np.random.randn(self.vector_dim)
            vector = base_vector + noise
            # 归一化向量
            vector = vector / np.linalg.norm(vector)
            vectors.append(vector)

        # 基础向量也归一化
        base_vector = base_vector / np.linalg.norm(base_vector)

        return vectors, base_vector

    def generate_different_person_vectors(self, n_persons=2, n_vectors_per_person=2):
        """
        生成不同人的向量集合

        参数:
        n_persons: 不同人的数量
        n_vectors_per_person: 每个人生成的向量数量

        返回:
        向量列表和对应的人员ID标签
        """
        all_vectors = []
        person_ids = []

        for person_id in range(n_persons):
            # 为每个人生成不同的基础向量，确保不同人之间有足够的区分度
            base_vector = self.different_person_scale * np.random.randn(self.vector_dim)

            # 为这个人生成多个向量
            person_vectors, _ = self.generate_same_person_vectors(
                n_vectors_per_person, base_vector
            )

            all_vectors.extend(person_vectors)
            person_ids.extend([person_id] * n_vectors_per_person)

        return np.array(all_vectors), np.array(person_ids)

    def compute_similarity_matrix(self, vectors):
        """计算向量之间的余弦相似度矩阵"""
        return cosine_similarity(vectors)

    def visualize_similarity(self, vectors, person_ids=None, title="Similarity Matrix"):
        """可视化向量之间的相似度矩阵"""
        similarity_matrix = self.compute_similarity_matrix(vectors)

        plt.figure(figsize=(10, 8))
        plt.imshow(similarity_matrix, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(label='Cosine Similarity')
        plt.title(title)

        # 如果提供了人员ID，可以添加分隔线显示不同的人群
        if person_ids is not None:
            # 找出不同人群的边界
            boundaries = []
            current_id = person_ids[0]
            for i, pid in enumerate(person_ids):
                if pid != current_id:
                    boundaries.append(i)
                    current_id = pid

            # 添加垂直线和水平线
            for boundary in boundaries:
                plt.axvline(x=boundary - 0.5, color='white', linestyle='--', linewidth=1)
                plt.axhline(y=boundary - 0.5, color='white', linestyle='--', linewidth=1)

        plt.savefig("similarity_visualization.png")
        plt.close()

    def evaluate_vector_quality(self, same_vectors, diff_vectors):
        """评估生成向量的质量，计算相同人和不同人之间的相似度统计"""
        # 计算相同人向量之间的相似度
        same_similarities = []
        for i in range(len(same_vectors)):
            for j in range(i + 1, len(same_vectors)):
                sim = cosine_similarity([same_vectors[i]], [same_vectors[j]])[0][0]
                same_similarities.append(sim)

        # 计算不同人向量之间的相似度
        diff_similarities = []
        for same_vec in same_vectors:
            for diff_vec in diff_vectors:
                sim = cosine_similarity([same_vec], [diff_vec])[0][0]
                diff_similarities.append(sim)

        return {
            'same_mean': np.mean(same_similarities),
            'same_std': np.std(same_similarities),
            'different_mean': np.mean(diff_similarities),
            'different_std': np.std(diff_similarities),
            'separation': np.mean(same_similarities) - np.mean(diff_similarities)
        }


# 示例使用
if __name__ == "__main__":
    # 创建向量生成器实例
    generator = PersonVectorGenerator(
        vector_dim=128,
        same_person_noise=0.1,
        different_person_scale=1.0
    )

    # 生成同一个人的多个向量
    same_person_vectors, base_vector = generator.generate_same_person_vectors(n_vectors=5)

    # 生成不同人的向量
    all_vectors, person_ids = generator.generate_different_person_vectors(
        n_persons=5,
        n_vectors_per_person=3
    )

    # 可视化相似度矩阵
    generator.visualize_similarity(all_vectors, person_ids, "Similarity Matrix of Different Persons")

    # 评估向量质量
    # 为了评估，我们需要获取不同人的向量
    diff_person_vectors = []
    for i in range(1, 5):  # 取后面4个人的第一个向量作为不同人样本
        idx = np.where(person_ids == i)[0][0]
        diff_person_vectors.append(all_vectors[idx])

    quality_metrics = generator.evaluate_vector_quality(same_person_vectors[:4], diff_person_vectors)

    print("向量质量评估结果:")
    print(f"相同人平均相似度: {quality_metrics['same_mean']:.4f} ± {quality_metrics['same_std']:.4f}")
    print(f"不同人平均相似度: {quality_metrics['different_mean']:.4f} ± {quality_metrics['different_std']:.4f}")
    print(f"分离度: {quality_metrics['separation']:.4f}")