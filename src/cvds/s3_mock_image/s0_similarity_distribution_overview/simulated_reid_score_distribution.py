import numpy as np
import matplotlib

matplotlib.use('Agg')  # 必须在 pyplot 之前设置！
import matplotlib.pyplot as plt
from scipy.stats import beta, gaussian_kde


class SimulatedMatcher:
    def __init__(self):
        # 定义三类分布参数（使用 Beta 分布）
        self.dist_params = {
            'same': (8, 2),  # high similarity
            # 原来的参数是(2, 8)，修改为(3, 7)使峰向右偏移
            'different': (3, 7),  # low similarity, shifted right
            'ambiguous': (4, 4)  # medium, more variance
        }

    def generate_score(self, pair_type):
        """
        生成一个相似度得分
        pair_type: 'same', 'different', or 'ambiguous'
        """
        a, b = self.dist_params[pair_type]
        return beta.rvs(a, b)

    def generate_batch(self, n_same, n_diff, n_amb):
        """生成一批得分及其标签"""
        scores = []
        labels = []

        for _ in range(n_same):
            scores.append(self.generate_score('same'))
            labels.append('same')
        for _ in range(n_diff):
            scores.append(self.generate_score('different'))
            labels.append('different')
        for _ in range(n_amb):
            scores.append(self.generate_score('ambiguous'))
            labels.append('ambiguous')

        return np.array(scores), np.array(labels)


# 示例使用
matcher = SimulatedMatcher()
scores, labels = matcher.generate_batch(1000, 1000, 500)

# 可视化 - 使用KDE曲线
plt.figure(figsize=(10, 6))

for label in ['same', 'ambiguous', 'different']:
    # 提取对应类别的分数
    label_scores = scores[labels == label]

    # 创建KDE对象
    kde = gaussian_kde(label_scores)

    # 生成x轴数据点
    x = np.linspace(0, 1, 200)

    # 计算KDE值
    kde_values = kde(x)

    # 绘制平滑曲线
    plt.plot(x, kde_values, label=label, linewidth=2)

    # 添加阴影区域增强视觉效果
    plt.fill_between(x, kde_values, alpha=0.2)

plt.legend()
plt.xlabel('Similarity Score')
plt.ylabel('Density')
plt.title('Simulated Score Distributions (KDE)')
plt.grid(True, alpha=0.3)
plt.savefig("genuine_vs_impostor_kde.png")