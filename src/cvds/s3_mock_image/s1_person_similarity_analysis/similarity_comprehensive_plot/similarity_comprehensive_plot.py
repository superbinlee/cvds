import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# 步骤1: 生成模拟数据
def generate_similarity_data():
    # 基于您之前提供的统计数据生成模拟数据
    sim_by_label = {
        'same_high': np.random.normal(0.8510, 0.0510, 10000).clip(0, 1),  # 相同人高相似度
        'same_low': np.random.normal(0.6801, 0.0489, 10000).clip(0, 1),  # 相同人低相似度

        'diff_high': np.random.normal(0.7536, 0.0314, 10000).clip(0, 1),  # 不同人高相似度
        'diff_low': np.random.normal(0.4643, 0.0633, 10000).clip(0, 1),  # 不同人低相似度

        'confuser': np.random.normal(0.7855, 0.0409, 10000).clip(0, 1)  # 混淆样本
    }

    # 确保每个类别的最小值和最大值符合统计数据
    sim_by_label['same_high'][-1] = 0.9999  # 设置最大值
    sim_by_label['same_high'][0] = 0.7495  # 设置最小值

    sim_by_label['same_low'][-1] = 0.8569
    sim_by_label['same_low'][0] = 0.5070

    sim_by_label['diff_high'][-1] = 0.8902
    sim_by_label['diff_high'][0] = 0.6594

    sim_by_label['diff_low'][-1] = 0.7028
    sim_by_label['diff_low'][0] = 0.4765

    sim_by_label['confuser'][-1] = 0.8883
    sim_by_label['confuser'][0] = 0.6057

    return sim_by_label


# 步骤2: 打印统计信息
def print_statistics(sim_by_label):
    print("各类型向量与目标向量的相似度统计：")
    for label, similarities in sim_by_label.items():
        print(f"  - {label:9}: 数量={len(similarities):2d}  均值={np.mean(similarities):.4f}  标准差={np.std(similarities):.4f}  最小值={np.min(similarities):.4f}  最大值={np.max(similarities):.4f}")


# 步骤3: 可视化结果
def visualize_results(sim_by_label):
    try:
        import matplotlib.pyplot as plt

        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 创建一个2x1的子图布局
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 1.5]})

        labels_for_plot = ['same_high', 'same_low', 'confuser', 'diff_high', 'diff_low']
        data_for_plot = [sim_by_label[label] for label in labels_for_plot if label in sim_by_label]
        labels_for_plot = [label for label in labels_for_plot if label in sim_by_label]

        # 设置颜色方案
        colors = ['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336']

        # 1. 第一个子图：箱线图（保留但改进）
        box_plot = ax1.boxplot(data_for_plot, tick_labels=labels_for_plot, patch_artist=True)
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax1.set_title('不同类型向量与目标向量的相似度分布')
        ax1.set_ylabel('加权余弦相似度')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_ylim(0, 1.05)

        # 添加阈值参考线
        ax1.axhline(y=0.75, color='black', linestyle='-', alpha=0.8, label='典型阈值 (0.75)')
        ax1.legend()

        # 2. 第二个子图：带密度曲线的直方图
        # 首先计算每个类别的均值
        means = [np.mean(data) for data in data_for_plot]

        # 设置透明度和直方图的bin数量
        alpha = 0.6
        bins = 20

        # 为每个类别绘制直方图
        for i, (data, label, color) in enumerate(zip(data_for_plot, labels_for_plot, colors)):
            # 绘制直方图
            n, bins, patches = ax2.hist(data, bins=bins, alpha=alpha, color=color,
                                        edgecolor='black', linewidth=1, label=label)

            # 在直方图上方绘制均值线
            ax2.axvline(x=means[i], color=color, linestyle='--', linewidth=2,
                        label=f'{label} 均值: {means[i]:.4f}')

        ax2.set_xlabel('加权余弦相似度')
        ax2.set_ylabel('样本数量')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_xlim(0, 1.05)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # 添加文本说明
        textstr = ("相同人特征通常在0.75以上\n"
                   "不同人特征通常在0.75以下\n"
                   "diff_high 为容易误识别的样本")
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)

        plt.tight_layout()
        plt.savefig("similarity_comprehensive_plot.png", dpi=300, bbox_inches='tight')
        print("\n已生成更全面的可视化图表: similarity_comprehensive_plot.png")

        # 也生成单独的直方图版本（更简单直观）
        plt.figure(figsize=(10, 6))
        for i, (data, label, color) in enumerate(zip(data_for_plot, labels_for_plot, colors)):
            plt.hist(data, bins=20, alpha=0.6, color=color, edgecolor='black',
                     label=label)
            plt.axvline(x=np.mean(data), color=color, linestyle='--', linewidth=2)

        plt.title('不同类型向量的相似度分布（直方图）')
        plt.xlabel('加权余弦相似度')
        plt.ylabel('样本数量')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.savefig("similarity_histogram.png", dpi=300, bbox_inches='tight')
        print("已生成简单直方图版本: similarity_histogram.png")

        # 可选：生成小提琴图（展示分布形状）
        plt.figure(figsize=(10, 6))
        violin_parts = plt.violinplot(data_for_plot, showmeans=True, showmedians=True)
        for i, pc in enumerate(violin_parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)

        plt.xticks(range(1, len(labels_for_plot) + 1), labels_for_plot)
        plt.title('不同类型向量的相似度分布（小提琴图）')
        plt.ylabel('加权余弦相似度')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig("similarity_violin_plot.png", dpi=300, bbox_inches='tight')
        print("已生成小提琴图版本: similarity_violin_plot.png")

        # 可选：显示图表（如果在交互式环境中运行）
        # plt.show()

    except ImportError:
        print("\n提示：未安装 matplotlib，跳过可视化。")
    except Exception as e:
        print(f"\n可视化过程中出错: {str(e)}")


# 主函数
def main():
    # 生成数据
    sim_by_label = generate_similarity_data()

    # 打印统计信息
    print_statistics(sim_by_label)

    # 可视化结果
    visualize_results(sim_by_label)


if __name__ == "__main__":
    main()
