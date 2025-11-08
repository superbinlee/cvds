# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm

# ==================== 1. 中文字体设置 ====================
def get_chinese_font():
    candidates = [
        'SimHei', 'Microsoft YaHei', 'PingFang SC',
        'Noto Sans CJK SC', 'Source Han Sans CN', 'Heiti TC',
        'WenQuanYi Micro Hei', 'Arial Unicode MS'
    ]
    for font in candidates:
        if font in [f.name for f in fm.fontManager.ttflist]:
            return font
    return 'DejaVu Sans'

plt.rcParams['font.sans-serif'] = [get_chinese_font()]
plt.rcParams['axes.unicode_minus'] = False

# ==================== 2. 生成模拟数据（你原始数据，保持不变） ====================
np.random.seed(42)

same_person = np.random.normal(loc=0.88, scale=0.05, size=10000)
same_person = np.clip(same_person, 0, 1)

boundary = np.random.normal(loc=0.78, scale=0.09, size=10000)
boundary = np.clip(boundary, 0, 1)

different = np.random.normal(loc=0.35, scale=0.13, size=10000)
different = np.clip(different, 0, 1)

# ==================== 3. KDE 估计（不归一化） ====================
x_grid = np.linspace(0, 1, 1000)

kde_same = gaussian_kde(same_person, bw_method=0.03)
kde_boundary = gaussian_kde(boundary, bw_method=0.06)
kde_diff = gaussian_kde(different, bw_method=0.08)

y_same = kde_same(x_grid)
y_boundary = kde_boundary(x_grid)
y_diff = kde_diff(x_grid)

# ==================== 4. 绘图 ====================
fig, ax = plt.subplots(figsize=(12, 7), dpi=150)

# 背景风险区
ax.axvspan(0, 0.70, color='#e6f2ff', alpha=0.6, zorder=0)
ax.axvspan(0.70, 1.0, color='#fff3e6', alpha=0.6, zorder=0)

# 填充曲线（从下往上）
ax.fill_between(x_grid, 0, y_diff, color='#ff9f40', alpha=0.9, label='不同人 (均值0.35)')
ax.fill_between(x_grid, 0, y_boundary, color='#66c2a5', alpha=0.9, label='边界样本 (均值0.78)')
ax.fill_between(x_grid, 0, y_same, color='#5ab4ac', alpha=0.9, label='同一个人 (均值0.88)')

# 推荐阈值线
ax.axvline(0.70, color='red', linestyle='--', linewidth=2, label='推荐阈值: 0.70')

# ==================== 5. 智能防溢出标注（无箭头线，框内） ====================
def annotate_peak_safe(x, y, kde_obj, label, color, ax, buffer=0.06):
    peak_idx = np.argmax(y)
    peak_x = x[peak_idx]
    peak_y = y[peak_idx]
    mean_val = kde_obj.dataset.mean()

    # 初始目标位置（右上方）
    target_x = peak_x + 0.03
    target_y = peak_y + 0.9

    # 获取当前坐标轴范围
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # 防溢出调整
    if target_x > xlim[1] - buffer:
        target_x = peak_x - 0.08  # 改到左侧
    if target_y > ylim[1] - 1.2:
        target_y = peak_y - 1.8   # 改到下方

    ax.annotate(f'{label}\n均值{mean_val:.2f}\n峰值{peak_y:.2f}',
                xy=(peak_x, peak_y),
                xytext=(target_x, target_y),
                fontsize=11, color=color,
                bbox=dict(boxstyle="round,pad=0.32", facecolor='white', edgecolor=color, alpha=0.88),
                ha='center', va='center')

# 调用标注
annotate_peak_safe(x_grid, y_same, kde_same, '同一个人', '#5ab4ac', ax)
annotate_peak_safe(x_grid, y_boundary, kde_boundary, '边界样本', '#66c2a5', ax)
annotate_peak_safe(x_grid, y_diff, kde_diff, '不同人', '#ff9f40', ax)

# ==================== 6. 美化 ====================
ax.set_title('图像相似度分布仿真 (KDE平滑曲线)\n'
             '蓝色=同一个人 | 绿色=边界样本 | 橙色=不同人',
             fontsize=14, pad=20)

ax.set_xlabel('相似度分数', fontsize=12)
ax.set_ylabel('概率密度', fontsize=12)

ax.set_xlim(0, 1)
ax.set_ylim(0, max(y_same.max(), 12))  # 保证空间
ax.grid(True, alpha=0.3)

# 图例
legend_elements = [
    mpatches.Patch(color='#e6f2ff', alpha=0.6, label='高阈值风险区 (同人被判低)'),
    mpatches.Patch(color='#fff3e6', alpha=0.6, label='边界风险区 (不同人被判高)'),
    mpatches.Patch(color='#5ab4ac', label='同一个人 (均值0.88)'),
    mpatches.Patch(color='#66c2a5', label='边界样本 (均值0.78)'),
    mpatches.Patch(color='#ff9f40', label='不同人 (均值0.35)'),
    plt.Line2D([0], [0], color='red', ls='--', lw=2, label='推荐阈值: 0.70')
]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.01, 0.98), fontsize=10)

plt.tight_layout()

# ==================== 7. 保存高清图 ====================
output_file = '相似度分布图_标注框内无箭头.png'
plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
print(f"图已保存：{output_file}")
print(f"当前用户：@lucky_binlee | 时间：2025-11-07 15:29 HKT")

# plt.show()  # PyCharm 建议注释