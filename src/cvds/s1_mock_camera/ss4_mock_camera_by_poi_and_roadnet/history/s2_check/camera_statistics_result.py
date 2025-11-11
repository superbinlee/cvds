import pandas as pd
import os

# 文件路径
file_path = './cameras_all.xlsx'

# 检查文件是否存在
if not os.path.exists(file_path):
    raise FileNotFoundError(f"文件未找到: {file_path}")

# 读取 Excel 文件（默认第一个 sheet）
try:
    df = pd.read_excel(file_path, engine='openpyxl')
except Exception as e:
    raise Exception(f"读取 Excel 文件失败: {e}")

# 1. 总数（包含标题行后减1，或直接统计数据行）
total_count = len(df)
print(f"总数: {total_count}")

# 2. 按 district（区域）统计数量
if 'district' in df.columns:
    district_counts = df['district'].value_counts().sort_index()
    print("\n按区域 (district) 统计数量:")
    print(district_counts)
else:
    print("警告: 列 'district' 不存在！")
    district_counts = None

# 3. 按 source_type 和 source 组合统计（构造 source 列）
if 'source_type' in df.columns and 'source' in df.columns:
    df['source_combined'] = df['source_type'].astype(str) + " " + df['source'].astype(str)
    source_counts = df['source_combined'].value_counts().sort_index()
    print("\n按 source 统计数量:")
    print(source_counts)
else:
    print("警告: 缺少 'source_type' 或 'source' 列，无法统计 source")
    source_counts = None

# 可选：保存结果到新的 Excel 文件
output_file = 'camera_statistics_result.xlsx'
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # 总数
    pd.DataFrame({'统计项': ['摄像头总数'], '数量': [total_count]}).to_excel(writer, sheet_name='Summary', index=False)

    # 按区域
    if district_counts is not None:
        district_counts.reset_index().to_excel(writer, sheet_name='By_District', index=False)

    # 按 source
    if source_counts is not None:
        source_counts.reset_index().to_excel(writer, sheet_name='By_Source', index=False)

print(f"\n统计完成！结果已保存至: {output_file}")
