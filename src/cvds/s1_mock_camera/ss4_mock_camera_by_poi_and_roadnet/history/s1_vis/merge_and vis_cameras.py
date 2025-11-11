# -*- coding: utf-8 -*-
"""
任务：
1. 合并两个 Excel 文件（不去重）
   → 打印：原始数量 + 合并后数量
   → 输出 cameras_all.xlsx

2. 绘制行政区边界 + 所有相机点位 → chengdu_cameras_with_districts.html
"""

import pandas as pd
import geopandas as gpd
from shapely.wkt import loads
import folium
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ==================== 文件路径 ====================
BASE_DIR = Path(".")

boundary_csv = BASE_DIR / "districts" / "chengdu_districts_boundary.csv"
smart_xlsx = BASE_DIR / "mock_camera_by_poi" / "cameras_boundary_smart.xlsx"
inter_xlsx = BASE_DIR / "mock_camera_by_roadnet" / "main_intersections_camera.xlsx"

output_xlsx = BASE_DIR / "cameras_all.xlsx"
output_html = BASE_DIR / "chengdu_cameras_with_districts.html"

# ==================== 1. 读取并合并相机表格（不去重） ====================
print("正在读取相机数据...")

df_poi = pd.read_excel(smart_xlsx)
df_road = pd.read_excel(inter_xlsx)

# 打印原始数量
print(f"POI 相机数量: {len(df_poi)}")
print(f"路口相机数量: {len(df_road)}")

# 添加来源标识
df_poi['source'] = 'POI边界智能相机'
df_road['source'] = '主干路口相机'

# 确保列顺序一致（补全缺失列）
all_cols = ['camera_id', 'name', 'latitude', 'longitude', 'district', 'source_type', 'source']
for df in [df_poi, df_road]:
    for col in all_cols:
        if col not in df.columns:
            df[col] = None

df_poi = df_poi[all_cols]
df_road = df_road[all_cols]

# 合并（不去重）
df_all = pd.concat([df_poi, df_road], ignore_index=True)

# 打印合并后数量
print(f"合并后总数量: {len(df_all)}")
print(f"   保存至：{output_xlsx}")
df_all.to_excel(output_xlsx, index=False)

# ==================== 2. 读取行政区边界 ====================
print("\n正在读取行政区边界...")

df_bound = pd.read_csv(boundary_csv)

def safe_load_wkt(wkt_str):
    wkt_str = str(wkt_str).strip().strip('"')
    # 补全缺失括号
    open_count = wkt_str.count('(')
    close_count = wkt_str.count(')')
    if open_count > close_count:
        wkt_str += ')' * (open_count - close_count)
    try:
        return loads(wkt_str)
    except Exception as e:
        print(f"WKT 解析失败: {wkt_str[:60]}... → {e}")
        return None

df_bound['geometry'] = df_bound['区域边界'].apply(safe_load_wkt)
df_bound = df_bound.dropna(subset=['geometry'])

gdf_bound = gpd.GeoDataFrame(df_bound, geometry='geometry', crs="EPSG:4326")
print(f"成功加载 {len(gdf_bound)} 个行政区边界")

# ==================== 3. Folium 绘图 ====================
print("\n正在生成交互地图...")

# 地图中心
center_lat = df_all['latitude'].mean()
center_lon = df_all['longitude'].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB positron")

# 行政区颜色
district_colors = {
    '金牛区': '#1f77b4', '成华区': '#ff7f0e', '郫都区': '#2ca02c',
    '新都区': '#d62728', '青羊区': '#9467bd'
}

# 绘制边界
for _, row in gdf_bound.iterrows():
    color = district_colors.get(row['区域名称'], '#808080')
    folium.GeoJson(
        row['geometry'],
        style_function=lambda x, c=color: {
            'fillColor': c, 'color': 'black', 'weight': 2.5, 'fillOpacity': 0.15
        },
        tooltip=f"<b>{row['区域名称']}</b>"
    ).add_to(m)

# 绘制相机点位
for _, row in df_all.iterrows():
    color = 'red' if row['source'] == 'POI边界智能相机' else 'blue'
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5,
        color=color,
        weight=1,
        fill=True,
        fillOpacity=0.9,
        popup=folium.Popup(
            f"""
            <b>相机ID:</b> {row['camera_id']}<br>
            <b>名称:</b> {row['name']}<br>
            <b>区域:</b> {row['district']}<br>
            <b>来源:</b> {row['source']}<br>
            <b>经纬度:</b> {row['longitude']:.6f}, {row['latitude']:.6f}
            """,
            max_width=300
        ),
        tooltip=f"{row['camera_id']}"
    ).add_to(m)

# 图例
legend_html = '''
<div style="position: fixed; bottom: 20px; left: 20px; width: 190px; height: 90px; 
     border:2px solid grey; z-index:9999; font-size:14px; background:white;
     padding: 10px; border-radius: 8px;">
  <p style="margin:3px"><i class="fa fa-circle" style="color:red"></i> POI边界智能相机</p>
  <p style="margin:3px"><i class="fa fa-circle" style="color:blue"></i> 主干路口相机</p>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# 保存
m.save(output_html)
print(f"地图生成完成 → {output_html}")
print("\n所有任务完成！")