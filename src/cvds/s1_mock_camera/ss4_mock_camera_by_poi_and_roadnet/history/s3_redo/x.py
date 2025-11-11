import pandas as pd
import folium
from shapely import wkt
import geopandas as gpd
import os

# ==================== 1. 读取摄像头数据 ====================
cameras_file = '../cameras_all.xlsx'
if not os.path.exists(cameras_file):
    raise FileNotFoundError(f"未找到: {cameras_file}")

df_cameras = pd.read_excel(cameras_file)

# 确保列名正确（根据你之前数据）
required_cols = ['camera_id', 'name', 'latitude', 'longitude', 'district']
if not all(col in df_cameras.columns for col in required_cols):
    print("警告: 摄像头数据缺少必要列，当前列:", df_cameras.columns.tolist())
    raise ValueError("请检查 cameras_all.xlsx 列名")

# 筛选 district == '未知' 的点
unknown_cameras = df_cameras[df_cameras['district'] == '未知'].copy()

if unknown_cameras.empty:
    print("没有找到 district='未知' 的摄像头")
else:
    print(f"找到 {len(unknown_cameras)} 个 district='未知' 的摄像头")

# ==================== 2. 读取行政区边界 ====================
boundary_file = '../districts/chengdu_districts_boundary.csv'
if not os.path.exists(boundary_file):
    raise FileNotFoundError(f"未找到: {boundary_file}")

df_boundary = pd.read_csv(boundary_file)

# 解析 WKT 字符串为几何对象
df_boundary['geometry'] = df_boundary['区域边界'].apply(wkt.loads)

# 转为 GeoDataFrame
gdf_boundary = gpd.GeoDataFrame(df_boundary, geometry='geometry', crs="EPSG:4326")

# ==================== 3. 创建 Folium 地图 ====================
# 以成都中心为地图中心
m = folium.Map(location=[30.67, 104.06], zoom_start=11, tiles='CartoDB positron')

# 绘制每个行政区边界 + 标签
for _, row in gdf_boundary.iterrows():
    district_name = row['区域名称']

    # 转为 GeoJSON 格式用于 Folium
    folium.GeoJson(
        row['geometry'],
        name=district_name,
        style_function=lambda x, name=district_name: {
            'fillColor': 'lightblue' if '成华' in name else 'lightgray',
            'color': 'black',
            'weight': 2,
            'fillOpacity': 0.3,
        }
    ).add_to(m)

    # 添加区域名称标签（取多边形中心）
    centroid = row['geometry'].centroid
    folium.Marker(
        location=[centroid.y, centroid.x],
        icon=folium.DivIcon(html=f"""
            <div style="font-size: 12pt; color: black; font-weight: bold; 
                        text-shadow: 1px 1px 3px white; text-align: center;">
                {district_name}
            </div>
        """),
        tooltip=district_name
    ).add_to(m)

# ==================== 4. 绘制未知摄像头点 ====================
for _, cam in unknown_cameras.iterrows():
    folium.CircleMarker(
        location=[cam['latitude'], cam['longitude']],
        radius=6,
        color='red',
        fill=True,
        fill_color='darkred',
        fill_opacity=0.9,
        popup=folium.Popup(f"""
            <b>{cam['name']}</b><br>
            ID: {cam['camera_id']}<br>
            经纬度: {cam['latitude']}, {cam['longitude']}<br>
            区域: {cam['district']}
        """, max_width=300),
        tooltip=cam['name']
    ).add_to(m)

# 添加图层控制
folium.LayerControl().add_to(m)

# ==================== 5. 保存地图 ====================
output_html = 'unknown_cameras_map.html'
m.save(output_html)
print(f"\n地图已生成: {output_html}")
print("请用浏览器打开查看（双击文件或拖入浏览器）")