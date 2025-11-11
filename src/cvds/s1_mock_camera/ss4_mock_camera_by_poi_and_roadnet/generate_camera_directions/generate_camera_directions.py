import pandas as pd
import numpy as np
import folium
from folium import DivIcon
import os
import shapely.wkt
from shapely.geometry import Polygon, MultiPolygon

# ================== 配置路径 ==================
CAMERA_PATH = '../cameras_all.xlsx'
DISTRICT_PATH = '../districts/chengdu_districts_boundary.csv'
OUTPUT_HTML = 'chengdu_cameras_with_districts.html'

# 检查文件
for p in [CAMERA_PATH, DISTRICT_PATH]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"文件未找到: {p}")

# ================== 1. 读取相机数据 ==================
print("正在读取相机数据...")
df_cameras = pd.read_excel(CAMERA_PATH)

required_cols = ['latitude', 'longitude', 'name', 'camera_id']
for col in required_cols:
    if col not in df_cameras.columns:
        raise ValueError(f"相机数据缺少列: {col}")

# 生成随机朝向（若无 direction 列）
np.random.seed(42)
if 'direction' not in df_cameras.columns:
    mask = df_cameras.get('source_type', '').str.lower() == 'point'
    directions = np.random.uniform(0, 360, size=len(df_cameras))
    directions[~mask] = np.nan
    df_cameras['direction'] = directions
else:
    df_cameras['direction'] = pd.to_numeric(df_cameras['direction'], errors='coerce')

# ================== 2. 读取并解析区域边界 ==================
print("正在读取区域边界...")
df_districts = pd.read_csv(DISTRICT_PATH)

# 确保列名正确
df_districts.columns = df_districts.columns.str.strip()
if '区域名称' not in df_districts.columns or '区域边界' not in df_districts.columns:
    raise ValueError("CSV 必须包含列: '区域名称', '区域边界'")

# 解析 WKT 为 GeoJSON 格式
district_geojson = {
    "type": "FeatureCollection",
    "features": []
}

color_map = {
    '金牛区': '#1f77b4',
    '成华区': '#ff7f0e',
    '郫都区': '#2ca02c',
    '新都区': '#d62728',
    '青羊区': '#9467bd'
}

for _, row in df_districts.iterrows():
    name = row['区域名称'].strip()
    wkt_str = row['区域边界'].strip()

    try:
        geom = shapely.wkt.loads(wkt_str)
    except Exception as e:
        print(f"解析失败 [{name}]: {e}")
        continue


    # 转为 GeoJSON 坐标
    def extract_coords(g):
        if isinstance(g, Polygon):
            return [list(g.exterior.coords)]
        elif isinstance(g, MultiPolygon):
            return [list(poly.exterior.coords) for poly in g.geoms]
        return []


    coords = extract_coords(geom)
    if not coords:
        continue

    feature = {
        "type": "Feature",
        "properties": {"name": name},
        "geometry": {
            "type": "MultiPolygon" if isinstance(geom, MultiPolygon) else "Polygon",
            "coordinates": coords if isinstance(geom, MultiPolygon) else [coords[0]]
        }
    }
    district_geojson["features"].append(feature)

# ================== 3. 创建 Folium 地图 ==================
center_lat = df_cameras['latitude'].mean()
center_lon = df_cameras['longitude'].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles='CartoDB positron')

# 添加区域边界（带颜色）
for feature in district_geojson["features"]:
    name = feature["properties"]["name"]
    color = color_map.get(name, '#888888')

    folium.GeoJson(
        feature,
        style_function=lambda x, c=color: {
            'fillColor': c,
            'color': 'black',
            'weight': 2,
            'fillOpacity': 0.15
        },
        tooltip=name,
        popup=folium.Popup(f"<b>区域:</b> {name}", max_width=200)
    ).add_to(m)

# 添加图例
legend_html = '''
<div style="position: fixed; bottom: 50px; left: 50px; width: 150px; height: auto; 
            border:2px solid grey; z-index:9999; background-color:white; padding: 10px;
            font-size:14px; border-radius:6px;">
<b>行政区</b><br>
'''
for name, color in color_map.items():
    legend_html += f'<i style="background:{color};width:12px;height:12px;float:left;margin-top:3px;"></i>&nbsp;{name}<br>'
legend_html += '</div>'
m.get_root().html.add_child(folium.Element(legend_html))

# ================== 4. 添加相机点 + 朝向箭头 ==================
ARROW_LENGTH = 0.00035  # 约 35~40米

for _, row in df_cameras.iterrows():
    lat, lon = row['latitude'], row['longitude']
    cam_id = row['camera_id']
    name = row['name']
    district = row.get('district', '未知')
    direction = row['direction']

    # Popup
    popup_html = f"""
    <div style="font-size:12px;">
    <b>相机ID:</b> {cam_id}<br>
    <b>名称:</b> {name}<br>
    <b>区域:</b> {district}<br>
    """
    if pd.notna(direction):
        popup_html += f"<b>朝向:</b> {direction:.1f}°"
    else:
        popup_html += "<b>朝向:</b> 未设置"
    popup_html += "</div>"

    # 相机图标
    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(popup_html, max_width=300),
        tooltip=name,
        icon=folium.Icon(color='blue', icon='video-camera', prefix='fa')
    ).add_to(m)

    # 绘制朝向箭头
    if pd.notna(direction):
        rad = np.deg2rad(direction)
        end_lat = lat + ARROW_LENGTH * np.cos(rad)
        end_lon = lon + ARROW_LENGTH * np.sin(rad)

        folium.PolyLine(
            locations=[[lat, lon], [end_lat, end_lon]],
            color='red',
            weight=3,
            opacity=0.9
        ).add_to(m)

        # 箭头尖
        folium.Marker(
            location=[end_lat, end_lon],
            icon=DivIcon(
                icon_size=(12, 12),
                icon_anchor=(6, 6),
                html=f'''
                <div style="font-size: 18px; color: red; 
                            transform: rotate({direction}deg);
                            text-shadow: 0 0 3px white;">▶</div>
                '''
            )
        ).add_to(m)

# ================== 5. 保存地图 ==================
m.save(OUTPUT_HTML)
print(f"\n地图生成成功！")
print(f"   文件: {OUTPUT_HTML}")
print(f"   相机数量: {len(df_cameras):,}")
print(f"   区域数量: {len(district_geojson['features'])}")
print(f"   有朝向相机: {df_cameras['direction'].notna().sum():,}")