# visualize_csv_with_folium.py
import pandas as pd
import geopandas as gpd
from shapely import wkt
import folium
from pathlib import Path
import random
import sys

# ==================== 路径 ====================
CSV_PATH = Path("./chengdu_districts_boundary.csv")
HTML_PATH = Path("./chengdu_districts_map.html")
HTML_PATH.parent.mkdir(parents=True, exist_ok=True)

if not CSV_PATH.exists():
    sys.exit(f"CSV 文件不存在: {CSV_PATH}")

# ==================== 读取 + 解析 WKT ====================
print("正在读取 CSV ...")
df = pd.read_csv(CSV_PATH, encoding="utf-8")
print(f"读取到 {len(df)} 行")

valid_rows = []
for idx, row in df.iterrows():
    name = str(row["区域名称"]).strip()
    wkt_str = str(row["区域边界"]).strip()

    if not wkt_str.upper().startswith(("POLYGON", "MULTIPOLYGON")):
        print(f"跳过 {name}: WKT 无效")
        continue

    try:
        geom = wkt.loads(wkt_str)
        if geom.is_empty: continue
        pts = len(geom.exterior.coords) if geom.geom_type == "Polygon" else sum(len(p.exterior.coords) for p in geom.geoms)
        valid_rows.append({"区域名称": name, "geometry": geom, "点数": pts})
        print(f"成功 {name}: {pts} 点")
    except Exception as e:
        print(f"失败 {name}: {e}")

if not valid_rows:
    sys.exit("没有有效数据！")

gdf = gpd.GeoDataFrame(valid_rows, geometry="geometry", crs="EPSG:4326")
print(f"\n成功加载 {len(gdf)} 个行政区")

# ==================== 计算地图中心（投影后）===================
gdf_utm = gdf.to_crs("EPSG:32648")
center_utm = gdf_utm.geometry.unary_union.centroid
center_wgs84 = gdf_utm.to_crs("EPSG:4326").geometry.unary_union.centroid
center_lat, center_lon = center_wgs84.y, center_wgs84.x

# ==================== 创建地图 ====================
m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB positron")


# 柔和随机颜色
def rand_color():
    return "#%06x" % random.randint(0x88CCFF, 0xFFCC88)


# ==================== 逐个绘制每个区（独立！）===================
for idx, row in gdf.iterrows():
    # 1. 转为 GeoJSON（只包含当前行）
    feature = {
        "type": "Feature",
        "properties": {"name": row["区域名称"]},
        "geometry": row.geometry.__geo_interface__
    }

    # 2. 独立上色 + 黑色边框
    color = rand_color()
    folium.GeoJson(
        feature,
        style_function=lambda x, c=color: {
            "fillColor": c,
            "color": "black",
            "weight": 2.5,
            "fillOpacity": 0.65,
        }
    ).add_to(m)

    # 3. 弹窗
    folium.Popup(
        f"<b style='font-size:15px'>{row['区域名称']}</b><br>"
        f"<small>边界点数: {row['点数']}</small>",
        max_width=300
    ).add_to(m)

    # 4. 中心标注区名（清晰、不重叠）
    centroid = row.geometry.centroid
    folium.Marker(
        location=[centroid.y, centroid.x],
        icon=folium.DivIcon(html=f"""
            <div style="
                font-size: 14pt;
                font-weight: bold;
                color: #1a1a1a;
                background: rgba(255,255,255,0.9);
                padding: 4px 8px;
                border-radius: 6px;
                border: 2px solid {color};
                white-space: nowrap;
                box-shadow: 0 1px 4px rgba(0,0,0,0.3);
            ">{row['区域名称']}</div>
        """),
        tooltip=row['区域名称']
    ).add_to(m)

# ==================== 底图切换 ====================
folium.TileLayer("OpenStreetMap").add_to(m)
folium.TileLayer("CartoDB dark_matter").add_to(m)
folium.LayerControl().add_to(m)

# ==================== 标题 ====================
title_html = f'''
<h3 align="center" style="margin:15px; font-weight:bold; color:#2c3e50; font-size:18px;">
    成都市行政区（{len(gdf)} 个区）<br>
    <small style="color:#7f8c8d;">每个区独立绘制 · 清晰标注</small>
</h3>
'''
m.get_root().html.add_child(folium.Element(title_html))

# ==================== 保存 ====================
print(f"\n正在保存 → {HTML_PATH}")
m.save(str(HTML_PATH))

import webbrowser, os

webbrowser.open("file://" + os.path.abspath(HTML_PATH))
print("地图已生成！浏览器已打开")
