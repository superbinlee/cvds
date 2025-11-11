#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
聚合展示 + 点击显示方向 + 8方向不同颜色 + 原始数据完整
输出：
1. ../cameras_all_with_directions.xlsx
2. camera_map_final.html
"""

import os
import pandas as pd
import geopandas as gpd
import networkx as nx
import folium
from folium.plugins import MarkerCluster
from shapely.geometry import Point, LineString
from shapely.wkt import loads
from tqdm import tqdm
import logging
from math import atan2, degrees

# ------------------- 配置 -------------------
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = ".."
DISTRICTS_CSV = os.path.join(BASE_DIR, "districts", "chengdu_districts_boundary.csv")
CAMERAS_XLSX = os.path.join(BASE_DIR, "cameras_all.xlsx")
ROAD_GPKG = os.path.join(BASE_DIR, "roadnet", "Chengdu_all_road_network.gpkg")

OUTPUT_XLSX = os.path.join(BASE_DIR, "cameras_all_with_directions.xlsx")
OUTPUT_HTML = "camera_map_final.html"

WGS84_CRS = "EPSG:4326"
PROJECTED_CRS = "EPSG:32648"

# 8个方向 → 8种颜色
DIRECTION_COLORS = {
    "N": "#1f77b4",  # 蓝色
    "NE": "#ff7f0e",  # 橙色
    "E": "#2ca02c",  # 绿色
    "SE": "#d62728",  # 红色
    "S": "#9467bd",  # 紫色
    "SW": "#8c564b",  # 棕色
    "W": "#e377c2",  # 粉色
    "NW": "#7f7f7f",  # 灰色
    "NO_NEAR_ROAD": "#000000",  # 黑色
    "UNKNOWN": "#cccccc"  # 浅灰
}


# ------------------- 加载路网 -------------------
def load_road_network(gpkg_path):
    logger.info("加载路网...")
    nodes = gpd.read_file(gpkg_path, layer="nodes")
    edges = gpd.read_file(gpkg_path, layer="edges")
    source_col = "u" if "u" in edges.columns else "source"
    target_col = "v" if "v" in edges.columns else "target"
    G = nx.from_pandas_edgelist(edges, source=source_col, target=target_col, create_using=nx.MultiGraph())
    node_id_col = "osmid" if "osmid" in nodes.columns else "id"
    for _, row in tqdm(nodes.iterrows(), total=len(nodes), desc="注入坐标", leave=False):
        nid = row[node_id_col]
        if nid in G.nodes:
            G.nodes[nid]["x"] = row["x"]
            G.nodes[nid]["y"] = row["y"]
    logger.info(f"路网加载完成: {G.number_of_nodes():,} 节点, {G.number_of_edges():,} 边")
    return G, nodes, edges


G, nodes_gdf, edges_gdf = load_road_network(ROAD_GPKG)

# ------------------- 读取相机点位 -------------------
logger.info("加载相机点位...")
cameras_df = pd.read_excel(CAMERAS_XLSX)
cameras_gdf = gpd.GeoDataFrame(
    cameras_df,
    geometry=gpd.points_from_xy(cameras_df.longitude, cameras_df.latitude),
    crs=WGS84_CRS
)
logger.info(f"相机点位加载完成: {len(cameras_gdf)} 个")

# ------------------- 读取行政区 -------------------
districts_df = pd.read_csv(DISTRICTS_CSV)


def fix_wkt(wkt_str):
    wkt_str = str(wkt_str).strip()
    open_cnt = wkt_str.count('(')
    close_cnt = wkt_str.count(')')
    if open_cnt > close_cnt:
        wkt_str += ')' * (open_cnt - close_cnt)
    return wkt_str


districts_df['区域边界'] = districts_df['区域边界'].apply(fix_wkt)
geometry_series = districts_df['区域边界'].apply(lambda x: loads(x) if pd.notna(x) else None)
districts_gdf = gpd.GeoDataFrame(districts_df.drop(columns=['区域边界'], errors='ignore'), geometry=geometry_series, crs=WGS84_CRS).dropna(subset=['geometry'])

# ------------------- 投影 + 路网构建 -------------------
cameras_gdf_proj = cameras_gdf.to_crs(PROJECTED_CRS)
edges_geom = []
for _, row in edges_gdf.iterrows():
    u = row["u"] if "u" in row else row["source"]
    v = row["v"] if "v" in row else row["target"]
    key = row.get("key", 0)
    pt1 = Point(G.nodes[u]["x"], G.nodes[u]["y"])
    pt2 = Point(G.nodes[v]["x"], G.nodes[v]["y"])
    edges_geom.append({"u": u, "v": v, "key": key, "geometry": LineString([pt1, pt2])})
edges_with_geom_proj = gpd.GeoDataFrame(edges_geom, crs=WGS84_CRS).to_crs(PROJECTED_CRS)


# ------------------- 方向计算 -------------------
def get_edge_direction(u, v):
    x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
    x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
    return (degrees(atan2(y2 - y1, x2 - x1)) + 360) % 360


def angle_to_cardinal(angle):
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return dirs[round(angle / 45) % 8]


edge_directions = {}
for u, v, key in G.edges(keys=True):
    edge_directions[(u, v, key)] = get_edge_direction(u, v)

# ------------------- 分配方向 -------------------
logger.info("分配相机方向...")
MAX_DISTANCE = 300
direction_records = []

for idx, cam in tqdm(cameras_gdf_proj.iterrows(), total=len(cameras_gdf_proj), desc="处理相机"):
    pt = cam.geometry
    distances = edges_with_geom_proj.distance(pt)
    if distances.empty or distances.min() > MAX_DISTANCE:
        direction_str = "NO_NEAR_ROAD"
        angle_val = None
    else:
        edge = edges_with_geom_proj.loc[distances.idxmin()]
        u, v, key = edge["u"], edge["v"], edge["key"]
        angle = edge_directions.get((u, v, key))
        if angle is None:
            rev = edge_directions.get((v, u, key))
            if rev is not None: angle = (rev + 180) % 360
        direction_str = angle_to_cardinal(angle) if angle else "UNKNOWN"
        angle_val = round(angle, 2) if angle else None
    direction_records.append({
        "camera_id": cam["camera_id"],
        "direction_angle": angle_val,
        "direction": direction_str
    })

direction_df = pd.DataFrame(direction_records)
cameras_with_dir = cameras_gdf.merge(direction_df, on="camera_id", how="left")

# ------------------- 保存 Excel -------------------
output_cols = list(cameras_df.columns) + ["direction_angle", "direction"]
cameras_output = cameras_with_dir[output_cols].reindex(columns=output_cols)
cameras_output.to_excel(OUTPUT_XLSX, index=False)
logger.info(f"已保存完整数据 → {OUTPUT_XLSX}")

# ------------------- 聚合地图（点击显示方向 + 颜色区分） -------------------
logger.info("生成聚合地图（点击显示方向 + 8色区分）...")
m = folium.Map(location=[30.662, 104.065], zoom_start=11, tiles="CartoDB positron")

# 行政区边界
folium.GeoJson(
    districts_gdf.rename(columns={"区域名称": "name"})[["name", "geometry"]].to_json(),
    name="行政区",
    style_function=lambda x: {"fillColor": "transparent", "color": "black", "weight": 2, "dashArray": "5, 5"},
    tooltip=folium.GeoJsonTooltip(fields=["name"], aliases=["区域："])
).add_to(m)

# MarkerCluster
marker_cluster = MarkerCluster(name="相机点位（点击展开）").add_to(m)


# 自定义 SVG 箭头（带方向 + 颜色）
def create_svg_icon(direction):
    color = DIRECTION_COLORS.get(direction, "#cccccc")
    if direction in ["NO_NEAR_ROAD", "UNKNOWN"]:
        return folium.DivIcon(html=f"""
            <div style="font-size: 14px; color: {color}; text-align: center;">Cam</div>
        """)

    dir_to_rot = {"N": 0, "NE": 45, "E": 90, "SE": 135, "S": 180, "SW": 225, "W": 270, "NW": 315}
    rot = dir_to_rot.get(direction, 0)
    return folium.DivIcon(html=f"""
        <div style="transform: rotate({rot}deg);">
            <svg width="20" height="20" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2L2 22L12 17L22 22L12 2Z" fill="{color}" stroke="black" stroke-width="1.5"/>
            </svg>
        </div>
    """)


# 添加所有点（原始数量）
valid_cameras = cameras_with_dir.dropna(subset=['geometry']).to_crs(WGS84_CRS)

for _, row in tqdm(valid_cameras.iterrows(), total=len(valid_cameras), desc="添加相机点"):
    lat, lon = row.geometry.y, row.geometry.x
    direction = row['direction']
    angle = row['direction_angle']

    popup_html = f"""
    <div style="font-size: 14px; font-family: Arial;">
        <b>{row.get('name', '未知')}</b><br>
        ID: {row['camera_id']}<br>
        区域: {row.get('district', '未知')}<br>
        <b>方向: {direction}</b>
        {f"<br>角度: {angle}°" if pd.notna(angle) else ""}
    </div>
    """

    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(popup_html, max_width=300),
        icon=create_svg_icon(direction),
        tooltip=direction  # 悬停显示方向
    ).add_to(marker_cluster)

folium.LayerControl().add_to(m)
m.save(OUTPUT_HTML)
logger.info(f"聚合地图已保存 → {OUTPUT_HTML}")

# ------------------- 完成 -------------------
print("\n" + "=" * 60)
print("任务完成！")
print(f"1. 完整数据: {os.path.abspath(OUTPUT_XLSX)}")
print(f"2. 聚合地图: {os.path.abspath(OUTPUT_HTML)}")
print("   - 远看：聚合点")
print("   - 近看：彩色箭头（8方向8色）")
print("   - 悬停：显示方向")
print("   - 点击：显示详细信息")
print("=" * 60)