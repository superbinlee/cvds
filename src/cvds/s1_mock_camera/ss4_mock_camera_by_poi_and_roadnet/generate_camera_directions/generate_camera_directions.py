#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生成相机方向（对齐到最近道路） + Folium 可视化
输出：
1. ../cameras_all_with_directions.xlsx   （新增 direction_angle, direction）
2. camera_map_with_directions.html       （行政区 + 相机 + 方向箭头）
"""

import os
import pandas as pd
import geopandas as gpd
import numpy as np
import networkx as nx
import folium
from shapely.geometry import Point, LineString
from shapely.wkt import loads
from tqdm import tqdm
import logging
from math import atan2, degrees

# ------------------- 配置 -------------------
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# 路径设置（请确保路径正确）
BASE_DIR = ".."  # 例如：/Users/leebin/PycharmProjects/cvds
DISTRICTS_CSV = os.path.join(BASE_DIR, "districts", "chengdu_districts_boundary.csv")
CAMERAS_XLSX = os.path.join(BASE_DIR, "cameras_all.xlsx")
ROAD_GPKG = os.path.join(BASE_DIR, "roadnet", "Chengdu_all_road_network.gpkg")

OUTPUT_XLSX = os.path.join(BASE_DIR, "cameras_all_with_directions.xlsx")
OUTPUT_HTML = "camera_map_with_directions.html"

# 投影坐标系（成都 → UTM 48N，单位：米）
PROJECTED_CRS = "EPSG:32648"
WGS84_CRS = "EPSG:4326"


# ------------------- 加载路网 -------------------
def load_road_network(gpkg_path):
    logger.info("加载路网...")
    nodes = gpd.read_file(gpkg_path, layer="nodes")
    edges = gpd.read_file(gpkg_path, layer="edges")

    source_col = "u" if "u" in edges.columns else "source"
    target_col = "v" if "v" in edges.columns else "target"
    logger.info(f"构建 NetworkX 图 (u={source_col}, v={target_col})...")
    G = nx.from_pandas_edgelist(edges, source=source_col, target=target_col, create_using=nx.MultiGraph())

    node_id_col = "osmid" if "osmid" in nodes.columns else "id"
    logger.info("注入坐标...")
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

# ------------------- 读取行政区边界（修复 WKT） -------------------
logger.info("加载行政区边界...")
districts_df = pd.read_csv(DISTRICTS_CSV)


def fix_incomplete_wkt(wkt_str):
    wkt_str = str(wkt_str).strip()
    open_cnt = wkt_str.count('(')
    close_cnt = wkt_str.count(')')
    if open_cnt > close_cnt:
        wkt_str += ')' * (open_cnt - close_cnt)
    return wkt_str


districts_df['区域边界'] = districts_df['区域边界'].apply(fix_incomplete_wkt)


def safe_load_wkt(wkt_str):
    try:
        return loads(wkt_str)
    except Exception as e:
        logger.warning(f"WKT 解析失败（跳过）: {wkt_str[:60]}... -> {e}")
        return None


geometry_series = districts_df['区域边界'].apply(safe_load_wkt)
districts_gdf = gpd.GeoDataFrame(
    districts_df.drop(columns=['区域边界']),
    geometry=geometry_series,
    crs=WGS84_CRS
).dropna(subset=['geometry'])

logger.info(f"行政区边界加载完成: {len(districts_gdf)} 个有效区域")

# ------------------- 投影到平面坐标系（用于距离计算） -------------------
logger.info(f"投影到 {PROJECTED_CRS} 以进行米级距离计算...")
cameras_gdf_proj = cameras_gdf.to_crs(PROJECTED_CRS)
districts_gdf_proj = districts_gdf.to_crs(PROJECTED_CRS)

# 构建带 geometry 的 edges（投影后）
logger.info("构建投影后的道路边 GeoDataFrame...")
edges_geom = []
for _, row in edges_gdf.iterrows():
    u = row["u"] if "u" in row else row["source"]
    v = row["v" if "v" in row else "target"]
    key = row.get("key", 0)
    pt1 = Point(G.nodes[u]["x"], G.nodes[u]["y"])
    pt2 = Point(G.nodes[v]["x"], G.nodes[v]["y"])
    edges_geom.append({"u": u, "v": v, "key": key, "geometry": LineString([pt1, pt2])})

edges_with_geom = gpd.GeoDataFrame(edges_geom, crs=WGS84_CRS)
edges_with_geom_proj = edges_with_geom.to_crs(PROJECTED_CRS)


# ------------------- 计算道路方向（在原始坐标系） -------------------
def get_edge_direction(u, v):
    x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
    x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
    angle = degrees(atan2(y2 - y1, x2 - x1))
    return (angle + 360) % 360


def angle_to_cardinal(angle):
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = round(angle / 45) % 8
    return dirs[idx]


logger.info("预计算所有道路边方向...")
edge_directions = {}
for u, v, key in tqdm(G.edges(keys=True), desc="计算边方向", leave=False):
    edge_directions[(u, v, key)] = get_edge_direction(u, v)

# ------------------- 为每个相机分配方向（投影坐标系） -------------------
logger.info("为相机分配最近道路方向（米级精度）...")
MAX_DISTANCE = 300  # 米，超过此距离视为无路
direction_records = []

for idx, cam in tqdm(cameras_gdf_proj.iterrows(), total=len(cameras_gdf_proj), desc="处理相机"):
    pt_proj = cam.geometry
    distances = edges_with_geom_proj.distance(pt_proj)

    if distances.empty or distances.min() > MAX_DISTANCE:
        direction_str = "NO_NEAR_ROAD"
        angle_val = None
    else:
        nearest_idx = distances.idxmin()
        edge = edges_with_geom_proj.loc[nearest_idx]
        u, v, key = edge["u"], edge["v"], edge["key"]

        angle = edge_directions.get((u, v, key))
        if angle is None:
            rev_angle = edge_directions.get((v, u, key))
            if rev_angle is not None:
                angle = (rev_angle + 180) % 360

        if angle is None:
            direction_str = "UNKNOWN"
            angle_val = None
        else:
            direction_str = angle_to_cardinal(angle)
            angle_val = round(angle, 2)

    direction_records.append({
        "camera_id": cam["camera_id"],
        "direction_angle": angle_val,
        "direction": direction_str
    })

direction_df = pd.DataFrame(direction_records)
cameras_with_dir = cameras_gdf.merge(direction_df, on="camera_id", how="left")

# ------------------- 保存 Excel -------------------
output_cols = list(cameras_df.columns) + ["direction_angle", "direction"]
cameras_output = cameras_with_dir[output_cols].copy()
cameras_output = cameras_output.reindex(columns=output_cols)
cameras_output.to_excel(OUTPUT_XLSX, index=False)
logger.success(f"已保存带方向的相机点位 → {OUTPUT_XLSX}")

# ------------------- Folium 地图（转回 WGS84） -------------------
logger.info("生成 Folium 交互地图...")
m = folium.Map(location=[30.662, 104.065], zoom_start=11, tiles="CartoDB positron")

# 行政区边界（WGS84）
folium.GeoJson(
    districts_gdf.rename(columns={"区域名称": "name"})[["name", "geometry"]].to_json(),
    name="行政区",
    style_function=lambda x: {
        "fillColor": "transparent",
        "color": "black",
        "weight": 2,
        "dashArray": "5, 5"
    },
    tooltip=folium.GeoJsonTooltip(fields=["name"], aliases=["区域："])
).add_to(m)

# 相机点 + 方向箭头（WGS84）
cameras_wgs84 = cameras_with_dir.set_geometry(cameras_gdf.geometry).to_crs(WGS84_CRS)


def get_arrow_icon(direction):
    if pd.isna(direction) or direction in ["NO_NEAR_ROAD", "UNKNOWN"]:
        return folium.Icon(color="gray", icon="camera")
    dir_to_angle = {"N": 0, "NE": 45, "E": 90, "SE": 135, "S": 180, "SW": 225, "W": 270, "NW": 315}
    angle = dir_to_angle.get(direction, 0)
    colors = ["blue", "cadetblue", "red", "orange", "green", "purple", "pink", "darkblue"]
    color = colors[list(dir_to_angle.values()).index(angle) % len(colors)]
    return folium.Icon(color=color, icon="arrow-up", prefix="fa", angle=angle)


for _, row in cameras_wgs84.iterrows():
    lat, lon = row.geometry.y, row.geometry.x
    popup_html = f"""
    <b>{row.get('name', '未知')}</b><br>
    ID: {row['camera_id']}<br>
    区域: {row.get('district', '未知')}<br>
    方向: <strong>{row['direction']}</strong>
    {f" ({row['direction_angle']}°)" if pd.notna(row['direction_angle']) else ""}
    """
    # 圆点
    folium.CircleMarker(
        location=[lat, lon],
        radius=6,
        color="black",
        weight=1,
        fill=True,
        fillColor="yellow",
        fillOpacity=0.9,
        popup=folium.Popup(popup_html, max_width=300)
    ).add_to(m)

    # 方向箭头
    folium.Marker(
        location=[lat, lon],
        icon=get_arrow_icon(row['direction']),
        tooltip=f"{row['direction']} {row['direction_angle'] or ''}°"
    ).add_to(m)

folium.LayerControl().add_to(m)
m.save(OUTPUT_HTML)
logger.success(f"交互地图已保存 → {OUTPUT_HTML}")

# ------------------- 完成 -------------------
print("\n" + "=" * 60)
print("任务全部完成！")
print(f"1. 带方向相机点位: {os.path.abspath(OUTPUT_XLSX)}")
print(f"2. 可视化地图文件: {os.path.abspath(OUTPUT_HTML)}")
print("=" * 60)