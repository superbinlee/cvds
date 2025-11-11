#!/usr/bin/env python
# -*- coding: utf-8__

"""
终极增强版：
- 悬停：u→v + 长度
- 点击：全部字段 + 坐标 + 长度
- 颜色：u→v 蓝，v→u 红
- 搜索框：camera_id / name
- 缓存：飞快
"""

import argparse
import logging
import os
import pickle

import folium
import geopandas as gpd
import networkx as nx
import pandas as pd
from folium.plugins import MarkerCluster, Search
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
from shapely.wkt import loads
from tqdm import tqdm

# ------------------- 配置 -------------------
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = ".."
DISTRICTS_CSV = os.path.join(BASE_DIR, "districts", "chengdu_districts_boundary.csv")
CAMERAS_XLSX = os.path.join(BASE_DIR, "cameras_all.xlsx")
ROAD_GPKG = os.path.join(BASE_DIR, "roadnet", "Chengdu_all_road_network.gpkg")

OUTPUT_XLSX = os.path.join(BASE_DIR, "cameras_all_with_directions.xlsx")
OUTPUT_HTML = "camera_map_ultra.html"

CACHE_DIR = ".cache"
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_ROADNET = os.path.join(CACHE_DIR, "roadnet_cache.pkl")
CACHE_DIRECTION = os.path.join(CACHE_DIR, "direction_ultra_cache.csv")

WGS84_CRS = "EPSG:4326"
PROJECTED_CRS = "EPSG:32648"

parser = argparse.ArgumentParser()
parser.add_argument("--no-cache", action="store_true")
args = parser.parse_args()

# ------------------- 加载路网 -------------------
def load_road_network_cached():
    if not args.no_cache and os.path.exists(CACHE_ROADNET):
        logger.info("加载路网缓存...")
        with open(CACHE_ROADNET, 'rb') as f:
            return pickle.load(f)

    logger.info("加载路网...")
    nodes = gpd.read_file(ROAD_GPKG, layer="nodes")
    edges = gpd.read_file(ROAD_GPKG, layer="edges")
    source_col = "u" if "u" in edges.columns else "source"
    target_col = "v" if "v" in edges.columns else "target"
    G = nx.from_pandas_edgelist(edges, source=source_col, target=target_col, create_using=nx.MultiGraph())
    node_id_col = "osmid" if "osmid" in nodes.columns else "id"
    for _, row in nodes.iterrows():
        nid = row[node_id_col]
        if nid in G.nodes:
            G.nodes[nid]["x"] = row["x"]
            G.nodes[nid]["y"] = row["y"]

    edges_geom = []
    for _, row in edges.iterrows():
        u = row[source_col]
        v = row[target_col]
        key = row.get("key", 0)
        pt1 = Point(G.nodes[u]["x"], G.nodes[u]["y"])
        pt2 = Point(G.nodes[v]["x"], G.nodes[v]["y"])
        edges_geom.append({"u": u, "v": v, "key": key, "geometry": LineString([pt1, pt2])})
    edges_gdf_proj = gpd.GeoDataFrame(edges_geom, crs=WGS84_CRS).to_crs(PROJECTED_CRS)

    with open(CACHE_ROADNET, 'wb') as f:
        pickle.dump((G, edges_gdf_proj), f)
    logger.info(f"路网已缓存")
    return G, edges_gdf_proj

G, edges_gdf_proj = load_road_network_cached()

# ------------------- 相机点位 -------------------
logger.info("加载相机点位...")
cameras_df = pd.read_excel(CAMERAS_XLSX)
cameras_gdf = gpd.GeoDataFrame(
    cameras_df,
    geometry=gpd.points_from_xy(cameras_df.longitude, cameras_df.latitude),
    crs=WGS84_CRS
)
cameras_gdf_proj = cameras_gdf.to_crs(PROJECTED_CRS)

# ------------------- 节点对方向 -------------------
def compute_node_directions():
    if not args.no_cache and os.path.exists(CACHE_DIRECTION):
        logger.info("加载方向缓存...")
        return pd.read_csv(CACHE_DIRECTION)

    logger.info("计算节点对方向...")
    MAX_DISTANCE = 300
    records = []

    for idx, cam in tqdm(cameras_gdf_proj.iterrows(), total=len(cameras_gdf_proj), desc="处理相机"):
        pt = cam.geometry
        distances = edges_gdf_proj.distance(pt)
        if distances.empty or distances.min() > MAX_DISTANCE:
            records.append({"camera_id": cam["camera_id"], "u_node": None, "v_node": None, "edge_length": None})
            continue

        edge = edges_gdf_proj.loc[distances.idxmin()]
        line = edge.geometry
        u, v = edge["u"], edge["v"]
        length = line.length

        nearest_pt = nearest_points(pt, line)[1]
        dist_u = nearest_pt.distance(Point(G.nodes[u]["x"], G.nodes[u]["y"]))
        dist_v = nearest_pt.distance(Point(G.nodes[v]["x"], G.nodes[v]["y"]))

        if dist_u < dist_v:
            records.append({"camera_id": cam["camera_id"], "u_node": u, "v_node": v, "edge_length": round(length, 2)})
        else:
            records.append({"camera_id": cam["camera_id"], "u_node": v, "v_node": u, "edge_length": round(length, 2)})

    df = pd.DataFrame(records)
    df.to_csv(CACHE_DIRECTION, index=False)
    logger.info(f"方向已缓存")
    return df

direction_df = compute_node_directions()
cameras_with_dir = cameras_gdf.merge(direction_df, on="camera_id", how="left")

# ------------------- 保存 Excel -------------------
output_cols = list(cameras_df.columns) + ["u_node", "v_node", "edge_length"]
cameras_output = cameras_with_dir[output_cols].reindex(columns=output_cols)
cameras_output.to_excel(OUTPUT_XLSX, index=False)
logger.info(f"已保存 → {OUTPUT_XLSX}")

# ------------------- 行政区 -------------------
districts_df = pd.read_csv(DISTRICTS_CSV)
def fix_wkt(s):
    s = str(s).strip()
    open_cnt = s.count('(')
    close_cnt = s.count(')')
    if open_cnt > close_cnt: s += ')' * (open_cnt - close_cnt)
    return s
districts_df['区域边界'] = districts_df['区域边界'].apply(fix_wkt)
districts_gdf = gpd.GeoDataFrame(
    districts_df.drop(columns=['区域边界'], errors='ignore'),
    geometry=districts_df['区域边界'].apply(lambda x: loads(x) if pd.notna(x) else None),
    crs=WGS84_CRS
).dropna(subset=['geometry'])

# ------------------- 地图 -------------------
logger.info("生成增强地图...")
m = folium.Map(location=[30.662, 104.065], zoom_start=11, tiles="CartoDB positron")

folium.GeoJson(
    districts_gdf.rename(columns={"区域名称": "name"})[["name", "geometry"]].to_json(),
    name="行政区",
    style_function=lambda x: {"fillColor": "transparent", "color": "black", "weight": 2, "dashArray": "5, 5"},
    tooltip=folium.GeoJsonTooltip(fields=["name"], aliases=["区域："])
).add_to(m)

marker_cluster = MarkerCluster(name="相机点位").add_to(m)

valid_cameras = cameras_with_dir.dropna(subset=['geometry', 'u_node', 'v_node']).to_crs(WGS84_CRS)

for _, row in valid_cameras.iterrows():
    lat, lon = row.geometry.y, row.geometry.x
    u, v = int(row.u_node), int(row.v_node)
    length = row.edge_length
    ux, uy = G.nodes[u]["x"], G.nodes[u]["y"]
    vx, vy = G.nodes[v]["x"], G.nodes[v]["y"]

    # 颜色：u→v 蓝，v→u 红
    color = "blue" if row.u_node == u else "red"

    popup_html = f"""
    <div style="font-family: Arial; font-size: 13px; width: 280px;">
        <b style="font-size: 15px;">{row.get('name', '未知')}</b><br>
        <b>ID:</b> {row['camera_id']}<br>
        <b>区域:</b> {row.get('district', '未知')}<br>
        <b>来源:</b> {row.get('source_type', '')} / {row.get('source', '')}<br>
        <hr style="margin: 8px 0;">
        <b>方向:</b> <span style="color: {color};">{u} → {v}</span><br>
        <b>长度:</b> {length} 米<br>
        <b>节点坐标:</b><br>
        &nbsp;&nbsp;u: ({ux:.6f}, {uy:.6f})<br>
        &nbsp;&nbsp;v: ({vx:.6f}, {vy:.6f})
    </div>
    """

    icon = folium.Icon(color=color, icon="arrow-right", prefix="fa")
    marker = folium.Marker(
        [lat, lon],
        popup=folium.Popup(popup_html, max_width=320),
        icon=icon,
        tooltip=f"<b>{u}→{v}</b><br>{length} 米"
    )
    marker.add_to(marker_cluster)

# 搜索框
search = Search(
    layer=marker_cluster,
    geom_type='Point',
    placeholder='搜索 camera_id 或 name',
    collapsed=False,
    search_label='name',
    weight=3
).add_to(m)

folium.LayerControl().add_to(m)
m.save(OUTPUT_HTML)
logger.info(f"增强地图已保存 → {OUTPUT_HTML}")

# ------------------- 完成 -------------------
print("\n" + "="*60)
print("任务完成！")
print(f"1. 完整数据: {os.path.abspath(OUTPUT_XLSX)}")
print(f"2. 增强地图: {os.path.abspath(OUTPUT_HTML)}")
print("   悬停：方向 + 长度")
print("   点击：全部字段 + 坐标 + 长度")
print("   搜索：camera_id / name")
if not args.no_cache:
    print(f"   缓存: {os.path.abspath(CACHE_DIR)}")
print("="*60)