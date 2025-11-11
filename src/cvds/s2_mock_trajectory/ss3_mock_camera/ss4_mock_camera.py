# -*- coding: utf-8 -*-
"""
成都智能相机部署系统（网格化 10,000 个相机 · 语法修复）
"""

import warnings
from pathlib import Path
import pickle
import math
import random

import folium
import geopandas as gpd
import networkx as nx
import pandas as pd
from folium.plugins import FastMarkerCluster
from loguru import logger
from tqdm import tqdm

warnings.filterwarnings("ignore")
random.seed(42)

# ==================== 路径配置 ====================
ROAD_GPKG = Path("road") / "Chengdu_all_road_network.gpkg"
DISTRICTS_CSV = Path("districts") / "chengdu_districts_boundary.csv"
OUTPUT_DIR = Path("output") / "camera"
CACHE_FILE = OUTPUT_DIR / "candidates_cache.pkl"
OUTPUT_EXCEL = OUTPUT_DIR / "chengdu_10000_cameras.xlsx"
OUTPUT_HTML = OUTPUT_DIR / "chengdu_10000_cameras_map.html"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for p in [ROAD_GPKG, DISTRICTS_CSV]:
    if not p.exists():
        raise FileNotFoundError(f"文件未找到: {p}")

# ==================== 参数配置 ====================
MIN_DEGREE = 2
GRID_SIZE_M = 150
TARGET_CAMERAS = 10000
INCLUDE_TERTIARY = True

DISTRICT_COLORS = {
    "金牛区": "#FF6B6B", "青羊区": "#4ECDC4", "成华区": "#45B7D1",
    "武侯区": "#96CEB4", "锦江区": "#FECA57", "高新区": "#DDA0DD",
    "天府新区": "#98D8C8", "其他区": "#95A5A6"
}

HIGHWAY_RANK = {
    'motorway': 6, 'trunk': 6, 'primary': 5, 'secondary': 4,
    'tertiary': 3, 'unclassified': 2, 'residential': 1, 'service': 0
}

# ==================== 日志配置 ====================
logger.remove()
logger.add(
    lambda msg: print(msg, end=""),
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{message}</cyan>",
    colorize=True
)


# ==================== 1. 加载路网 ====================
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

    logger.success(f"路网加载完成: {G.number_of_nodes():,} 节点, {G.number_of_edges():,} 边")
    return G, nodes, edges


# ==================== 2. 加载行政区 ====================
def load_districts(csv_path):
    logger.info(f"加载行政区: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.GeoSeries.from_wkt(df["区域边界"]),
        crs="EPSG:4326"
    ).rename(columns={'区域名称': 'district'})
    gdf = gdf[['district', 'geometry']].dropna()
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    gdf = gdf.dissolve(by='district').reset_index()
    logger.success(f"行政区加载完成: {len(gdf)} 个")
    return gdf


# ==================== 3. 扫描 + 缓存候选点 ====================
def scan_and_cache_candidates(G, edges_gdf):
    if CACHE_FILE.exists():
        logger.info(f"检测到缓存文件: {CACHE_FILE}")
        with open(CACHE_FILE, 'rb') as f:
            candidates = pickle.load(f)
        logger.success(f"缓存加载完成: {len(candidates)} 个候选点")
        return candidates

    logger.info("未找到缓存，开始扫描路口...")
    edges = edges_gdf.copy()
    edges['highway'] = edges['highway'].fillna('residential')
    edges['is_roundabout'] = edges['junction'].isin(['roundabout', 'mini_roundabout'])

    degrees = dict(G.degree())
    candidates = []

    for node in tqdm(G.nodes(), desc="扫描路口", leave=False):
        x = G.nodes[node].get("x")
        y = G.nodes[node].get("y")
        if x is None or y is None: continue

        deg = degrees[node]
        if deg < MIN_DEGREE: continue

        neighbor_edges = edges[(edges['u'] == node) | (edges['v'] == node)]
        if neighbor_edges.empty: continue

        highway_list = [item.strip() for h in neighbor_edges['highway'] for item in str(h).split(';') if item.strip()]
        if not highway_list: continue
        highway_type = max(highway_list, key=lambda x: HIGHWAY_RANK.get(x, 0))
        max_rank = HIGHWAY_RANK.get(highway_type, 0)

        if max_rank < 3: continue
        if max_rank == 3 and not INCLUDE_TERTIARY: continue

        is_roundabout = neighbor_edges['is_roundabout'].any()

        candidates.append({
            'osmid': node, 'x': x, 'y': y,
            'degree': deg, 'max_rank': max_rank,
            'highway': highway_type, 'is_roundabout': is_roundabout
        })

    logger.success(f"扫描完成: {len(candidates)} 个候选点")
    logger.info(f"保存缓存到: {CACHE_FILE}")
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(candidates, f)
    logger.success("缓存保存成功！")
    return candidates


# ==================== 4. 网格化部署 10,000 个相机 ====================
def deploy_key_intersections(G, edges_gdf, districts_gdf):
    logger.info("开始网格化部署 10,000 个相机...")

    candidates = scan_and_cache_candidates(G, edges_gdf)
    if not candidates:
        return pd.DataFrame()

    df = pd.DataFrame(candidates)
    logger.success(f"初步候选: {len(df)} 个")

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df['x'], df['y']),
        crs="EPSG:4326"
    )

    joined = gpd.sjoin(gdf, districts_gdf, how="left", predicate="within")
    joined['district'] = joined['district'].fillna("其他区")

    # 网格化
    DEG_PER_M = 1 / 111320
    grid_size_deg = GRID_SIZE_M * DEG_PER_M

    def get_grid_key(x, y):
        grid_x = math.floor(x / grid_size_deg)
        grid_y = math.floor(y / grid_size_deg)
        return f"{grid_x}_{grid_y}"

    joined['grid_key'] = joined.apply(lambda row: get_grid_key(row['x'], row['y']), axis=1)

    # 每个格子随机保留 1 个点
    selected_rows = []
    for grid_key, group in tqdm(joined.groupby('grid_key'), desc="网格化去重"):
        if group.empty: continue
        selected = group.sample(1).iloc[0]
        selected_rows.append(selected.to_dict())

    final_gdf = gpd.GeoDataFrame(selected_rows, crs="EPSG:4326").reset_index(drop=True)

    # 如果不足 10,000，补齐
    if len(final_gdf) < TARGET_CAMERAS:
        logger.warning(f"网格化仅 {len(final_gdf)} 个，补齐到 {TARGET_CAMERAS} 个...")
        need = TARGET_CAMERAS - len(final_gdf)
        extra = joined.sample(min(need, len(joined)), replace=False)
        final_gdf = pd.concat([final_gdf, extra], ignore_index=True)

    # 最终采样到 10,000
    final_gdf = final_gdf.sample(TARGET_CAMERAS, replace=False).reset_index(drop=True)

    logger.success(f"最终相机点: {len(final_gdf)} 个")

    # 生成相机
    cameras = []
    for i, row in final_gdf.iterrows():
        name = ""
        if row.get('is_roundabout', False):
            name = "环岛"
        elif row['max_rank'] >= 5:
            name = "主干道"
        elif row['max_rank'] == 4:
            name = "次干道"
        else:
            name = "重要支路"
        name += f"-{row['degree']}向"

        cameras.append({
            'camera_id': f"C{i + 1:06d}",
            'name': name,
            'longitude': row['x'],
            'latitude': row['y'],
            'district': row['district'],
            'source_type': 'Grid10000',
            'layer': row['highway'],
            'grid_size_m': GRID_SIZE_M
        })

    result_df = pd.DataFrame(cameras)
    logger.success(f"相机部署完成: {len(result_df)} 个")
    return result_df


# ==================== 5. 生成地图 ====================
def create_map(df_cameras, districts_gdf):
    if df_cameras.empty: return
    logger.info("生成 10,000 相机地图...")
    center = [df_cameras['latitude'].mean(), df_cameras['longitude'].mean()]
    m = folium.Map(location=center, zoom_start=11, tiles='CartoDB positron')

    for _, row in districts_gdf.iterrows():
        color = DISTRICT_COLORS.get(row['district'], "#95A5A6")
        folium.GeoJson(row['geometry'], style_function=lambda x, c=color: {
            'fillColor': c, 'color': 'black', 'weight': 1.5, 'fillOpacity': 0.3
        }, tooltip=folium.Tooltip(f"<b>{row['district']}</b>")).add_to(m)

    cluster = FastMarkerCluster(data=list(zip(df_cameras.latitude, df_cameras.longitude))).add_to(m)

    layer_colors = {'motorway': '#8B0000', 'trunk': '#DC143C', 'primary': '#FF4500',
                    'secondary': '#32CD32', 'tertiary': '#1E90FF'}

    for _, row in df_cameras.iterrows():
        color = layer_colors.get(row['layer'], '#808080')
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=2, color=color, fill=True, fill_color=color, fill_opacity=0.9,
            popup=folium.Popup(f"<b>{row['name']}</b><br>ID: {row['camera_id']}", max_width=200)
        ).add_to(cluster)

    folium.LayerControl().add_to(m)
    m.save(str(OUTPUT_HTML))
    logger.success(f"地图已保存: {OUTPUT_HTML}")


# ==================== 主程序 ====================
if __name__ == "__main__":
    logger.info("成都智能相机部署系统（10,000 个相机 · 语法修复）")

    G, nodes_gdf, edges_gdf = load_road_network(ROAD_GPKG)
    districts_gdf = load_districts(DISTRICTS_CSV)
    df_cameras = deploy_key_intersections(G, edges_gdf, districts_gdf)

    if not df_cameras.empty:
        df_out = df_cameras[['latitude', 'longitude', 'name', 'camera_id', 'district', 'source_type', 'layer', 'grid_size_m']]
        df_out.to_excel(OUTPUT_EXCEL, index=False)
        logger.success(f"Excel 已保存: {OUTPUT_EXCEL}")

    create_map(df_cameras, districts_gdf)

    print("\n" + "=" * 70)
    print("10,000 个相机部署完成！")
    print("=" * 70)
    print(f"相机总数: {len(df_cameras):,}")
    if not df_cameras.empty:
        print(f"\n分层分布:")
        print(df_cameras['layer'].value_counts().head(6).to_string())
    print(f"\n输出文件:")
    print(f" Excel: {OUTPUT_EXCEL}")
    print(f" 地图: {OUTPUT_HTML}")
    print(f" 缓存: {CACHE_FILE}")
    print("=" * 70)