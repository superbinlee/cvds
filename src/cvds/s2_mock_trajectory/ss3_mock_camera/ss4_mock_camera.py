# -*- coding: utf-8 -*-
"""
成都关键路口智能相机部署系统（修复 + 高效 + 稳定）
已修复：空候选集崩溃问题
"""

import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

import folium
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from folium.plugins import FastMarkerCluster
from loguru import logger
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ==================== 路径配置 ====================
ROAD_GPKG = Path("road") / "Chengdu_all_road_network.gpkg"
DISTRICTS_CSV = Path("districts") / "chengdu_districts_boundary.csv"
OUTPUT_DIR = Path("output") / "camera"
OUTPUT_EXCEL = OUTPUT_DIR / "key_intersections_camera.xlsx"
OUTPUT_HTML = OUTPUT_DIR / "key_intersections_map.html"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for p in [ROAD_GPKG, DISTRICTS_CSV]:
    if not p.exists():
        raise FileNotFoundError(f"文件未找到: {p}")

# ==================== 参数 ====================
MIN_DEGREE = 3
CLUSTER_DISTANCE_M = 200
CORE_DISTRICTS = {"金牛区", "青羊区", "武侯区", "锦江区", "成华区", "高新区", "天府新区"}
MAX_WORKERS = 6

DISTRICT_COLORS = {
    "金牛区": "#FF6B6B", "青羊区": "#4ECDC4", "成华区": "#45B7D1",
    "武侯区": "#96CEB4", "锦江区": "#FECA57", "高新区": "#DDA0DD",
    "天府新区": "#98D8C8", "温江区": "#F7DC6F", "郫都区": "#BB8FCE",
    "新都区": "#85C1E2", "双流区": "#FF9999", "龙泉驿区": "#77DD77",
    "其他区": "#95A5A6"
}

HIGHWAY_RANK = {
    'motorway': 6, 'trunk': 6, 'primary': 5, 'secondary': 4,
    'tertiary': 3, 'unclassified': 2, 'residential': 1, 'service': 0
}

# ==================== 日志 ====================
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


# ==================== 3. 并行评分 ====================
def score_node(args):
    node, G, edges, HIGHWAY_RANK = args
    x = G.nodes[node].get("x")
    y = G.nodes[node].get("y")
    if x is None or y is None:
        return None

    deg = G.degree[node]
    if deg < MIN_DEGREE:
        return None

    neighbor_edges = edges[(edges['u'] == node) | (edges['v'] == node)]
    if neighbor_edges.empty:
        return None

    max_rank = neighbor_edges['rank'].max()
    is_roundabout = neighbor_edges['is_roundabout'].any()

    score = 0
    if deg >= 5:
        score += 30
    elif deg == 4:
        score += 20
    elif deg == 3:
        score += 10
    if max_rank >= 5:
        score += 25
    elif max_rank >= 4:
        score += 15
    elif max_rank >= 3:
        score += 5
    if is_roundabout: score += 40

    return {
        'osmid': node, 'x': x, 'y': y,
        'degree': deg, 'max_rank': max_rank,
        'is_roundabout': is_roundabout, 'score': score
    }


# ==================== 4. 关键路口识别（安全去重） ====================
def deploy_key_intersections(G, edges_gdf, districts_gdf):
    logger.info("开始关键路口识别...")

    edges = edges_gdf.copy()
    edges['highway'] = edges['highway'].fillna('residential')
    edges['rank'] = edges['highway'].apply(lambda x: HIGHWAY_RANK.get(str(x).split(';')[0], 0))
    edges['is_roundabout'] = edges['junction'] == 'roundabout'

    tasks = [(node, G, edges, HIGHWAY_RANK) for node in G.nodes()]
    node_scores = []

    logger.info(f"并行评分 {len(tasks):,} 个节点...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(score_node, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="评分进度"):
            result = future.result()
            if result:
                node_scores.append(result)

    if not node_scores:
        logger.warning("无候选路口")
        return pd.DataFrame()

    scores_df = pd.DataFrame(node_scores)
    logger.success(f"初步候选: {len(scores_df)} 个")

    points_gdf = gpd.GeoDataFrame(
        scores_df,
        geometry=gpd.points_from_xy(scores_df['x'], scores_df['y']),
        crs="EPSG:4326"
    )

    # 过滤无效几何
    points_gdf = points_gdf[points_gdf.geometry.notna() & ~points_gdf.geometry.is_empty]

    logger.info("空间连接行政区...")
    joined = gpd.sjoin(points_gdf, districts_gdf, how="left", predicate="within")
    joined['district'] = joined['district'].fillna("其他区")
    joined['is_core'] = joined['district'].isin(CORE_DISTRICTS)
    joined['score'] = joined['score'] + joined['is_core'].apply(lambda x: 20 if x else 0)

    # 安全去重
    logger.info(f"空间去重（{CLUSTER_DISTANCE_M}m）...")
    joined = joined.sort_values('score', ascending=False).reset_index(drop=True)
    buffer_gdf = joined.copy()
    buffer_gdf['geometry'] = buffer_gdf.geometry.buffer(CLUSTER_DISTANCE_M / 111320)

    selected = []
    used = set()
    for idx in tqdm(buffer_gdf.index, desc="去重进度"):
        if idx in used:
            continue
        row = buffer_gdf.loc[idx]
        if pd.isna(row.geometry) or row.geometry.is_empty:
            continue
        candidates = buffer_gdf[buffer_gdf.geometry.overlaps(row.geometry)]
        if candidates.empty:
            continue
        best_idx = candidates['score'].idxmax()
        best = candidates.loc[best_idx]
        selected.append(best)
        used.update(candidates.index)

    final_gdf = gpd.GeoDataFrame(selected, crs="EPSG:4326").reset_index(drop=True)
    logger.success(f"最终关键路口: {len(final_gdf)} 个")

    cameras = []
    for i, row in final_gdf.iterrows():
        name = "关键路口"
        if row['is_roundabout']: name = "环岛路口"
        if row['max_rank'] >= 5: name = "主干道-" + name
        name += f"-{row['degree']}向"
        cameras.append({
            'camera_id': f"C{i + 1:06d}",
            'name': name,
            'longitude': row['x'],
            'latitude': row['y'],
            'district': row['district'],
            'source_type': 'KeyIntersection',
            'score': int(row['score'])
        })

    df = pd.DataFrame(cameras)
    logger.success(f"相机部署完成: {len(df)} 个")
    return df


# ==================== 5. 生成地图 ====================
def create_map(df_cameras, districts_gdf):
    if df_cameras.empty:
        logger.warning("无相机，跳过地图")
        return

    logger.info("生成地图...")
    center = [df_cameras['latitude'].mean(), df_cameras['longitude'].mean()]
    m = folium.Map(location=center, zoom_start=12, tiles='CartoDB positron')

    for _, row in districts_gdf.iterrows():
        color = DISTRICT_COLORS.get(row['district'], "#95A5A6")
        folium.GeoJson(
            row['geometry'],
            style_function=lambda x, c=color: {'fillColor': c, 'color': 'black', 'weight': 1.5, 'fillOpacity': 0.3},
            tooltip=folium.Tooltip(f"<b>{row['district']}</b>")
        ).add_to(m)

    cluster = FastMarkerCluster(data=list(zip(df_cameras.latitude, df_cameras.longitude))).add_to(m)
    min_score = df_cameras['score'].min()
    max_score = df_cameras['score'].max()
    for _, row in df_cameras.iterrows():
        radius = 4 + 6 * (row['score'] - min_score) / (max_score - min_score + 1)
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=radius, color="red", fill=True, fill_color="red", fill_opacity=0.9,
            popup=folium.Popup(
                f"<b>{row['name']}</b><br>ID: {row['camera_id']}<br>区: {row['district']}<br>评分: {row['score']}",
                max_width=200
            )
        ).add_to(cluster)

    folium.LayerControl().add_to(m)
    m.get_root().html.add_child(folium.Element("""
    <div style="position: fixed; bottom: 50px; left: 50px; width: 220px; padding: 10px; 
                background: white; border: 2px solid grey; border-radius: 8px; font-size: 14px; z-index: 9999;">
      <b>图例</b><br>
      <i class="fa fa-circle" style="color:red"></i> 关键路口相机 (大小=重要性)<br>
      <small>缩放查看聚类</small><hr style="margin:5px 0;">
      <b>行政区</b><br>
    """ + "".join([f'<i style="background:{c}; width:12px; height:12px; display:inline-block; border:1px solid #666;"></i> {d}<br>'
                   for d, c in DISTRICT_COLORS.items() if d in districts_gdf['district'].values]) + "</div>"))
    m.save(str(OUTPUT_HTML))
    logger.success(f"地图已保存: {OUTPUT_HTML}")


# ==================== 主程序 ====================
if __name__ == "__main__":
    logger.info("成都关键路口智能相机部署系统（稳定版）")

    G, nodes_gdf, edges_gdf = load_road_network(ROAD_GPKG)
    districts_gdf = load_districts(DISTRICTS_CSV)
    df_cameras = deploy_key_intersections(G, edges_gdf, districts_gdf)

    if not df_cameras.empty:
        df_out = df_cameras[['latitude', 'longitude', 'name', 'camera_id', 'district', 'source_type']]
        df_out.to_excel(OUTPUT_EXCEL, index=False)
        logger.success(f"Excel 已保存: {OUTPUT_EXCEL}")

    create_map(df_cameras, districts_gdf)

    print("\n" + "=" * 70)
    print("关键路口相机部署完成！")
    print("=" * 70)
    print(f"相机总数: {len(df_cameras):,}")
    if not df_cameras.empty:
        print(f"\n各区分布 (Top 5):")
        print(df_cameras['district'].value_counts().head(5).to_string())
    print(f"\n输出文件:")
    print(f" Excel: {OUTPUT_EXCEL}")
    print(f" 地图: {OUTPUT_HTML}")
    print("=" * 70)
