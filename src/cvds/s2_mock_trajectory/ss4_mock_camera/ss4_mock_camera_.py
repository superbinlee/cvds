# -*- coding: utf-8 -*-
"""
SIMPLE VERSION: 只部署主要路口相机
- 无投影
- 无密度判断
- 无道路段相机
- 运行 < 10 秒
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import networkx as nx
from loguru import logger
from pathlib import Path
from tqdm import tqdm
import folium
from folium.plugins import FastMarkerCluster

# ==================== 路径配置 ====================
BASE_DIR = Path("road")
INPUT_GPKG = BASE_DIR / "Chengdu_all_road_network.gpkg"
OUTPUT_EXCEL = BASE_DIR / "main_intersections_camera.xlsx"
OUTPUT_HTML = BASE_DIR / "main_intersections_map.html"

if not INPUT_GPKG.exists():
    raise FileNotFoundError(f"路网文件未找到: {INPUT_GPKG}")

# ==================== 参数 ====================
MIN_DEGREE = 3  # 最低度数：三叉口及以上
MAX_DISPLAY = 20000  # 地图最大显示点数


# ==================== 加载路网 ====================
def load_road_network(gpkg_path):
    logger.info("加载路网...")
    nodes = gpd.read_file(gpkg_path, layer="nodes")
    edges = gpd.read_file(gpkg_path, layer="edges")

    source_col = "u" if "u" in edges.columns else "source"
    target_col = "v" if "v" in edges.columns else "target"

    G = nx.from_pandas_edgelist(
        edges, source=source_col, target=target_col,
        create_using=nx.MultiGraph()
    )

    node_id_col = "osmid" if "osmid" in nodes.columns else "id"
    for _, row in nodes.iterrows():
        nid = row[node_id_col]
        if nid in G.nodes:
            G.nodes[nid]["x"] = row["x"]
            G.nodes[nid]["y"] = row["y"]

    logger.success(f"加载完成: {G.number_of_nodes():,} 节点, {G.number_of_edges():,} 边")
    return G


# ==================== 部署主要路口相机 ====================
def deploy_main_intersections(G):
    logger.info(f"部署主要路口相机（度 ≥ {MIN_DEGREE}）...")
    cameras = []

    # 获取所有节点度数
    degrees = dict(G.degree())
    main_nodes = [n for n, deg in degrees.items() if deg >= MIN_DEGREE]

    for osmid in tqdm(main_nodes, desc="部署路口相机", unit="节点"):
        x, y = G.nodes[osmid].get("x"), G.nodes[osmid].get("y")
        if x is None or y is None: continue

        # 获取出边方向（90° 分组）
        directions = set()
        for _, v, _, _ in G.edges(osmid, keys=True, data=True):
            if v == osmid: continue
            vx, vy = G.nodes[v].get("x"), G.nodes[v].get("y")
            if vx is None: continue
            dx, dy = vx - x, vy - y
            if abs(dx) < 1e-8 and abs(dy) < 1e-8: continue
            bearing = np.degrees(np.arctan2(dy, dx)) % 360
            dir_bin = round(bearing / 90) * 90
            if dir_bin not in directions:
                directions.add(dir_bin)
                cameras.append({
                    'camera_id': f"cam_main_{osmid}_dir{int(dir_bin)}",
                    'name': f"主路口-{len(directions)}向",
                    'longitude': x,
                    'latitude': y,
                    'degree': degrees[osmid],
                    'source_type': 'main_intersection'
                })

    logger.success(f"部署完成: {len(cameras):,} 个相机（来自 {len(main_nodes):,} 个主要路口）")
    return pd.DataFrame(cameras)


# ==================== 生成地图 ====================
def create_simple_map(df, output_path):
    if df.empty:
        logger.warning("无数据，跳过地图")
        return

    logger.info("生成地图...")
    center = [df['latitude'].mean(), df['longitude'].mean()]
    m = folium.Map(location=center, zoom_start=11, tiles='CartoDB positron')

    # 图例
    legend = '''
    <div style="position: fixed; bottom: 50px; left: 50px; width: 120px; height: 50px;
         border:2px solid grey; z-index:9999; font-size:14px; background:white; padding: 8px;">
     <b>主要路口相机</b><br>
     <i style="background:red; width:10px; height:10px; border-radius:50%; display:inline-block;"></i> 度≥3
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend))

    df_show = df.sample(min(MAX_DISPLAY, len(df)), random_state=42) if len(df) > MAX_DISPLAY else df
    FastMarkerCluster(
        data=list(zip(df_show.latitude, df_show.longitude)),
        name="主要路口相机"
    ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(str(output_path))
    logger.success(f"地图保存: {output_path}")


# ==================== 主流程 ====================
if __name__ == "__main__":
    G = load_road_network(INPUT_GPKG)
    df = deploy_main_intersections(G)

    # 保存 Excel
    if not df.empty:
        df = df[['camera_id', 'name', 'longitude', 'latitude', 'degree', 'source_type']]
        df.to_excel(OUTPUT_EXCEL, index=False)
        logger.success(f"Excel 保存: {OUTPUT_EXCEL}，共 {len(df):,} 条")

    # 生成地图
    create_simple_map(df, OUTPUT_HTML)

    # 统计
    print("\n" + "=" * 50)
    print("主要路口相机部署完成！")
    print(f"主要路口数: {len(df['camera_id'].str.split('_').str[2].unique()):,}")
    print(f"相机总数: {len(df):,}")
    print(f"平均每路口相机数: {len(df) / len(df['camera_id'].str.split('_').str[2].unique()):.1f}")
    print(f"输出: {OUTPUT_EXCEL}, {OUTPUT_HTML}")
    print("=" * 50)
