# -*- coding: utf-8 -*-
"""
SIMPLE + FULL DISPLAY + CUSTOM OUTPUT
- 只部署主要路口（度 ≥ 3）
- 全量展示所有相机点
- 输出字段：latitude, longitude, name, camera_id, district, source_type
- camera_id 格式：CXXXXXX
- source_type：Point
"""

import geopandas as gpd
import pandas as pd
import networkx as nx
from loguru import logger
from pathlib import Path
from tqdm import tqdm
import folium
from folium.plugins import FastMarkerCluster
import numpy as np

# ==================== 路径配置 ====================
BASE_DIR = Path("road")
INPUT_GPKG = BASE_DIR / "Chengdu_all_road_network.gpkg"
OUTPUT_EXCEL = BASE_DIR / "main_intersections_camera.xlsx"
OUTPUT_HTML = BASE_DIR / "main_intersections_map_full.html"

if not INPUT_GPKG.exists():
    raise FileNotFoundError(f"路网文件未找到: {INPUT_GPKG.resolve()}")

# ==================== 参数 ====================
MIN_DEGREE = 3  # 最低部署度数：3叉口及以上

# 行政区映射
DISTRICT_MAP = {
    "温江区": "Wenjiang",
    "金牛区": "Jinniu",
    "成华区": "Chenghua",
    "郫都区": "Pidu",
    "新都区": "Xindu",
    "青羊区": "Qingyang"
}


# ==================== 行政区提取 ====================
def extract_district(osmid):
    try:
        suffix = str(osmid).split("_")[-1]
        for cn, en in DISTRICT_MAP.items():
            if en.lower() in suffix.lower():
                return cn
        return "其他区"
    except:
        return "其他区"


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
    camera_counter = 0  # 用于生成 CXXXXXX 格式 ID

    degrees = dict(G.degree())
    main_nodes = [n for n, deg in degrees.items() if deg >= MIN_DEGREE]

    for osmid in tqdm(main_nodes, desc="部署路口相机", unit="节点"):
        x, y = G.nodes[osmid].get("x"), G.nodes[osmid].get("y")
        if x is None or y is None: continue

        district = extract_district(osmid)
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
                camera_counter += 1
                cameras.append({
                    'camera_id': f"C{camera_counter:06d}",  # C000001 格式
                    'name': f"主路口-{len(directions)}向",
                    'longitude': x,
                    'latitude': y,
                    'district': district,
                    'source_type': 'Point'
                })

    logger.success(f"部署完成: {len(cameras):,} 个相机（来自 {len(main_nodes):,} 个主要路口）")
    return pd.DataFrame(cameras)


# ==================== 全量展示地图 ====================
def create_full_map(df, output_path):
    if df.empty:
        logger.warning("无数据，跳过地图")
        return

    logger.info(f"生成全量地图（{len(df):,} 个点） → {output_path}")
    center = [df['latitude'].mean(), df['longitude'].mean()]
    m = folium.Map(location=center, zoom_start=11, tiles='CartoDB positron')

    # 图例
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; width: 150px; height: 60px;
         border:2px solid grey; z-index:9999; font-size:14px; background:white; padding: 10px;">
     <b>主要路口相机</b><br>
     <i style="background:red; width:12px; height:12px; border-radius:50%; display:inline-block;"></i> 度≥3<br>
     <small>缩放可查看细节</small>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # 全量数据，FastMarkerCluster 自动聚类
    FastMarkerCluster(
        data=list(zip(df.latitude, df.longitude)),
        name="主要路口相机"
    ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(str(output_path))
    logger.success(f"全量地图保存: {output_path}")


# ==================== 主流程 ====================
if __name__ == "__main__":
    # 1. 加载路网
    G = load_road_network(INPUT_GPKG)

    # 2. 部署相机
    df = deploy_main_intersections(G)

    # 3. 保存 Excel（按指定字段顺序）
    if not df.empty:
        df_out = df[['latitude', 'longitude', 'name', 'camera_id', 'district', 'source_type']]
        df_out.to_excel(OUTPUT_EXCEL, index=False)
        logger.success(f"Excel 保存: {OUTPUT_EXCEL}，共 {len(df_out):,} 条")

    # 4. 生成全量地图
    create_full_map(df, OUTPUT_HTML)

    # 5. 统计
    print("\n" + "=" * 60)
    print("主要路口相机部署完成！")
    print("=" * 60)
    print(f"主要路口数: {len(df['camera_id'].str[1:].astype(int).unique()):,}")
    print(f"相机总数: {len(df):,}")
    print(f"平均每路口相机数: {len(df) / len(df['camera_id'].str[1:].astype(int).unique()):.1f}")
    print(f"\n各区分布:")
    print(df['district'].value_counts() if not df.empty else "无")
    print(f"\n输出文件:")
    print(f"  Excel: {OUTPUT_EXCEL}")
    print(f"  地图: {OUTPUT_HTML}")
    print("=" * 60)
