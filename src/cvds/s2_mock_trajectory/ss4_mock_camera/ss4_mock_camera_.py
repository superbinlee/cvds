# -*- coding: utf-8 -*-
"""
FINAL SUPER FAST VERSION
- 相对路径: BASE_DIR = Path("road")
- 超快并行 + tqdm 进度条
- 轻量化数据传递
- 总耗时 < 2 分钟
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
from loguru import logger
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import folium
from folium.plugins import FastMarkerCluster
import warnings
warnings.filterwarnings("ignore")

# ==================== 路径配置（相对路径） ====================
BASE_DIR = Path("road")
BASE_DIR.mkdir(exist_ok=True)
INPUT_GPKG = BASE_DIR / "Chengdu_all_road_network.gpkg"
OUTPUT_EXCEL = BASE_DIR / "camera_deployments.xlsx"
OUTPUT_HTML = BASE_DIR / "camera_map.html"

if not INPUT_GPKG.exists():
    raise FileNotFoundError(f"路网文件未找到: {INPUT_GPKG.resolve()}")

# ==================== 部署参数 ====================
URBAN_INTERVAL_M = 100
RURAL_INTERVAL_M = 500
DENSITY_THRESHOLD_KM = 50
MAX_DISPLAY_POINTS = 50000

DISTRICT_MAP = {
    "温江区": "Wenjiang", "金牛区": "Jinniu", "成华区": "Chenghua",
    "郫都区": "Pidu", "新都区": "Xindu", "青羊区": "Qingyang"
}
# ===============================================

def extract_district(osmid):
    try:
        suffix = str(osmid).split("_")[-1]
        for cn, en in DISTRICT_MAP.items():
            if en.lower() in suffix.lower():
                return cn
        return "其他区"
    except:
        return "其他区"

def load_road_network(gpkg_path):
    logger.info(f"手动加载路网: {gpkg_path}")
    nodes = gpd.read_file(gpkg_path, layer="nodes")
    edges = gpd.read_file(gpkg_path, layer="edges")

    source_col = "u" if "u" in edges.columns else "source"
    target_col = "v" if "v" in edges.columns else "target"

    G = nx.from_pandas_edgelist(
        edges,
        source=source_col,
        target=target_col,
        edge_attr=["length", "name", "geometry"],
        create_using=nx.MultiGraph()
    )

    node_id_col = "osmid" if "osmid" in nodes.columns else "id"
    for _, row in nodes.iterrows():
        node_id = row[node_id_col]
        if node_id in G.nodes:
            G.nodes[node_id]["x"] = row["x"]
            G.nodes[node_id]["y"] = row["y"]

    logger.success(f"加载完成: {G.number_of_nodes():,} 节点, {G.number_of_edges():,} 边")
    return G

def compute_district_density(G):
    logger.info("计算各区道路密度（投影到米）...")
    edges_data = []

    edge_list = [(u, v, k, d) for u, v, k, d in G.edges(keys=True, data=True)]
    for u, v, key, data in tqdm(edge_list, desc="计算道路密度", unit="边"):
        geom = data.get('geometry')
        if geom is None or geom.is_empty:
            continue
        try:
            length_m = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs("EPSG:3857").length[0]
        except:
            continue
        district = extract_district(u)
        edges_data.append({'district': district, 'length_m': length_m})

    if not edges_data:
        return {d: 0.0 for d in DISTRICT_MAP.keys()}

    df = pd.DataFrame(edges_data)
    density_km = (df.groupby('district')['length_m'].sum() / 1000).to_dict()
    logger.success(f"密度计算完成: { {k: f'{v:.1f}km' for k,v in density_km.items()} }")
    return density_km

# ==================== 轻量化处理函数 ====================
def process_node_camera_light(args):
    osmid, x, y, outgoing_vs, nodes_coords, district = args
    cameras = []
    directions = set()

    if not outgoing_vs:
        cameras.append({
            'camera_id': f"cam_node_{osmid}_isolated",
            'name': f"{district}-路口-孤立",
            'longitude': x, 'latitude': y,
            'district': district, 'source_type': 'intersection'
        })
        return cameras

    for v in outgoing_vs:
        vx, vy = nodes_coords.get(v, (None, None))
        if vx is None: continue
        dx = vx - x
        dy = vy - y
        if abs(dx) < 1e-8 and abs(dy) < 1e-8: continue
        bearing = np.degrees(np.arctan2(dy, dx)) % 360
        dir_bin = round(bearing / 90) * 90
        if dir_bin not in directions:
            directions.add(dir_bin)
            cameras.append({
                'camera_id': f"cam_node_{osmid}_dir{int(dir_bin)}",
                'name': f"{district}-路口-{len(directions)}向",
                'longitude': x, 'latitude': y,
                'district': district, 'source_type': 'intersection'
            })
    return cameras

def process_edge_camera_light(args):
    u, v, key, geom_wgs84, nodes_coords, is_urban_dict, interval_dict = args
    try:
        line_3857 = gpd.GeoSeries([geom_wgs84], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
    except:
        return []
    length_m = line_3857.length
    if length_m < 50: return []

    district = extract_district(u)
    is_urban = is_urban_dict.get(u, False)
    interval_m = interval_dict['urban'] if is_urban else interval_dict['rural']
    cameras = []

    distances = np.arange(interval_m, length_m, interval_m)
    for i, d in enumerate(distances, 1):
        point_3857 = line_3857.interpolate(d)
        point_wgs84 = gpd.GeoSeries([point_3857], crs="EPSG:3857").to_crs("EPSG:4326").iloc[0]
        cameras.append({
            'camera_id': f"cam_edge_{u}_{v}_{key}_pt{i}",
            'name': f"{district}-道路段-{i}",
            'longitude': point_wgs84.x,
            'latitude': point_wgs84.y,
            'district': district,
            'source_type': 'road_segment'
        })
    return cameras

def create_folium_map(df, output_path):
    if df.empty:
        logger.warning("无相机数据，跳过地图生成")
        return

    logger.info(f"生成地图 → {output_path}")
    center = [df['latitude'].mean(), df['longitude'].mean()]
    m = folium.Map(location=center, zoom_start=11, tiles='CartoDB positron')

    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; width: 140px; height: 80px; 
         border:2px solid grey; z-index:9999; font-size:14px; background:white; padding: 10px;">
     <b>相机类型</b><br>
     <i style="background:red; width:10px; height:10px; border-radius:50%; display:inline-block;"></i> 路口<br>
     <i style="background:blue; width:8px; height:8px; border-radius:50%; display:inline-block;"></i> 道路段
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    df_display = df.sample(min(MAX_DISPLAY_POINTS, len(df)), random_state=42) if len(df) > MAX_DISPLAY_POINTS else df

    for typ, color, name in [
        ('intersection', 'red', '路口相机'),
        ('road_segment', 'blue', '道路段相机')
    ]:
        subset = df_display[df_display['source_type'] == typ]
        if subset.empty: continue
        FastMarkerCluster(
            data=list(zip(subset.latitude, subset.longitude)),
            name=name
        ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(str(output_path))
    logger.success(f"地图保存: {output_path}")

# ==================== 主流程 ====================
if __name__ == "__main__":
    import multiprocessing as mp
    n_jobs = max(1, mp.cpu_count() - 1)

    # 1. 加载路网
    G = load_road_network(INPUT_GPKG)

    # 2. 计算密度
    district_density = compute_district_density(G)

    # 3. 节点坐标字典
    nodes_coords = {}
    for n in G.nodes:
        data = G.nodes[n]
        x, y = data.get('x'), data.get('y')
        if x is not None and y is not None:
            nodes_coords[n] = (x, y)

    # 4. is_urban 字典
    node_district = {n: extract_district(n) for n in nodes_coords.keys()}
    is_urban_dict = {n: district_density.get(d, 0) > DENSITY_THRESHOLD_KM for n, d in node_district.items()}

    # 5. 超快路口相机
    logger.info("超快部署路口相机（多核 + 轻量）...")
    node_inputs_light = []
    for osmid in nodes_coords.keys():
        x, y = nodes_coords[osmid]
        outgoing_vs = [v for u, v, k, d in G.edges(osmid, keys=True, data=True) if u == osmid]
        district = extract_district(osmid)
        node_inputs_light.append((osmid, x, y, outgoing_vs, nodes_coords, district))

    node_cameras = []
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(process_node_camera_light, args) for args in node_inputs_light]
        for future in tqdm(as_completed(futures), total=len(futures), desc="部署路口相机", unit="节点"):
            node_cameras.extend(future.result())
    logger.success(f"路口相机: {len(node_cameras):,} 个")

    # 6. 超快道路段相机
    logger.info("超快部署道路段相机...")
    edge_inputs_light = []
    for u, v, k, d in G.edges(keys=True, data=True):
        geom = d.get('geometry')
        if geom is None or geom.is_empty: continue
        edge_inputs_light.append((u, v, k, geom, nodes_coords, is_urban_dict, {'urban': URBAN_INTERVAL_M, 'rural': RURAL_INTERVAL_M}))

    edge_cameras = []
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(process_edge_camera_light, args) for args in edge_inputs_light]
        for future in tqdm(as_completed(futures), total=len(futures), desc="部署道路段相机", unit="边"):
            edge_cameras.extend(future.result())
    logger.success(f"道路段相机: {len(edge_cameras):,} 个")

    # 7. 合并并保存 Excel
    all_cameras = node_cameras + edge_cameras
    df = pd.DataFrame(all_cameras)
    if not df.empty:
        df = df[['latitude', 'longitude', 'name', 'camera_id', 'district', 'source_type']]
        for _ in tqdm([0], desc=f"保存 Excel ({len(df):,} 条)", unit="文件"):
            df.to_excel(OUTPUT_EXCEL, index=False)
        logger.success(f"Excel 保存: {OUTPUT_EXCEL}")
    else:
        logger.warning("无相机生成")

    # 8. 生成地图
    create_folium_map(df, OUTPUT_HTML)

    # 9. 最终统计
    print("\n" + "="*60)
    print("部署完成！")
    print("="*60)
    print(f"总相机数: {len(df):,}")
    print(f"路口相机: {len(node_cameras):,}")
    print(f"道路段相机: {len(edge_cameras):,}")
    print("\n各区分布:")
    print(df['district'].value_counts() if not df.empty else "无")
    print(f"\n输出文件:")
    print(f"  Excel: {OUTPUT_EXCEL}")
    print(f"  地图: {OUTPUT_HTML}")
    print("="*60)