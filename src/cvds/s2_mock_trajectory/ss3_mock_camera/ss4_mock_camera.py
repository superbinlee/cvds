# -*- coding: utf-8 -*-
"""
成都智能相机部署系统（生产级 · 优化版）
输入：
  road/Chengdu_all_road_network.gpkg
  districts/chengdu_districts_boundary.csv
输出：
  output/camera/smart_intersections_camera.xlsx
  output/camera/smart_intersections_map.html
"""

import warnings
from pathlib import Path

import folium
import geopandas as gpd
import networkx as nx
import pandas as pd
from folium.plugins import FastMarkerCluster
from loguru import logger
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ==================== 路径配置 ====================
ROAD_GPKG = Path("road") / "Chengdu_all_road_network.gpkg"
DISTRICTS_CSV = Path("districts") / "chengdu_districts_boundary.csv"
OUTPUT_DIR = Path("output") / "camera"
OUTPUT_EXCEL = OUTPUT_DIR / "smart_intersections_camera.xlsx"
OUTPUT_HTML = OUTPUT_DIR / "smart_intersections_map.html"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for p in [ROAD_GPKG, DISTRICTS_CSV]:
    if not p.exists():
        raise FileNotFoundError(f"文件未找到: {p}")

# ==================== 参数配置 ====================
MIN_DEGREE = 3
CORE_DISTRICTS = {"金牛区", "青羊区", "武侯区", "锦江区", "成华区", "高新区", "天府新区"}
INCLUDE_TERTIARY = True

# 动态间距（米）
SPACING_RULE = {6: 300, 5: 300, 4: 400, 3: 1000}

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


# ==================== 3. 智能相机部署（全局去重） ====================
def deploy_key_intersections(G, edges_gdf, districts_gdf):
    logger.info("开始智能相机部署（全局去重 + 动态间距）...")

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

        # 提取最高等级道路
        highway_list = []
        for h in neighbor_edges['highway']:
            highway_list.extend([item.strip() for item in str(h).split(';') if item.strip()])
        if not highway_list: continue
        highway_type = max(highway_list, key=lambda x: HIGHWAY_RANK.get(x, 0))
        max_rank = HIGHWAY_RANK.get(highway_type, 0)

        if max_rank < 3: continue
        if max_rank == 3 and not INCLUDE_TERTIARY: continue

        spacing = SPACING_RULE.get(max_rank, 1000)
        is_roundabout = neighbor_edges['is_roundabout'].any()

        candidates.append({
            'osmid': node, 'x': x, 'y': y,
            'degree': deg, 'max_rank': max_rank,
            'highway': highway_type, 'is_roundabout': is_roundabout,
            'spacing': spacing, 'priority': max_rank * 100 + deg
        })

    if not candidates:
        logger.warning("无候选路口")
        return pd.DataFrame()

    df = pd.DataFrame(candidates)
    logger.success(f"初步候选: {len(df)} 个")

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df['x'], df['y']),
        crs="EPSG:4326"
    )

    # 行政区匹配
    joined = gpd.sjoin(gdf, districts_gdf, how="left", predicate="within")
    joined['district'] = joined['district'].fillna("其他区")

    # 全局去重（按优先级）
    joined = joined.sort_values('priority', ascending=False)
    buffer_gdf = joined.copy()
    buffer_gdf['buffer_dist'] = buffer_gdf['spacing'] / 111320
    buffer_gdf['geometry'] = buffer_gdf.geometry.buffer(buffer_gdf['buffer_dist'])

    selected = []
    used = set()
    for idx in tqdm(buffer_gdf.index, desc="全局去重", leave=False):
        if idx in used: continue
        row = buffer_gdf.loc[idx]
        if pd.isna(row.geometry): continue
        candidates = buffer_gdf[buffer_gdf.geometry.overlaps(row.geometry)]
        if candidates.empty: continue
        best = candidates.loc[candidates['priority'].idxmax()]
        selected.append(best)
        used.update(candidates.index)

    final_gdf = gpd.GeoDataFrame(selected, crs="EPSG:4326").drop_duplicates(subset=['osmid']).reset_index(drop=True)
    logger.success(f"最终相机点: {len(final_gdf)} 个")

    # 生成相机
    cameras = []
    for i, row in final_gdf.iterrows():
        name = ""
        if row['is_roundabout']:
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
            'source_type': 'SmartIntersection',
            'layer': row['highway'],
            'spacing_m': row['spacing']
        })

    result_df = pd.DataFrame(cameras)
    logger.success(f"相机部署完成: {len(result_df)} 个")
    return result_df


# ==================== 4. 生成地图 ====================
def create_map(df_cameras, districts_gdf):
    if df_cameras.empty:
        logger.warning("无相机，跳过地图")
        return

    logger.info("生成智能交互地图...")
    center = [df_cameras['latitude'].mean(), df_cameras['longitude'].mean()]
    m = folium.Map(location=center, zoom_start=11, tiles='CartoDB positron')

    # 行政区
    for _, row in districts_gdf.iterrows():
        color = DISTRICT_COLORS.get(row['district'], "#95A5A6")
        folium.GeoJson(
            row['geometry'],
            style_function=lambda x, c=color: {'fillColor': c, 'color': 'black', 'weight': 1.5, 'fillOpacity': 0.3},
            tooltip=folium.Tooltip(f"<b>{row['district']}</b>")
        ).add_to(m)

    # 相机点
    cluster = FastMarkerCluster(data=list(zip(df_cameras.latitude, df_cameras.longitude)), name="智能相机").add_to(m)

    layer_colors = {
        'motorway': '#8B0000', 'trunk': '#DC143C', 'primary': '#FF4500',
        'secondary': '#32CD32', 'tertiary': '#1E90FF'
    }

    for _, row in df_cameras.iterrows():
        highway = row['layer']
        color = layer_colors.get(highway, '#808080')
        radius = 5 if highway in ['motorway', 'trunk', 'primary'] else 4 if highway == 'secondary' else 3
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=radius, color=color, fill=True, fill_color=color, fill_opacity=0.9,
            popup=folium.Popup(
                f"<b>{row['name']}</b><br>ID: {row['camera_id']}<br>区: {row['district']}<br>层级: {highway}",
                max_width=200
            )
        ).add_to(cluster)

    folium.LayerControl().add_to(m)

    # 图例
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; width: 180px; padding: 10px; 
                background: white; border: 2px solid grey; border-radius: 8px; font-size: 14px; z-index: 9999;">
      <b>图例</b><br>
      <i class="fa fa-circle" style="color:#DC143C"></i> 主干道 (300m)<br>
      <i class="fa fa-circle" style="color:#32CD32"></i> 次干道 (400m)<br>
      <i class="fa fa-circle" style="color:#1E90FF"></i> 支路 (1000m)<br>
      <small>缩放查看聚类</small>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(str(OUTPUT_HTML))
    logger.success(f"地图已保存: {OUTPUT_HTML}")


# ==================== 主程序 ====================
if __name__ == "__main__":
    logger.info("成都智能相机部署系统（生产级 · 优化版）")

    G, nodes_gdf, edges_gdf = load_road_network(ROAD_GPKG)
    districts_gdf = load_districts(DISTRICTS_CSV)
    df_cameras = deploy_key_intersections(G, edges_gdf, districts_gdf)

    if not df_cameras.empty:
        df_out = df_cameras[['latitude', 'longitude', 'name', 'camera_id', 'district', 'source_type', 'layer', 'spacing_m']]
        df_out.to_excel(OUTPUT_EXCEL, index=False)
        logger.success(f"Excel 已保存: {OUTPUT_EXCEL}")

    create_map(df_cameras, districts_gdf)

    print("\n" + "=" * 70)
    print("智能相机部署完成！")
    print("=" * 70)
    print(f"相机总数: {len(df_cameras):,}")
    if not df_cameras.empty:
        print(f"\n分层分布:")
        print(df_cameras['layer'].value_counts().head(6).to_string())
    print(f"\n输出文件:")
    print(f" Excel: {OUTPUT_EXCEL}")
    print(f" 地图: {OUTPUT_HTML}")
    print("=" * 70)
