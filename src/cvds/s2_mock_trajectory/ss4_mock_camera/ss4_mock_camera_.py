# -*- coding: utf-8 -*-
"""
主要路口相机部署 + 真实行政区匹配（支持你的CSV格式）+ 可视化
"""
import warnings
from pathlib import Path

import folium
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from folium.plugins import FastMarkerCluster
from loguru import logger

warnings.filterwarnings("ignore")
# ==================== 路径配置 ====================
BASE_DIR = Path("./road")
INPUT_GPKG = BASE_DIR / "Chengdu_all_road_network.gpkg"
INPUT_DISTRICTS = Path('districts') / "chengdu_districts_boundary.csv"
OUTPUT_EXCEL = BASE_DIR / "main_intersections_camera.xlsx"
OUTPUT_HTML = BASE_DIR / "main_intersections_map_full.html"

for p in [INPUT_GPKG, INPUT_DISTRICTS]:
    if not p.exists():
        raise FileNotFoundError(f"文件未找到: {p.resolve()}")

# ==================== 参数 ====================
MIN_DEGREE = 3
DISTRICT_COLORS = {
    "金牛区": "#FF6B6B", "青羊区": "#4ECDC4", "成华区": "#45B7D1",
    "武侯区": "#96CEB4", "锦江区": "#FECA57", "高新区": "#DDA0DD",
    "天府新区": "#98D8C8", "温江区": "#F7DC6F", "郫都区": "#BB8FCE",
    "新都区": "#85C1E2", "双流区": "#FF9999", "龙泉驿区": "#77DD77",
    "其他区": "#95A5A6"
}


# ==================== 加载行政区边界（修复列名）===================
def load_districts(districts_path):
    logger.info(f"加载行政区边界: {districts_path}")
    df = pd.read_csv(districts_path)

    # 你的真实列名
    NAME_COL = "区域名称"
    GEOM_COL = "区域边界"

    if NAME_COL not in df.columns:
        raise ValueError(f"CSV 必须包含列: '{NAME_COL}'")
    if GEOM_COL not in df.columns:
        raise ValueError(f"CSV 必须包含列: '{GEOM_COL}'（WKT 格式）")

    # 清理空格
    df[NAME_COL] = df[NAME_COL].str.strip()
    df = df.dropna(subset=[NAME_COL, GEOM_COL])

    gdf = gpd.GeoDataFrame(
        df[[NAME_COL]],
        geometry=gpd.GeoSeries.from_wkt(df[GEOM_COL]),
        crs="EPSG:4326"
    )
    gdf = gdf.rename(columns={NAME_COL: 'district'})
    gdf = gdf[['district', 'geometry']].dropna(subset=['geometry'])

    # 合并相同区的多边形
    gdf = gdf.dissolve(by='district').reset_index()

    logger.success(f"加载行政区: {len(gdf)} 个 → {list(gdf['district'])}")
    return gdf


# ==================== 加载路网 ====================
def load_road_network(gpkg_path):
    logger.info("加载路网...")
    nodes = gpd.read_file(gpkg_path, layer="nodes")
    edges = gpd.read_file(gpkg_path, layer="edges")

    source_col = "u" if "u" in edges.columns else "source"
    target_col = "v" if "v" in edges.columns else "target"
    G = nx.from_pandas_edgelist(edges, source=source_col, target=target_col, create_using=nx.MultiGraph())

    node_id_col = "osmid" if "osmid" in nodes.columns else "id"
    for _, row in nodes.iterrows():
        nid = row[node_id_col]
        if nid in G.nodes:
            G.nodes[nid]["x"] = row["x"]
            G.nodes[nid]["y"] = row["y"]

    logger.success(f"加载完成: {G.number_of_nodes():,} 节点, {G.number_of_edges():,} 边")
    return G


# ==================== 部署相机 + 空间匹配 ====================
def deploy_main_intersections(G, districts_gdf):
    logger.info(f"部署主要路口相机（度 ≥ {MIN_DEGREE}）...")
    degrees = dict(G.degree())
    main_nodes = [n for n, deg in degrees.items() if deg >= MIN_DEGREE]

    points_data = []
    osmids = []
    for osmid in main_nodes:
        x = G.nodes[osmid].get("x")
        y = G.nodes[osmid].get("y")
        if x is None or y is None: continue
        points_data.append((x, y))
        osmids.append(osmid)

    if not points_data:
        logger.warning("无有效坐标点")
        return pd.DataFrame()

    points_gdf = gpd.GeoDataFrame(
        {'osmid': osmids},
        geometry=gpd.points_from_xy([p[0] for p in points_data], [p[1] for p in points_data]),
        crs="EPSG:4326"
    )

    logger.info("执行空间连接：匹配点到行政区...")
    joined = gpd.sjoin(points_gdf, districts_gdf, how="left", predicate="within")
    joined['district'] = joined['district'].fillna("其他区")

    cameras = []
    camera_counter = 0
    for _, row in joined.iterrows():
        osmid = row['osmid']
        x, y = G.nodes[osmid]["x"], G.nodes[osmid]["y"]
        district = row['district']

        directions = set()
        for _, v, _, _ in G.edges(osmid, keys=True, data=True):
            if v == osmid: continue
            vx, vy = G.nodes[v].get("x"), G.nodes[v].get("y")
            if vx is None: continue
            dx, dy = vx - x, vy - y
            if abs(dx) < 1e-8 and abs(dy) < 1e-8: continue
            bearing = np.degrees(np.arctan2(dy, dx)) % 360
            dir_bin = round(bearing / 90) * 90
            directions.add(dir_bin)

        camera_counter += 1
        cameras.append({
            'camera_id': f"C{camera_counter:06d}",
            'name': f"主路口-{len(directions)}向",
            'longitude': x,
            'latitude': y,
            'district': district,
            'source_type': 'Point'
        })

    df = pd.DataFrame(cameras)
    logger.success(f"部署完成: {len(df):,} 个相机")
    return df


# ==================== 生成地图（相机 + 行政区） ====================
def create_full_map(df_cameras, districts_gdf, output_path):
    if df_cameras.empty:
        logger.warning("无相机数据，跳过地图")
        return

    logger.info(f"生成地图（{len(df_cameras):,} 相机 + {len(districts_gdf):,} 区）→ {output_path}")
    center = [df_cameras['latitude'].mean(), df_cameras['longitude'].mean()]
    m = folium.Map(location=center, zoom_start=11, tiles='CartoDB positron')

    # === 行政区边界 ===
    for _, row in districts_gdf.iterrows():
        district = row['district']
        color = DISTRICT_COLORS.get(district, "#95A5A6")
        folium.GeoJson(
            row['geometry'],
            style_function=lambda x, c=color, d=district: {
                'fillColor': c,
                'color': 'black',
                'weight': 1.5,
                'fillOpacity': 0.3,
            },
            tooltip=folium.Tooltip(f"<b>{district}</b>", sticky=True),
            name=f"行政区: {district}"
        ).add_to(m)

    # === 相机点（聚类）===
    marker_cluster = FastMarkerCluster(
        data=list(zip(df_cameras.latitude, df_cameras.longitude)),
        name="主要路口相机"
    ).add_to(m)

    for _, row in df_cameras.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=4,
            color="red",
            fill=True,
            fill_color="red",
            fill_opacity=0.9,
            popup=folium.Popup(
                f"<b>{row['name']}</b><br>"
                f"ID: {row['camera_id']}<br>"
                f"区: {row['district']}",
                max_width=200
            )
        ).add_to(marker_cluster)

    # === 图层控制 + 图例 ===
    folium.LayerControl().add_to(m)

    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; width: 190px; height: auto; 
                border:2px solid grey; z-index:9999; font-size:14px; background:white; padding: 10px;
                border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.2);">
      <b style="font-size:16px;">图例</b><br>
      <i class="fa fa-circle" style="color:red"></i>&nbsp;主要路口相机 (度≥3)<br>
      <small>缩放查看聚类</small><hr style="margin:5px 0;">
      <small><b>行政区边界</b></small><br>
    '''
    for d, c in DISTRICT_COLORS.items():
        if d in districts_gdf['district'].values:
            legend_html += f'<i style="background:{c}; width:12px; height:12px; display:inline-block; border:1px solid #666;"></i>&nbsp;{d}<br>'
    legend_html += '</div>'
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(str(output_path))
    logger.success(f"地图保存: {output_path}")


# ==================== 主流程 ====================
if __name__ == "__main__":
    G = load_road_network(INPUT_GPKG)
    districts_gdf = load_districts(INPUT_DISTRICTS)
    df = deploy_main_intersections(G, districts_gdf)

    if not df.empty:
        df_out = df[['latitude', 'longitude', 'name', 'camera_id', 'district', 'source_type']]
        df_out.to_excel(OUTPUT_EXCEL, index=False)
        logger.success(f"Excel 保存: {OUTPUT_EXCEL}，共 {len(df_out):,} 条")

    create_full_map(df, districts_gdf, OUTPUT_HTML)

    print("\n" + "=" * 60)
    print("主要路口相机部署完成！")
    print("=" * 60)
    print(f"相机总数: {len(df):,}")
    print(f"\n各区分布:")
    print(df['district'].value_counts().to_string() if not df.empty else "无")
    print(f"\n输出文件:")
    print(f" Excel: {OUTPUT_EXCEL}")
    print(f" 地图: {OUTPUT_HTML}")
    print("=" * 60)
