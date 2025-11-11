#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
为每个相机点分配方向（u → v 节点编号），支持 u/v 为 "123456_行政区" 格式。
输出：
    ../cameras_all_with_directions.xlsx
    ../map_with_cameras_and_directions.html
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import folium
import pyproj
from shapely.geometry import Point
from tqdm import tqdm
import logging
from pathlib import Path

# ------------------- 配置 -------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path("..")
DISTRICTS_CSV = BASE_DIR / "districts" / "chengdu_districts_boundary.csv"
CAMERAS_XLSX = BASE_DIR / "cameras_all.xlsx"
ROAD_GPKG = BASE_DIR / "roadnet" / "Chengdu_all_road_network.gpkg"

OUTPUT_XLSX = BASE_DIR / "cameras_all_with_directions.xlsx"
OUTPUT_HTML = BASE_DIR / "map_with_cameras_and_directions.html"

PROJECTED_CRS = "EPSG:32648"  # 成都 UTM Zone 48N


# ------------------- 加载数据 -------------------
def load_districts(csv_path: Path) -> gpd.GeoDataFrame:
    logger.info("加载行政区边界...")
    df = pd.read_csv(csv_path)

    def fix_wkt(wkt: str) -> str:
        wkt = wkt.strip()
        if not (wkt.endswith('))') or wkt.endswith(')))')):
            if 'POLYGON' in wkt:
                wkt += '))'
            elif 'MULTIPOLYGON' in wkt:
                wkt += ')))'
        return wkt

    df['区域边界'] = df['区域边界'].apply(fix_wkt)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.GeoSeries.from_wkt(df['区域边界']),
        crs="EPSG:4326"
    ).to_crs(PROJECTED_CRS)
    return gdf


def load_cameras(xlsx_path: Path) -> gpd.GeoDataFrame:
    logger.info("加载相机点位...")
    df = pd.read_excel(xlsx_path)
    required = {'latitude', 'longitude', 'name', 'camera_id', 'district'}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"相机文件缺少列: {missing}")
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326").to_crs(PROJECTED_CRS)
    return gdf


def load_road_network(gpkg_path: Path):
    logger.info("加载路网...")
    nodes = gpd.read_file(gpkg_path, layer="nodes").to_crs(PROJECTED_CRS)
    edges = gpd.read_file(gpkg_path, layer="edges").to_crs(PROJECTED_CRS)

    source_col = "u" if "u" in edges.columns else "source"
    target_col = "v" if "v" in edges.columns else "target"
    logger.info(f"构建 NetworkX 图 (u={source_col}, v={target_col})...")
    G = nx.from_pandas_edgelist(edges, source=source_col, target=target_col,
                                create_using=nx.MultiGraph())

    node_id_col = "osmid" if "osmid" in nodes.columns else "id"
    logger.info("注入坐标到 NetworkX...")
    for _, row in tqdm(nodes.iterrows(), total=len(nodes), desc="注入节点坐标", leave=False):
        nid = row[node_id_col]
        if nid in G.nodes:
            G.nodes[nid]["x"] = row.geometry.x
            G.nodes[nid]["y"] = row.geometry.y

    logger.info(f"路网加载完成: {G.number_of_nodes():,} 节点, {G.number_of_edges():,} 边")
    return G, nodes, edges, source_col, target_col


# ------------------- 分配方向（支持 "123_行政区" 格式） -------------------
def assign_directions_to_cameras(cameras_gdf: gpd.GeoDataFrame,
                                 G: nx.MultiGraph,
                                 edges_gdf: gpd.GeoDataFrame,
                                 source_col: str,
                                 target_col: str) -> gpd.GeoDataFrame:
    logger.info("开始为相机分配方向（投影后米级匹配）...")
    edges_gdf = edges_gdf.copy().dropna(subset=['geometry']).reset_index(drop=True)

    results = []
    for district in cameras_gdf['district'].unique():
        cams_in_dist = cameras_gdf[cameras_gdf['district'] == district].copy()
        cams_in_dist = cams_in_dist.sample(frac=1, random_state=42).reset_index(drop=True)

        nearest = gpd.sjoin_nearest(
            cams_in_dist,
            edges_gdf,
            how='left',
            max_distance=50,
            distance_col='dist_to_edge',
            lsuffix='cam',
            rsuffix='edge'
        )

        nearest = nearest.drop_duplicates(subset=['camera_id']).reset_index(drop=True)

        # 动态列名映射
        name_col = 'name_cam' if 'name_cam' in nearest.columns else 'name'
        cam_id_col = 'camera_id_cam' if 'camera_id_cam' in nearest.columns else 'camera_id'
        u_col = f'{source_col}_edge' if f'{source_col}_edge' in nearest.columns else source_col
        v_col = f'{target_col}_edge' if f'{target_col}_edge' in nearest.columns else target_col

        if len(results) == 0:
            logger.info(f"列名映射: name='{name_col}', id='{cam_id_col}', u='{u_col}', v='{v_col}'")

        forward = True
        for i in range(len(nearest)):
            row = nearest.iloc[i]

            try:
                cam_id = row[cam_id_col]
                name   = row[name_col] if name_col in nearest.columns else "未知"
                lat    = row.geometry.y
                lon    = row.geometry.x
                u_raw  = row[u_col]
                v_raw  = row[v_col]
                dist   = row['dist_to_edge'] if 'dist_to_edge' in nearest.columns else None
            except Exception as e:
                logger.warning(f"跳过第 {i} 行: {e}")
                continue

            # 关键修复：提取 _ 前面的数字
            try:
                u = int(str(u_raw).split('_')[0])
                v = int(str(v_raw).split('_')[0])
            except (ValueError, TypeError, IndexError):
                logger.warning(f"无效节点ID: u={u_raw}, v={v_raw}，跳过")
                u_dir = v_dir = None
                desc = "无效节点"
            else:
                if forward:
                    u_dir, v_dir = u, v
                    desc = f"{u} to {v}"
                else:
                    u_dir, v_dir = v, u
                    desc = f"{v} to {u}"
                forward = not forward

            results.append({
                'camera_id'    : cam_id,
                'name'         : name,
                'latitude'     : lat,
                'longitude'    : lon,
                'district'     : district,
                'direction_u'  : u_dir,
                'direction_v'  : v_dir,
                'direction_desc': desc,
                'dist_to_edge' : dist
            })

    result_df = pd.DataFrame(results)
    result_gdf = gpd.GeoDataFrame(
        result_df,
        geometry=gpd.points_from_xy(result_df.longitude, result_df.latitude),
        crs=PROJECTED_CRS
    ).to_crs("EPSG:4326")
    return result_gdf


# ------------------- Folium 地图 -------------------
def draw_folium_map(districts_gdf: gpd.GeoDataFrame,
                    cameras_with_dir_gdf: gpd.GeoDataFrame,
                    output_html: Path,
                    G: nx.MultiGraph):
    logger.info("绘制 Folium 地图...")
    center_lat = cameras_with_dir_gdf['latitude'].mean()
    center_lon = cameras_with_dir_gdf['longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11,
                   tiles='CartoDB positron')

    # 行政区边界
    for _, row in districts_gdf.to_crs("EPSG:4326").iterrows():
        folium.GeoJson(
            row['geometry'],
            style_function=lambda x: {
                'fillColor': 'transparent',
                'color': 'black',
                'weight': 2,
                'dashArray': '5,5'
            },
            tooltip=f"<b>{row['区域名称']}</b>"
        ).add_to(m)

    # 相机点 + 方向箭头
    transformer = pyproj.Transformer.from_crs(PROJECTED_CRS, "EPSG:4326", always_xy=True)

    for _, row in cameras_with_dir_gdf.iterrows():
        u, v = row['direction_u'], row['direction_v']
        popup = f"<b>{row['name']}</b><br>ID: {row['camera_id']}<br>方向: {row['direction_desc']}"

        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color='blue',
            fill=True,
            fillOpacity=0.8,
            popup=popup,
            tooltip=row['direction_desc']
        ).add_to(m)

        if pd.notna(u) and pd.notna(v) and u in G.nodes and v in G.nodes:
            try:
                ux, uy = G.nodes[u]['x'], G.nodes[u]['y']
                vx, vy = G.nodes[v]['x'], G.nodes[v]['y']
                ux_wgs, uy_wgs = transformer.transform(ux, uy)
                vx_wgs, vy_wgs = transformer.transform(vx, vy)

                dx = vx_wgs - ux_wgs
                dy = vy_wgs - uy_wgs
                length = np.hypot(dx, dy)
                if length > 1e-8:
                    dx, dy = dx / length, dy / length
                    s_lat = row['latitude'] + 0.00003 * dy
                    s_lon = row['longitude'] + 0.00003 * dx
                    e_lat = s_lat + 0.00012 * dy
                    e_lon = s_lon + 0.00012 * dx
                    folium.PolyLine(
                        [[s_lat, s_lon], [e_lat, e_lon]],
                        color='red',
                        weight=3,
                        popup=popup,
                        tooltip=row['direction_desc']
                    ).add_to(m)
            except Exception:
                pass

    m.save(output_html)
    logger.info(f"地图已保存: {output_html}")


# ------------------- 主程序 -------------------
if __name__ == "__main__":
    districts_gdf = load_districts(DISTRICTS_CSV)
    cameras_gdf   = load_cameras(CAMERAS_XLSX)
    logger.info(f"相机原始列: {list(cameras_gdf.columns)}")

    G, nodes, edges, source_col, target_col = load_road_network(ROAD_GPKG)
    logger.info(f"路网 u/v 列: {source_col}/{target_col}")

    cameras_with_dir = assign_directions_to_cameras(
        cameras_gdf, G, edges, source_col, target_col
    )

    # 输出 Excel
    out_df = cameras_with_dir.drop(columns=['geometry'])
    out_df.to_excel(OUTPUT_XLSX, index=False)
    logger.info(f"已保存方向文件: {OUTPUT_XLSX}")

    # 绘图
    draw_folium_map(districts_gdf, cameras_with_dir, OUTPUT_HTML, G)