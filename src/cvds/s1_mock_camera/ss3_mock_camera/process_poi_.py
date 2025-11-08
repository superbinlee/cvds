# -*- coding: utf-8 -*-
"""
终极版：22万相机点流畅展示
使用 FastMarkerCluster + 智能命名 + Excel
"""

import os
import pandas as pd
import geopandas as gpd
import numpy as np
import folium
from folium.plugins import FastMarkerCluster
from loguru import logger
from tqdm import tqdm
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.ops import unary_union

# ------------------- 配置 -------------------
INPUT_DIR = "./poi"
EXCEL_OUTPUT = "cameras.xlsx"
OUTPUT_MAP = "camera_placement_220k.html"

TILES = "CartoDB Positron"
TILES_URL = "https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png"
TILES_ATTR = '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="http://cartodb.com/attributions">CartoDB</a>'


# ------------------------------------------------
def load_all_pois() -> gpd.GeoDataFrame:
    all_path = os.path.join(INPUT_DIR, "chengdu_all_pois.xlsx")
    if not os.path.exists(all_path):
        raise FileNotFoundError(f"未找到 {all_path}")
    logger.info(f"读取 POI 数据: {all_path}")
    df = pd.read_excel(all_path)
    return gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df["geometry"]), crs="EPSG:4326")


def smart_fill_name(row, idx, camera_id) -> str:
    name = row.get("name")
    if pd.notna(name) and str(name).strip():
        return str(name).strip()

    brand = row.get("brand")
    operator = row.get("operator")
    street = row.get("addr:street")
    category = row.get("category", "unknown")
    district = row.get("district", "未知")

    if pd.notna(brand) and pd.notna(operator):
        return f"{brand}_{operator}"
    if pd.notna(brand) and pd.notna(street):
        return f"{brand}_{street}"
    if pd.notna(operator) and pd.notna(street):
        return f"{operator}_{street}"
    if pd.notna(street):
        return f"{category}_{street}"
    if category != "unknown":
        return f"{category}_{district}"
    return f"{category}_{camera_id:06d}"


def get_camera_count(area_km2: float) -> int:
    if area_km2 < 0.01: return 4
    if area_km2 < 0.1: return 8
    if area_km2 < 1.0: return 16
    if area_km2 < 5.0: return 32
    return 64


def sample_boundary_points(geom, target: int):
    if geom.is_empty: return []
    if isinstance(geom, Polygon):
        boundaries = [geom.exterior]
    elif isinstance(geom, MultiPolygon):
        boundaries = [p.exterior for p in geom.geoms if not p.exterior.is_empty]
    else:
        return []

    pts = []
    total_length = sum(b.length for b in boundaries)
    if total_length == 0: return []

    for b in boundaries:
        if b.length == 0: continue
        ratio = b.length / total_length
        n = max(2, int(target * ratio))
        ds = np.linspace(0, b.length, n + 2)[1:-1]
        pts.extend([b.interpolate(d) for d in ds])

    if len(pts) > target:
        pts = [pts[i] for i in np.random.choice(len(pts), target, replace=False)]
    return pts


def generate_cameras_to_excel(gdf: gpd.GeoDataFrame):
    records = []
    camera_id = 0
    stats = {"Point": 0, "LineString": 0, "Polygon": 0, "MultiPolygon": 0}

    for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="生成相机"):
        geom = row.geometry
        district = row.get("district", "未知")
        src_type = geom.geom_type
        name = smart_fill_name(row, idx, camera_id)

        if isinstance(geom, Point):
            records.append([geom.y, geom.x, name, f"C{camera_id:06d}", district, src_type])
            stats["Point"] += 1
            camera_id += 1
            continue

        if isinstance(geom, LineString):
            length_m = geom.length * 111320
            spacing = 150
            num = max(2, int(np.ceil(length_m / spacing)))
            ds = np.linspace(0, geom.length, num + 1)[1:-1]
            for d in ds:
                pt = geom.interpolate(d)
                records.append([pt.y, pt.x, name, f"C{camera_id:06d}", district, src_type])
                camera_id += 1
            stats["LineString"] += len(ds)
            continue

        # Polygon & MultiPolygon
        area_m2 = geom.area * 111320 ** 2
        area_km2 = area_m2 / 1e6
        target = get_camera_count(area_km2)
        pts = sample_boundary_points(geom, target)

        for pt in pts:
            records.append([pt.y, pt.x, name, f"C{camera_id:06d}", district, geom.geom_type])
            camera_id += 1
        stats[geom.geom_type] += len(pts)

    df = pd.DataFrame(records, columns=["latitude", "longitude", "name", "camera_id", "district", "source_type"])
    df.to_excel(EXCEL_OUTPUT, index=False, engine='openpyxl')

    total = len(df)
    logger.info(f"相机生成完成！总计: {total} 个")
    logger.info(f"   Point: {stats['Point']} | Line: {stats['LineString']} | Polygon: {stats['Polygon']} | Multi: {stats['MultiPolygon']}")
    return df


def visualize_220k_cameras_with_cluster(df: pd.DataFrame):
    center = [30.66, 104.06]
    m = folium.Map(location=center, zoom_start=11, tiles=None)

    # 底图
    folium.TileLayer(tiles=TILES_URL, attr=TILES_ATTR, name="CartoDB").add_to(m)
    folium.TileLayer("OpenStreetMap").add_to(m)

    # FastMarkerCluster 回调函数
    callback = """
    function (row) {
        var marker = L.circleMarker(new L.LatLng(row[0], row[1]), {
            radius: 4,
            color: 'red',
            fillOpacity: 0.8
        });
        marker.bindPopup(
            "<b>" + row[2] + "</b><br>" +
            "ID: " + row[3] + "<br>" +
            "区: " + row[4] + "<br>" +
            "来源: " + row[5]
        );
        return marker;
    };
    """

    FastMarkerCluster(
        data=df[["latitude", "longitude", "name", "camera_id", "district", "source_type"]].values.tolist(),
        callback=callback
    ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(OUTPUT_MAP)
    logger.info(f"22万相机地图已保存: {OUTPUT_MAP}（使用聚类，缩放流畅！）")


if __name__ == "__main__":
    logger.info("=== 开始：22万相机流畅展示 ===")
    poi_gdf = load_all_pois()
    cam_df = generate_cameras_to_excel(poi_gdf)
    visualize_220k_cameras_with_cluster(cam_df)
    logger.info("=== 完成 ===")
    print(f"\n输出：")
    print(f"   Excel: {EXCEL_OUTPUT} ({len(cam_df)} 行)")
    print(f"   地图: {OUTPUT_MAP}（双击打开，缩放即显示）")
