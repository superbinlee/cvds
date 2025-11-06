# -*- coding: utf-8 -*-
"""
修复：Polygon = 0 → 正常生成
合理密度 + 智能命名 + Excel + 全量地图
"""

import os
import pandas as pd
import geopandas as gpd
import numpy as np
import folium
from loguru import logger
from tqdm import tqdm
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.ops import unary_union

# ------------------- 配置 -------------------
INPUT_DIR = "./poi"
EXCEL_OUTPUT = "cameras.xlsx"
OUTPUT_MAP = "camera_placement_full.html"

TILES = "CartoDB Positron"
TILES_URL = {
    "CartoDB Positron": "https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png",
    "OpenStreetMap": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
}
ATTR = {
    "CartoDB Positron": '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="http://cartodb.com/attributions">CartoDB</a>',
    "OpenStreetMap": '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
}


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
    """统一采样边界点（支持 Polygon 和 MultiPolygon）"""
    if geom.is_empty:
        return []

    # 获取所有外边界
    if isinstance(geom, Polygon):
        boundaries = [geom.exterior]
    elif isinstance(geom, MultiPolygon):
        boundaries = [poly.exterior for poly in geom.geoms if not poly.exterior.is_empty]
    else:
        return []

    pts = []
    total_length = sum(b.length for b in boundaries)
    if total_length == 0:
        return []

    # 按长度比例分配点数
    for boundary in boundaries:
        if boundary.length == 0:
            continue
        ratio = boundary.length / total_length
        n = max(2, int(target * ratio))
        ds = np.linspace(0, boundary.length, n + 2)[1:-1]
        pts.extend([boundary.interpolate(d) for d in ds])

    # 补齐或裁剪
    if len(pts) > target:
        indices = np.random.choice(len(pts), target, replace=False)
        pts = [pts[i] for i in indices]
    return pts


def generate_cameras_to_excel(gdf: gpd.GeoDataFrame):
    records = []
    camera_id = 0
    stats = {"Point": 0, "LineString": 0, "Polygon": 0, "MultiPolygon": 0}

    logger.info(f"开始生成相机，POI 总数: {len(gdf)}")

    for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="生成相机"):
        geom = row.geometry
        district = row.get("district", "未知")
        src_type = geom.geom_type
        name = smart_fill_name(row, idx, camera_id)

        # === Point ===
        if isinstance(geom, Point):
            records.append({
                "camera_id": f"C{camera_id:06d}",
                "name": name,
                "district": district,
                "source_type": src_type,
                "latitude": geom.y,
                "longitude": geom.x
            })
            stats["Point"] += 1
            camera_id += 1
            continue

        # === LineString ===
        if isinstance(geom, LineString):
            length_m = geom.length * 111320
            spacing = 150
            num = max(2, int(np.ceil(length_m / spacing)))
            distances = np.linspace(0, geom.length, num + 1)[1:-1]
            for d in distances:
                pt = geom.interpolate(d)
                records.append({
                    "camera_id": f"C{camera_id:06d}",
                    "name": name,
                    "district": district,
                    "source_type": src_type,
                    "latitude": pt.y,
                    "longitude": pt.x
                })
                camera_id += 1
            stats["LineString"] += len(distances)
            continue

        # === Polygon ===
        if isinstance(geom, Polygon):
            area_m2 = geom.area * 111320 ** 2
            area_km2 = area_m2 / 1e6
            target = get_camera_count(area_km2)
            pts = sample_boundary_points(geom, target)
            for pt in pts:
                records.append({
                    "camera_id": f"C{camera_id:06d}",
                    "name": name,
                    "district": district,
                    "source_type": "Polygon",
                    "latitude": pt.y,
                    "longitude": pt.x
                })
                camera_id += 1
            stats["Polygon"] += len(pts)
            continue

        # === MultiPolygon ===
        if isinstance(geom, MultiPolygon):
            area_m2 = geom.area * 111320 ** 2
            area_km2 = area_m2 / 1e6
            target = get_camera_count(area_km2)
            pts = sample_boundary_points(geom, target)
            for pt in pts:
                records.append({
                    "camera_id": f"C{camera_id:06d}",
                    "name": name,
                    "district": district,
                    "source_type": "MultiPolygon",
                    "latitude": pt.y,
                    "longitude": pt.x
                })
                camera_id += 1
            stats["MultiPolygon"] += len(pts)

    df_cams = pd.DataFrame(records)
    df_cams.to_excel(EXCEL_OUTPUT, index=False, engine='openpyxl')

    total = sum(stats.values())
    logger.info(f"相机生成完成！")
    logger.info(f"   Point: {stats['Point']}")
    logger.info(f"   LineString: {stats['LineString']}")
    logger.info(f"   Polygon: {stats['Polygon']}")
    logger.info(f"   MultiPolygon: {stats['MultiPolygon']}")
    logger.info(f"   总计: {total} 个相机")

    return df_cams


def visualize_with_stable_map():
    if not os.path.exists(EXCEL_OUTPUT):
        raise FileNotFoundError(f"未找到 {EXCEL_OUTPUT}")

    df = pd.read_excel(EXCEL_OUTPUT)
    center = [30.66, 104.06]
    m = folium.Map(location=center, zoom_start=11, tiles=None)

    folium.TileLayer(tiles=TILES_URL[TILES], attr=ATTR[TILES], name=TILES).add_to(m)
    folium.TileLayer(tiles=TILES_URL["OpenStreetMap"], attr=ATTR["OpenStreetMap"], name="OSM").add_to(m)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="添加相机"):
        popup = f"<b>{row['name']}</b><br>ID: {row['camera_id']}<br>区: {row['district']}<br>来源: {row['source_type']}"
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=3,
            color="red",
            fill=True,
            fillOpacity=0.8,
            popup=folium.Popup(popup, max_width=300)
        ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(OUTPUT_MAP)
    logger.info(f"地图已保存: {OUTPUT_MAP}")


if __name__ == "__main__":
    logger.info("=== 开始：修复 Polygon + 合理密度 ===")
    poi_gdf = load_all_pois()
    cam_df = generate_cameras_to_excel(poi_gdf)
    visualize_with_stable_map()
    logger.info("=== 完成 ===")
    print(f"\n输出：")
    print(f"   Excel: {EXCEL_OUTPUT}")
    print(f"   地图: {OUTPUT_MAP}")
    print(f"   相机总数: {len(cam_df)}")
