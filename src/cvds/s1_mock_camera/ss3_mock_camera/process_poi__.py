# -*- coding: utf-8 -*-
"""
修复版：支持 MultiPolygon + 智能剔除楼栋 + 仅小区外边界相机
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
EXCEL_OUTPUT = "cameras_boundary_smart.xlsx"
OUTPUT_MAP = "camera_boundary_smart.html"

BUILDING_AREA_THRESHOLD = 0.05  # km²
MIN_PERIMETER_M = 100

TILES_URL = "https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png"
TILES_ATTR = '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="http://cartodb.com/attributions">CartoDB</a>'


# ------------------------------------------------
def load_all_pois() -> gpd.GeoDataFrame:
    all_path = os.path.join(INPUT_DIR, "chengdu_all_pois.xlsx")
    if not os.path.exists(all_path):
        raise FileNotFoundError(f"未找到 {all_path}")
    logger.info(f"读取 POI 数据: {all_path}")
    df = pd.read_excel(all_path)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df["geometry"]), crs="EPSG:4326")
    logger.info(f"原始 POI 总数: {len(gdf)}")
    return gdf


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


def is_building(geom) -> bool:
    if not isinstance(geom, (Polygon, MultiPolygon)):
        return False
    area_m2 = geom.area * 111320 ** 2
    return area_m2 / 1e6 < BUILDING_AREA_THRESHOLD


def get_outer_boundary_clean(gdf_polygons) -> list:
    valid_polys = []
    for geom in gdf_polygons.geometry:
        if is_building(geom):
            continue
        if isinstance(geom, Polygon):
            valid_polys.append(geom)
        elif isinstance(geom, MultiPolygon):
            valid_polys.extend(geom.geoms)

    if not valid_polys:
        return []

    temp_gdf = gpd.GeoDataFrame(geometry=valid_polys, crs="EPSG:4326")
    sindex = temp_gdf.sindex
    outer_boundaries = []

    for idx, poly in enumerate(valid_polys):
        possible_containing = list(sindex.query(poly, predicate="contains"))
        is_inner = any(
            poly.within(valid_polys[i])
            for i in possible_containing
            if i != idx
        )
        if not is_inner:
            outer_boundaries.append(poly.exterior)

    if not outer_boundaries:
        return []

    merged = unary_union(outer_boundaries)
    if merged.geom_type == "LineString":
        return [merged] if merged.length * 111320 >= MIN_PERIMETER_M else []
    elif merged.geom_type == "MultiLineString":
        return [line for line in merged.geoms if line.length * 111320 >= MIN_PERIMETER_M]
    return []


def get_boundary_camera_count(perimeter_m: float) -> int:
    if perimeter_m < 200: return 4
    if perimeter_m < 500: return 8
    if perimeter_m < 1000: return 12
    if perimeter_m < 2000: return 16
    return 20


def generate_smart_boundary_cameras(gdf: gpd.GeoDataFrame):
    records = []
    camera_id = 0
    stats = {"Point": 0, "LineString": 0, "Boundary": 0, "Building_Discarded": 0}

    gdf_point = gdf[gdf.geometry.geom_type == "Point"]
    gdf_line = gdf[gdf.geometry.geom_type == "LineString"]
    gdf_poly = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]

    # 统计丢弃楼栋
    discarded = sum(1 for _, r in gdf_poly.iterrows() if is_building(r.geometry))
    stats["Building_Discarded"] = discarded
    logger.info(f"识别并丢弃楼栋（面积 < {BUILDING_AREA_THRESHOLD} km²）: {discarded} 个")

    # Point
    for _, row in tqdm(gdf_point.iterrows(), total=len(gdf_point), desc="处理 Point"):
        pt = row.geometry
        records.append([pt.y, pt.x, smart_fill_name(row, 0, camera_id), f"C{camera_id:06d}", row.get("district", "未知"), "Point"])
        stats["Point"] += 1
        camera_id += 1

    # LineString
    for _, row in tqdm(gdf_line.iterrows(), total=len(gdf_line), desc="处理 LineString"):
        line = row.geometry
        length_m = line.length * 111320
        spacing = 150
        num = max(2, int(np.ceil(length_m / spacing)))
        ds = np.linspace(0, line.length, num + 1)[1:-1]
        for d in ds:
            pt = line.interpolate(d)
            records.append([pt.y, pt.x, smart_fill_name(row, 0, camera_id), f"C{camera_id:06d}", row.get("district", "未知"), "LineString"])
            camera_id += 1
        stats["LineString"] += len(ds)

    # Polygon 外边界
    if len(gdf_poly) > 0:
        logger.info("提取小区外边界中...")
        outer_lines = get_outer_boundary_clean(gdf_poly)
        logger.info(f"提取到 {len(outer_lines)} 条有效外边界")

        for line in tqdm(outer_lines, desc="部署边界相机"):
            perimeter_m = line.length * 111320
            target = get_boundary_camera_count(perimeter_m)
            ds = np.linspace(0, line.length, target + 1)[1:-1]
            for d in ds:
                pt = line.interpolate(d)
                records.append([pt.y, pt.x, "小区边界", f"C{camera_id:06d}", "未知", "Boundary"])
                camera_id += 1
            stats["Boundary"] += len(ds)

    df = pd.DataFrame(records, columns=["latitude", "longitude", "name", "camera_id", "district", "source_type"])
    df.to_excel(EXCEL_OUTPUT, index=False, engine='openpyxl')

    total = len(df)
    logger.info(f"智能相机生成完成！总计: {total} 个")
    logger.info(f"   Point: {stats['Point']} | Line: {stats['LineString']} | 边界: {stats['Boundary']} | 丢弃楼栋: {stats['Building_Discarded']}")
    return df


def visualize_with_cluster(df: pd.DataFrame):
    center = [30.66, 104.06]
    m = folium.Map(location=center, zoom_start=11, tiles=None)
    folium.TileLayer(tiles=TILES_URL, attr=TILES_ATTR, name="CartoDB").add_to(m)
    folium.TileLayer("OpenStreetMap").add_to(m)

    callback = """
    function (row) {
        var color = row[5] === 'Boundary' ? 'blue' : 
                    row[5] === 'Point' ? 'red' : 'green';
        var marker = L.circleMarker(new L.LatLng(row[0], row[1]), {
            radius: 4, color: color, fillOpacity: 0.8
        });
        marker.bindPopup(
            "<b>" + row[2] + "</b><br>ID: " + row[3] + "<br>区: " + row[4] + "<br>类型: " + row[5]
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
    logger.info(f"地图已保存: {OUTPUT_MAP}")


if __name__ == "__main__":
    logger.info("=== 开始：智能剔除楼栋 + 仅小区外边界相机 ===")
    poi_gdf = load_all_pois()
    cam_df = generate_smart_boundary_cameras(poi_gdf)
    visualize_with_cluster(cam_df)
    logger.info("=== 完成 ===")
    print(f"\n输出：")
    print(f"   Excel: {EXCEL_OUTPUT}")
    print(f"   地图: {OUTPUT_MAP}")
    print(f"   相机总数: {len(cam_df)}")
