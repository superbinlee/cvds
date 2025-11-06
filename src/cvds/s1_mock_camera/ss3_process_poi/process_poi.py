# -*- coding: utf-8 -*-
"""
1. POI → Excel
2. Excel → 全量相机点（无聚类）
3. 强制使用稳定底图（CartoDB Positron）
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

# 稳定底图（推荐）
TILES = "CartoDB Positron"  # 或者 "OpenStreetMap"
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


def generate_cameras_to_excel(gdf: gpd.GeoDataFrame):
    records = []
    camera_id = 0

    for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="生成相机"):
        geom = row.geometry
        district = row.get("district", "未知")
        name = row.get("name", f"{row.get('category','unknown')}_{idx}")
        src_type = geom.geom_type

        if isinstance(geom, Point):
            records.append({
                "camera_id": f"C{camera_id:06d}",
                "name": name,
                "district": district,
                "source_type": src_type,
                "latitude": geom.y,
                "longitude": geom.x
            })
            camera_id += 1

        elif isinstance(geom, LineString):
            length_m = geom.length * 111320
            spacing = 200
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

        elif isinstance(geom, (Polygon, MultiPolygon)):
            area_m2 = geom.area * 111320 ** 2
            area_km2 = area_m2 / 1e6
            target = 4 if area_km2 < 0.5 else 8 if area_km2 < 2 else 12

            boundary = geom.exterior if isinstance(geom, Polygon) else unary_union([p.exterior for p in geom.geoms])
            pts = []

            def sample_line(line, n):
                if line.length == 0: return []
                ds = np.linspace(0, line.length, n + 2)[1:-1]
                return [line.interpolate(d) for d in ds]

            if boundary.geom_type == "LineString":
                pts = sample_line(boundary, target * 3)
            elif boundary.geom_type == "MultiLineString":
                per_line = max(1, target // len(boundary.geoms))
                for line in boundary.geoms:
                    pts.extend(sample_line(line, per_line))

            if len(pts) > target:
                pts = [pts[i] for i in np.random.choice(len(pts), target, replace=False)]
            else:
                pts = pts[:target]

            for pt in pts:
                records.append({
                    "camera_id": f"C{camera_id:06d}",
                    "name": name,
                    "district": district,
                    "source_type": src_type,
                    "latitude": pt.y,
                    "longitude": pt.x
                })
                camera_id += 1

    df_cams = pd.DataFrame(records)
    df_cams.to_excel(EXCEL_OUTPUT, index=False, engine='openpyxl')
    logger.info(f"相机已保存为 Excel: {EXCEL_OUTPUT}（共 {len(df_cams)} 个）")
    return df_cams


def visualize_with_stable_map():
    if not os.path.exists(EXCEL_OUTPUT):
        raise FileNotFoundError(f"未找到 {EXCEL_OUTPUT}")

    logger.info(f"从 Excel 加载相机: {EXCEL_OUTPUT}")
    df = pd.read_excel(EXCEL_OUTPUT)

    center = [30.66, 104.06]
    m = folium.Map(
        location=center,
        zoom_start=11,
        tiles=None  # 先不加载默认
    )

    # 添加稳定底图
    folium.TileLayer(
        tiles=TILES_URL[TILES],
        attr=ATTR[TILES],
        name=TILES,
        overlay=False,
        control=True
    ).add_to(m)

    # 添加备用底图（OpenStreetMap）
    folium.TileLayer(
        tiles=TILES_URL["OpenStreetMap"],
        attr=ATTR["OpenStreetMap"],
        name="OpenStreetMap",
        overlay=False,
        control=True
    ).add_to(m)

    # 全量添加相机点（无聚类）
    for _, row in tqdm(df.iterrows(), total=len(df), desc="添加相机"):
        popup_html = f"""
        <b>{row['name']}</b><br>
        ID: {row['camera_id']}<br>
        区: {row['district']}<br>
        来源: {row['source_type']}
        """
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=4,
            color="red",
            fill=True,
            fillOpacity=0.9,
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(OUTPUT_MAP)
    logger.info(f"地图已保存: {OUTPUT_MAP}（底图：{TILES} + 备用，相机：{len(df)} 个）")


if __name__ == "__main__":
    logger.info("=== 开始处理 ===")
    poi_gdf = load_all_pois()
    cam_df = generate_cameras_to_excel(poi_gdf)
    visualize_with_stable_map()
    logger.info("=== 完成 ===")
    print(f"\n输出：")
    print(f"   Excel: {EXCEL_OUTPUT}")
    print(f"   地图: {OUTPUT_MAP}（双击打开，切换底图可测试）")