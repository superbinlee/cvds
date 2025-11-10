# -*- coding: utf-8 -*-
"""
成都摄像头智能部署系统（最终完整版）
功能：
1. 智能剔除楼栋 + 仅外边界部署
2. 每 500 米 1 个摄像头（最小 1 个）
3. 仅保留行政区内点
4. 行政区边界绘制在地图上
5. 输出 Excel + 交互地图（聚类 + 图例）
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
DISTRICTS_CSV = "./districts/chengdu_districts_boundary.csv"
EXCEL_OUTPUT = "cameras_boundary_final.xlsx"
OUTPUT_MAP = "camera_boundary_final.html"

BUILDING_AREA_THRESHOLD = 0.05  # km²，小于该值视为楼栋
MIN_PERIMETER_M = 100  # 边界最小周长（米）
BOUNDARY_SPACING_M = 500  # 每 500 米 1 个摄像头
MIN_CAMERAS_PER_BOUNDARY = 1  # 最小部署 1 个

TILES_URL = "https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png"
TILES_ATTR = '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="http://cartodb.com/attributions">CartoDB</a>'


# ------------------- 加载行政区边界 -------------------
def load_districts() -> gpd.GeoDataFrame:
    if not os.path.exists(DISTRICTS_CSV):
        raise FileNotFoundError(f"未找到行政区文件: {DISTRICTS_CSV}")
    logger.info(f"读取行政区边界: {DISTRICTS_CSV}")
    df = pd.read_csv(DISTRICTS_CSV)
    df.columns = [c.strip() for c in df.columns]
    if '区域名称' not in df.columns or '区域边界' not in df.columns:
        raise KeyError("CSV 必须包含列: '区域名称', '区域边界'")
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.GeoSeries.from_wkt(df["区域边界"]),
        crs="EPSG:4326"
    )
    gdf = gdf[['区域名称', 'geometry']].rename(columns={'区域名称': 'district'})
    logger.info(f"成功加载行政区: {len(gdf)} 个")
    return gdf


# ------------------- 加载 POI -------------------
def load_all_pois() -> gpd.GeoDataFrame:
    all_path = os.path.join(INPUT_DIR, "chengdu_all_pois.xlsx")
    if not os.path.exists(all_path):
        raise FileNotFoundError(f"未找到 POI 文件: {all_path}")
    logger.info(f"读取 POI 数据: {all_path}")
    df = pd.read_excel(all_path)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df["geometry"]), crs="EPSG:4326")
    logger.info(f"原始 POI 总数: {len(gdf)}")
    return gdf


# ------------------- 智能名称填充 -------------------
def smart_fill_name(row, idx, camera_id) -> str:
    name = row.get("name")
    if pd.notna(name) and str(name).strip():
        return str(name).strip()
    brand = row.get("brand")
    operator = row.get("operator")
    street = row.get("addr:street")
    category = row.get("category", "unknown")
    if pd.notna(brand) and pd.notna(operator):
        return f"{brand}_{operator}"
    if pd.notna(brand) and pd.notna(street):
        return f"{brand}_{street}"
    if pd.notna(operator) and pd.notna(street):
        return f"{operator}_{street}"
    if pd.notna(street):
        return f"{category}_{street}"
    if category != "unknown":
        return f"{category}_{camera_id:06d}"
    return f"camera_{camera_id:06d}"


# ------------------- 楼栋判断 -------------------
def is_building(geom) -> bool:
    if not isinstance(geom, (Polygon, MultiPolygon)):
        return False
    area_m2 = geom.area * 111320 ** 2
    return area_m2 / 1e6 < BUILDING_AREA_THRESHOLD


# ------------------- 提取最外边界 -------------------
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
            for i in possible_containing if i != idx
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


# ------------------- 边界布点数量 -------------------
def get_boundary_camera_count(perimeter_m: float) -> int:
    if perimeter_m < MIN_PERIMETER_M:
        return 0
    return max(MIN_CAMERAS_PER_BOUNDARY, int(np.ceil(perimeter_m / BOUNDARY_SPACING_M)))


# ------------------- 空间匹配：点 → 行政区 -------------------
def get_district_name(point: Point, districts_gdf: gpd.GeoDataFrame) -> str:
    candidates = districts_gdf[districts_gdf.geometry.contains(point)]
    return candidates.iloc[0]['district'] if len(candidates) > 0 else "未知"


# ------------------- 主生成逻辑（仅行政区内） -------------------
def generate_smart_boundary_cameras(poi_gdf: gpd.GeoDataFrame, districts_gdf: gpd.GeoDataFrame):
    records = []
    camera_id = 0
    stats = {
        "Point": 0, "LineString": 0, "Boundary": 0,
        "Building_Discarded": 0, "Outside_District_Removed": 0
    }

    gdf_point = poi_gdf[poi_gdf.geometry.geom_type == "Point"]
    gdf_line = poi_gdf[poi_gdf.geometry.geom_type == "LineString"]
    gdf_poly = poi_gdf[poi_gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]

    # 统计丢弃楼栋
    discarded = sum(1 for _, r in gdf_poly.iterrows() if is_building(r.geometry))
    stats["Building_Discarded"] = discarded
    logger.info(f"识别并丢弃楼栋: {discarded} 个")

    # 构建行政区整体区域（用于裁剪）
    admin_union = unary_union(districts_gdf.geometry)
    admin_gdf = gpd.GeoDataFrame(geometry=[admin_union], crs="EPSG:4326")
    logger.info(f"行政区合并为 {admin_union.geom_type}，用于空间裁剪")

    def is_within_district(pt: Point) -> bool:
        return admin_gdf.contains(pt).iloc[0]

    # 处理 Point
    for _, row in tqdm(gdf_point.iterrows(), total=len(gdf_point), desc="处理 Point"):
        pt = row.geometry
        if not is_within_district(pt):
            stats["Outside_District_Removed"] += 1
            continue
        district = get_district_name(pt, districts_gdf)
        name = smart_fill_name(row, 0, camera_id)
        records.append([pt.y, pt.x, name, f"C{camera_id:06d}", district, "Point"])
        stats["Point"] += 1
        camera_id += 1

    # 处理 LineString
    for _, row in tqdm(gdf_line.iterrows(), total=len(gdf_line), desc="处理 LineString"):
        line = row.geometry
        length_m = line.length * 111320
        spacing = 150
        num = max(2, int(np.ceil(length_m / spacing)))
        ds = np.linspace(0, line.length, num + 1)[1:-1]
        added = 0
        for d in ds:
            pt = line.interpolate(d)
            if not is_within_district(pt):
                stats["Outside_District_Removed"] += 1
                continue
            district = get_district_name(pt, districts_gdf)
            name = smart_fill_name(row, 0, camera_id)
            records.append([pt.y, pt.x, name, f"C{camera_id:06d}", district, "LineString"])
            camera_id += 1
            added += 1
        stats["LineString"] += added

    # 处理外边界（每 500 米 1 个）
    if len(gdf_poly) > 0:
        logger.info("提取小区外边界中...")
        outer_lines = get_outer_boundary_clean(gdf_poly)
        logger.info(f"提取到 {len(outer_lines)} 条有效外边界")
        for line in tqdm(outer_lines, desc="部署边界相机"):
            perimeter_m = line.length * 111320
            target = get_boundary_camera_count(perimeter_m)
            if target == 0:
                continue
            ds = np.linspace(0, line.length, target + 1)[1:-1]
            for d in ds:
                pt = line.interpolate(d)
                if not is_within_district(pt):
                    stats["Outside_District_Removed"] += 1
                    continue
                district = get_district_name(pt, districts_gdf)
                records.append([pt.y, pt.x, "小区边界", f"C{camera_id:06d}", district, "Boundary"])
                camera_id += 1
            stats["Boundary"] += sum(1 for d in ds if is_within_district(line.interpolate(d)))

    df = pd.DataFrame(records, columns=["latitude", "longitude", "name", "camera_id", "district", "source_type"])
    df.to_excel(EXCEL_OUTPUT, index=False, engine='openpyxl')
    total = len(df)
    logger.info(f"智能相机生成完成！总计: {total} 个（裁剪后）")
    logger.info(f" Point: {stats['Point']} | Line: {stats['LineString']} | 边界: {stats['Boundary']} "
                f"| 楼栋丢弃: {stats['Building_Discarded']} | 区外移除: {stats['Outside_District_Removed']}")
    return df


# ------------------- 可视化：摄像头 + 行政区边界 -------------------
def visualize_with_cluster(df: pd.DataFrame, districts_gdf: gpd.GeoDataFrame):
    center = [30.66, 104.06]
    m = folium.Map(location=center, zoom_start=11, tiles=None)

    # 1. 添加底图
    folium.TileLayer(tiles=TILES_URL, attr=TILES_ATTR, name="CartoDB 浅色").add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)

    # 2. 绘制行政区边界
    district_layer = folium.FeatureGroup(name="行政区边界", show=True)
    for _, row in districts_gdf.iterrows():
        district_name = row['district']
        geom = row['geometry']
        if geom.geom_type == 'Polygon':
            polys = [geom]
        elif geom.geom_type == 'MultiPolygon':
            polys = geom.geoms
        else:
            continue
        for poly in polys:
            coords = list(poly.exterior.coords)
            folium.Polygon(
                locations=[[(lat, lon) for lon, lat in coords]],
                color="red",
                weight=2,
                fill=True,
                fillOpacity=0.05,
                popup=folium.Popup(f"<b>{district_name}</b>", parse_html=True),
                tooltip=district_name
            ).add_to(district_layer)
    district_layer.add_to(m)

    # 3. 摄像头点（聚类）
    callback = """
    function (row) {
        var color = row[5] === 'Boundary' ? 'blue' : 
                    row[5] === 'Point' ? 'red' : 'green';
        var marker = L.circleMarker(new L.LatLng(row[0], row[1]), {
            radius: 5, color: color, fillOpacity: 0.9, weight: 1
        });
        marker.bindPopup(
            "<b>" + row[2] + "</b><br>" +
            "ID: " + row[3] + "<br>" +
            "区: <b>" + row[4] + "</b><br>" +
            "类型: " + row[5]
        );
        return marker;
    };
    """
    FastMarkerCluster(
        data=df[["latitude", "longitude", "name", "camera_id", "district", "source_type"]].values.tolist(),
        callback=callback,
        name="摄像头点"
    ).add_to(m)

    # 4. 图层控制 + 图例
    folium.LayerControl(collapsed=False).add_to(m)

    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; width: 150px; height: 130px; 
                border:2px solid grey; z-index:9999; font-size:14px; background:white;
                padding: 10px; border-radius: 6px; opacity: 0.9;">
     <b>图例</b><br>
     <i class="fa fa-circle" style="color:red"></i> Point<br>
     <i class="fa fa-circle" style="color:green"></i> LineString<br>
     <i class="fa fa-circle" style="color:blue"></i> 边界<br>
     <i style="border: 1px solid red; display: inline-block; width: 12px; height: 12px;"></i> 行政区边界
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(OUTPUT_MAP)
    logger.info(f"地图已保存（含行政区边界）: {OUTPUT_MAP}")


# ------------------- 主程序 -------------------
if __name__ == "__main__":
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), colorize=True)

    logger.info("=== 开始：智能摄像头部署 + 行政区边界可视化 ===")

    # 1. 加载数据
    districts_gdf = load_districts()
    poi_gdf = load_all_pois()

    # 2. 生成摄像头
    cam_df = generate_smart_boundary_cameras(poi_gdf, districts_gdf)

    # 3. 可视化（含边界）
    visualize_with_cluster(cam_df, districts_gdf)

    logger.info("=== 全部完成 ===")
    print(f"\n输出文件：")
    print(f"   Excel: {EXCEL_OUTPUT}")
    print(f"   地图: {OUTPUT_MAP}")
    print(f"   摄像头总数: {len(cam_df)}")