import folium
import geopandas as gpd
import pandas as pd
from shapely import wkt
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, MultiPoint, MultiLineString
from loguru import logger
import os
from pathlib import Path
import warnings

# 屏蔽 folium 警告
warnings.filterwarnings("ignore", category=UserWarning, module="folium")


def parse_wkt_geometry(wkt_str):
    try:
        if isinstance(wkt_str, str):
            return wkt.loads(wkt_str.strip())
        return None
    except:
        return None


def infer_district(row):
    text = f"{row.get('name', '')} {row.get('addr:street', '')} {row.get('addr:city', '')}"
    districts = ['成华区', '郫都区', '新都区', '金牛区', '青羊区', '温江区', '锦江区', '武侯区', '龙泉驿区']
    for d in districts:
        if d in text:
            return d
    return "未知区"


def get_map_center(gdf: gpd.GeoDataFrame):
    if gdf.empty or gdf.geometry.isna().all():
        return [30.65, 104.06]
    try:
        utm_crs = gdf.estimate_utm_crs() if hasattr(gdf, 'estimate_utm_crs') else "EPSG:32648"
        centroid = gdf.to_crs(utm_crs).geometry.centroid.union_all().centroid
        center = gpd.GeoSeries([centroid], crs=utm_crs).to_crs("EPSG:4326")
        return [center.y.iloc[0], center.x.iloc[0]]
    except:
        b = gdf.total_bounds
        return [(b[1] + b[3]) / 2, (b[0] + b[2]) / 2]


def add_poi_to_map(poi, group, color_map):
    name = poi.get('name', '未知')
    cat = str(poi.get('category', 'unknown')).strip() or 'unknown'
    popup_text = f"<b>{name}</b><br>类型: {cat}"
    color = color_map.get(cat, '#999999')
    geom = poi.geometry
    if not geom or geom.is_empty:
        return

    if isinstance(geom, Point):
        folium.CircleMarker(
            location=[geom.y, geom.x], radius=5, color=color, fill=True, fillOpacity=0.7,
            popup=folium.Popup(popup_text, max_width=300), tooltip=name
        ).add_to(group)

    elif isinstance(geom, LineString):
        coords = [(y, x) for x, y in geom.coords]
        folium.PolyLine(locations=coords, color=color, weight=3).add_to(group)

    elif isinstance(geom, (Polygon, MultiPolygon)):
        if geom.geom_type == 'GeometryCollection':
            geom = [g for g in geom.geoms if g.geom_type in ['Polygon', 'MultiPolygon']]
            if not geom: return
            geom = geom[0] if len(geom) == 1 else MultiPolygon(geom)
        folium.GeoJson(
            gpd.GeoSeries([geom]).__geo_interface__,
            style_function=lambda x, c=color: {'fillColor': c, 'color': c, 'weight': 2, 'fillOpacity': 0.4},
            popup=folium.Popup(popup_text, max_width=300)
        ).add_to(group)

    elif isinstance(geom, MultiPoint):
        for p in geom.geoms:
            folium.CircleMarker(location=[p.y, p.x], radius=4, color=color, fill=True).add_to(group)

    elif isinstance(geom, MultiLineString):
        for line in geom.geoms:
            coords = [(y, x) for x, y in line.coords]
            folium.PolyLine(locations=coords, color=color, weight=3).add_to(group)


def create_district_maps(pois_df: pd.DataFrame, output_dir: str = "./vis/maps"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 解析 WKT
    if isinstance(pois_df['geometry'].iloc[0], str):
        logger.info("正在解析 WKT...")
        pois_df['geometry'] = pois_df['geometry'].apply(parse_wkt_geometry)
        pois_df = pois_df.dropna(subset=['geometry']).copy()

    pois = gpd.GeoDataFrame(pois_df, geometry='geometry', crs="EPSG:4326")
    if pois.empty:
        logger.error("无有效 POI")
        return

    # 自动添加 district
    if 'district' not in pois.columns:
        logger.warning("无 district 字段，自动推断...")
        pois['district'] = pois.apply(infer_district, axis=1)
    else:
        pois['district'] = pois['district'].fillna('未知区').astype(str)

    districts = pois['district'].unique()
    logger.info(f"识别区划: {list(districts)}")

    # 配色
    color_map = {
        'park': '#1a9850', 'playground': '#91cf60', 'swimming_pool': '#1f78b4',
        'sauna': '#e31a1c', 'resort': '#ff7f00', 'pitch': '#6a3d9a',
        'marina': '#33a02c', 'fitness_station': '#b15928',
        'hotel': '#e7298a', 'hostel': '#66a61e', 'supermarket': '#e6ab02',
        'attraction': '#7570b3', 'yes': '#1b9e77', 'unknown': '#999999'
    }

    # ================== 1. 分区地图 ==================
    for d in districts:
        sub = pois[pois['district'] == d].copy()
        if sub.empty: continue

        m = folium.Map(location=get_map_center(sub), zoom_start=13, tiles="CartoDB positron")
        cats = sub['category'].fillna('unknown').unique().tolist()
        if 'unknown' not in cats: cats.append('unknown')

        groups = {}
        for c in cats:
            cnt = len(sub[sub['category'].fillna('unknown') == c])
            fg = folium.FeatureGroup(name=f"{c} ({cnt})")
            groups[c] = fg
            m.add_child(fg)

        for _, p in sub.iterrows():
            c = p.get('category', 'unknown') or 'unknown'
            add_poi_to_map(p, groups[c], color_map)

        folium.LayerControl(collapsed=False).add_to(m)
        m.save(output_path / f"map_{d}_pois.html")
        logger.success(f"已保存: map_{d}_pois.html")

    # ================== 2. 全域总图 ==================
    logger.info("生成全域总图...")
    m_all = folium.Map(location=get_map_center(pois), zoom_start=11, tiles="CartoDB positron")

    all_cats = pois['category'].fillna('unknown').unique().tolist()
    if 'unknown' not in all_cats: all_cats.append('unknown')

    groups_all = {}
    for c in all_cats:
        cnt = len(pois[pois['category'].fillna('unknown') == c])
        fg = folium.FeatureGroup(name=f"{c} ({cnt})")
        groups_all[c] = fg
        m_all.add_child(fg)

    for _, p in pois.iterrows():
        c = p.get('category', 'unknown') or 'unknown'
        add_poi_to_map(p, groups_all[c], color_map)

    # 关键修复：不使用 GeoJsonTooltip，避免字段检查
    if len(districts) > 1:
        try:
            boundaries = pois.dissolve(by='district')[['geometry']]  # 只保留 geometry
            geojson = boundaries.__geo_interface__

            # 手动添加 popup（不使用 tooltip）
            for idx, (district_name, row) in enumerate(boundaries.iterrows()):
                popup = folium.Popup(f"<b>行政区：</b>{district_name}", max_width=200)
                folium.GeoJson(
                    data={"type": "Feature", "geometry": row.geometry.__geo_interface__},
                    style_function=lambda x: {'color': '#333', 'weight': 3, 'fillOpacity': 0},
                    name="行政区边界"
                ).add_child(popup).add_to(m_all)
        except Exception as e:
            logger.debug(f"边界绘制失败（已跳过）: {e}")

    folium.LayerControl(collapsed=False).add_to(m_all)
    all_map_file = output_path / "map_Chengdu_ALL_pois.html"
    m_all.save(str(all_map_file))
    logger.success(f"全域地图已保存: {all_map_file.name}")

    logger.info("所有地图生成完成！")


# ================== 主程序 ==================
if __name__ == "__main__":
    try:
        POI_FILE = "./vis/chengdu_all_pois.xlsx"
        OUTPUT_DIR = "./vis/maps"

        if not os.path.exists(POI_FILE):
            logger.error(f"文件不存在: {POI_FILE}")
            exit(1)

        logger.info(f"读取 POI ← {POI_FILE}")
        df = pd.read_excel(POI_FILE)

        if 'geometry' not in df.columns:
            logger.error("缺少 geometry 字段")
            exit(1)

        create_district_maps(df, OUTPUT_DIR)
        logger.info("任务完成！请查看 ./vis/maps/")

    except Exception as e:
        logger.error(f"程序异常: {e}")
        raise
