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


def build_full_popup(row: pd.Series) -> str:
    """
    将 POI 的所有非空字段（除 geometry）渲染为 HTML 表格
    """
    ignore_cols = {"geometry", "index", "_id", "level_0", "level_1"}
    rows = []
    for col, val in row.items():
        if col in ignore_cols:
            continue
        if pd.isna(val):
            continue
        val_str = str(val).strip()
        if len(val_str) > 300:
            val_str = val_str[:297] + "..."
        # 转义 HTML 特殊字符
        val_str = val_str.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        rows.append(f"<tr><td style='padding:2px 8px;'><b>{col}</b></td>"
                    f"<td style='padding:2px 8px; max-width:400px; word-wrap:break-word;'>{val_str}</td></tr>")

    if not rows:
        return "<b>未知 POI</b>"

    table = f"""
    <div style="font-family: Arial, sans-serif; font-size: 13px; max-height: 400px; overflow-y: auto;">
        <table style="width:100%; border-collapse: collapse;">
            {''.join(rows)}
        </table>
    </div>
    """
    return table


def add_poi_to_map(poi, group, color_map):
    name = poi.get('name', '未知')
    cat = str(poi.get('category', 'unknown')).strip() or 'unknown'

    # 悬停提示：仅名称
    tooltip = name

    # 弹窗：全部字段
    popup_html = build_full_popup(poi)

    color = color_map.get(cat, '#999999')
    geom = poi.geometry
    if not geom or geom.is_empty:
        return

    # Point
    if isinstance(geom, Point):
        folium.CircleMarker(
            location=[geom.y, geom.x],
            radius=6,
            color=color,
            fill=True,
            fillOpacity=0.8,
            popup=folium.Popup(popup_html, max_width=600),
            tooltip=tooltip
        ).add_to(group)

    # LineString
    elif isinstance(geom, LineString):
        coords = [(y, x) for x, y in geom.coords]
        folium.PolyLine(
            locations=coords,
            color=color,
            weight=4,
            popup=folium.Popup(popup_html, max_width=600),
            tooltip=tooltip
        ).add_to(group)

    # Polygon / MultiPolygon
    elif isinstance(geom, (Polygon, MultiPolygon)):
        if geom.geom_type == 'GeometryCollection':
            geom = [g for g in geom.geoms if g.geom_type in ['Polygon', 'MultiPolygon']]
            if not geom:
                return
            geom = geom[0] if len(geom) == 1 else MultiPolygon(geom)

        folium.GeoJson(
            gpd.GeoSeries([geom]).__geo_interface__,
            style_function=lambda x, c=color: {
                'fillColor': c, 'color': c, 'weight': 2, 'fillOpacity': 0.4
            },
            popup=folium.Popup(popup_html, max_width=600),
            tooltip=tooltip
        ).add_to(group)

    # MultiPoint
    elif isinstance(geom, MultiPoint):
        for p in geom.geoms:
            folium.CircleMarker(
                location=[p.y, p.x],
                radius=5,
                color=color,
                fill=True,
                fillOpacity=0.8,
                popup=folium.Popup(popup_html, max_width=600),
                tooltip=tooltip
            ).add_to(group)

    # MultiLineString
    elif isinstance(geom, MultiLineString):
        for line in geom.geoms:
            coords = [(y, x) for x, y in line.coords]
            folium.PolyLine(
                locations=coords,
                color=color,
                weight=4,
                popup=folium.Popup(popup_html, max_width=600),
                tooltip=tooltip
            ).add_to(group)


def create_district_maps(pois_df: pd.DataFrame, output_dir: str = "./vis/maps"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 解析 WKT
    if 'geometry' in pois_df.columns and isinstance(pois_df['geometry'].iloc[0], str):
        logger.info("正在解析 WKT 几何...")
        pois_df['geometry'] = pois_df['geometry'].apply(parse_wkt_geometry)
        pois_df = pois_df.dropna(subset=['geometry']).copy()

    if pois_df.empty:
        logger.error("无有效 POI 数据")
        return

    pois = gpd.GeoDataFrame(pois_df, geometry='geometry', crs="EPSG:4326")

    # 自动推断 district
    if 'district' not in pois.columns:
        logger.warning("未发现 district 字段，尝试自动推断...")
        pois['district'] = pois.apply(infer_district, axis=1)
    else:
        pois['district'] = pois['district'].fillna('未知区').astype(str)

    districts = sorted(pois['district'].unique())
    logger.info(f"识别行政区: {districts}")

    # 配色方案
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
        if sub.empty:
            continue

        m = folium.Map(location=get_map_center(sub), zoom_start=13, tiles="CartoDB positron")
        cats = sub['category'].fillna('unknown').unique().tolist()
        if 'unknown' not in cats:
            cats.append('unknown')

        groups = {}
        for c in cats:
            cnt = len(sub[sub['category'].fillna('unknown') == c])
            fg = folium.FeatureGroup(name=f"{c} ({cnt})")
            groups[c] = fg
            m.add_child(fg)

        for _, p in sub.iterrows():
            c = str(p.get('category', 'unknown')).strip() or 'unknown'
            add_poi_to_map(p, groups[c], color_map)

        folium.LayerControl(collapsed=False).add_to(m)
        map_file = output_path / f"map_{d}_pois.html"
        m.save(map_file)
        logger.success(f"已生成: {map_file.name}")

    # ================== 2. 全域总图 ==================
    logger.info("正在生成全域总图...")
    m_all = folium.Map(location=get_map_center(pois), zoom_start=11, tiles="CartoDB positron")

    all_cats = pois['category'].fillna('unknown').unique().tolist()
    if 'unknown' not in all_cats:
        all_cats.append('unknown')

    groups_all = {}
    for c in all_cats:
        cnt = len(pois[pois['category'].fillna('unknown') == c])
        fg = folium.FeatureGroup(name=f"{c} ({cnt})")
        groups_all[c] = fg
        m_all.add_child(fg)

    for _, p in pois.iterrows():
        c = str(p.get('category', 'unknown')).strip() or 'unknown'
        add_poi_to_map(p, groups_all[c], color_map)

    # 添加行政区边界（可选）
    if len(districts) > 1:
        try:
            boundaries = pois.dissolve(by='district')[['geometry']]
            for idx, (dist_name, row) in enumerate(boundaries.iterrows()):
                popup = folium.Popup(f"<b>行政区：</b>{dist_name}", max_width=200)
                folium.GeoJson(
                    data={"type": "Feature", "geometry": row.geometry.__geo_interface__},
                    style_function=lambda x: {'color': '#3388ff', 'weight': 3, 'fillOpacity': 0},
                    name="行政区边界"
                ).add_child(popup).add_to(m_all)
        except Exception as e:
            logger.warning(f"行政区边界绘制失败: {e}")

    folium.LayerControl(collapsed=False).add_to(m_all)
    all_map_file = output_path / "map_Chengdu_ALL_pois.html"
    m_all.save(all_map_file)
    logger.success(f"全域地图已保存: {all_map_file.name}")

    logger.info(f"所有地图已生成！请查看目录：{output_path.resolve()}")


# ================== 主程序入口 ==================
if __name__ == "__main__":
    try:
        POI_FILE = "./vis/chengdu_all_pois.xlsx"   # 修改为你的文件路径
        OUTPUT_DIR = "./vis/maps"

        if not os.path.exists(POI_FILE):
            logger.error(f"POI 文件不存在: {POI_FILE}")
            exit(1)

        logger.info(f"正在读取 POI 数据 ← {POI_FILE}")
        df = pd.read_excel(POI_FILE)

        if 'geometry' not in df.columns:
            logger.error("Excel 中必须包含 'geometry' 列（WKT 格式）")
            exit(1)

        create_district_maps(df, OUTPUT_DIR)

    except Exception as e:
        logger.error(f"程序运行出错: {e}")
        raise