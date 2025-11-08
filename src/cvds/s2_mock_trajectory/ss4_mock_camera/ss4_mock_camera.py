import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from loguru import logger
from pathlib import Path
import folium
from folium import CircleMarker
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置区 ====================
BASE_DIR = Path("./road")
INPUT_GPKG = BASE_DIR / "Chengdu_all_road_network.gpkg"
OUTPUT_EXCEL = BASE_DIR / "camera_deployments.xlsx"
OUTPUT_HTML = BASE_DIR / "camera_map.html"

# 部署参数
URBAN_INTERVAL_KM = 0.1      # 市区每 100m 一个
RURAL_INTERVAL_KM = 0.5      # 外围每 500m 一个
DENSITY_THRESHOLD = 5.0      # km/0.01°网格，>此为市区
GRID_SIZE_DEG = 0.01         # 网格大小

# 行政区映射（从文件名提取）
DISTRICT_MAP = {
    "温江区": "Wenjiang",
    "金牛区": "Jinniu",
    "成华区": "Chenghua",
    "郫都区": "Pidu",
    "新都区": "Xindu",
    "青羊区": "Qingyang",
}
# ===============================================

def extract_district_from_osmid(osmid: str) -> str:
    """从 osmid 中提取区名前缀（如 123_Wenjiang → 温江区）"""
    try:
        suffix = osmid.split("_")[-1]
        for cn, en in DISTRICT_MAP.items():
            if en in suffix:
                return cn
        return "未知区"
    except:
        return "未知区"

def compute_road_density(edges_gdf: gpd.GeoDataFrame, nodes_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    logger.info("计算道路密度（用于区分市区/外围）...")
    edges_gdf['length_km'] = edges_gdf.to_crs('EPSG:3857').length / 1000
    edges_gdf = edges_gdf.to_crs('EPSG:4326')

    # 创建网格
    bounds = edges_gdf.total_bounds
    lons = np.arange(bounds[0], bounds[2], GRID_SIZE_DEG)
    lats = np.arange(bounds[1], bounds[3], GRID_SIZE_DEG)
    grid_centers = [Point(lon + GRID_SIZE_DEG/2, lat + GRID_SIZE_DEG/2) for lon in lons for lat in lats]
    grid_gdf = gpd.GeoDataFrame({'grid_id': range(len(grid_centers))}, geometry=grid_centers, crs='EPSG:4326')

    # 边中心点 → 网格
    edge_centers = edges_gdf.geometry.centroid
    edges_with_grid = gpd.sjoin(edges_gdf[['length_km', 'geometry']].set_geometry(edge_centers),
                                grid_gdf, how='left', predicate='intersects')
    density = edges_with_grid.groupby('grid_id')['length_km'].sum().reset_index()
    grid_gdf = grid_gdf.merge(density, on='grid_id', how='left').fillna(0)

    # 节点 → 密度
    nodes_with_density = gpd.sjoin(nodes_gdf, grid_gdf[['geometry', 'length_km']],
                                   how='left', predicate='intersects')
    nodes_gdf['density_km_per_grid'] = nodes_with_density['length_km'].fillna(0)
    logger.success(f"密度计算完成，平均密度: {nodes_gdf['density_km_per_grid'].mean():.2f} km/网格")
    return nodes_gdf

def deploy_intersection_cameras(nodes_gdf, edges_gdf) -> list:
    logger.info("部署路口相机...")
    cameras = []
    for idx, node in nodes_gdf.iterrows():
        osmid = node['osmid']
        x, y = node.geometry.x, node.geometry.y
        district = extract_district_from_osmid(osmid)
        density = node['density_km_per_grid']
        is_urban = density > DENSITY_THRESHOLD

        out_edges = edges_gdf[edges_gdf['u'] == osmid]
        directions = set()

        if len(out_edges) == 0:
            # 孤立节点
            cam_id = f"cam_node_{osmid}_default"
            cameras.append({
                'camera_id': cam_id,
                'name': f"{district}-路口-孤立",
                'longitude': x,
                'latitude': y,
                'district': district,
                'source_type': 'intersection',
                'direction_bearing': 0
            })
            continue

        for _, edge in out_edges.iterrows():
            line = edge.geometry
            if line.geom_type != 'LineString':
                continue
            coord0 = np.array(line.coords[0])
            node_pt = np.array([x, y])
            vec = coord0 - node_pt
            bearing = np.degrees(np.arctan2(vec[1], vec[0])) % 360
            dir_bin = round(bearing / 90) * 90
            if dir_bin not in directions:
                directions.add(dir_bin)
                cam_id = f"cam_node_{osmid}_dir{int(dir_bin)}"
                cameras.append({
                    'camera_id': cam_id,
                    'name': f"{district}-路口-{len(directions)}向",
                    'longitude': x,
                    'latitude': y,
                    'district': district,
                    'source_type': 'intersection',
                    'direction_bearing': bearing
                })
    logger.success(f"路口相机: {len(cameras)} 个")
    return cameras

def deploy_road_cameras(edges_gdf, nodes_gdf) -> list:
    logger.info("部署道路段相机...")
    cameras = []
    edges_gdf = edges_gdf.copy()
    edges_gdf['length_m'] = edges_gdf.to_crs('EPSG:3857').length
    edges_gdf = edges_gdf.to_crs('EPSG:4326')

    for idx, edge in edges_gdf.iterrows():
        if edge['length_m'] < 50:  # 太短跳过
            continue
        u = edge['u']
        v = edge['v']
        length_km = edge['length_m'] / 1000
        district = extract_district_from_osmid(u)  # u/v 任一
        u_node = nodes_gdf[nodes_gdf['osmid'] == u].iloc[0]
        is_urban = u_node['density_km_per_grid'] > DENSITY_THRESHOLD
        interval_km = URBAN_INTERVAL_KM if is_urban else RURAL_INTERVAL_KM
        num_points = max(1, int(length_km / interval_km))

        line = edge.geometry
        for i in range(1, num_points):  # 跳过两端（已有路口相机）
            frac = i / (num_points + 1)
            point = line.interpolate(frac, normalized=True)
            x, y = point.x, point.y

            # 朝向：边方向
            coords = list(line.coords)
            dx = coords[-1][0] - coords[0][0]
            dy = coords[-1][1] - coords[0][1]
            bearing = np.degrees(np.arctan2(dy, dx)) % 360

            cam_id = f"cam_edge_{edge.name}_pt{i}"
            cameras.append({
                'camera_id': cam_id,
                'name': f"{district}-道路段-{i}",
                'longitude': x,
                'latitude': y,
                'district': district,
                'source_type': 'road_segment',
                'direction_bearing': bearing
            })
    logger.success(f"道路相机: {len(cameras)} 个")
    return cameras

def create_folium_map(df: pd.DataFrame, output_path: Path):
    logger.info(f"生成 Folium 地图 → {output_path}")
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles='CartoDB positron')

    # 图例
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; width: 150px; height: 90px; 
         border:2px solid grey; z-index:9999; font-size:14px; background-color:white;
         padding: 10px;">
     <b>相机类型</b><br>
     <i class="fa fa-circle" style="color:red"></i>&nbsp;路口<br>
     <i class="fa fa-circle" style="color:blue"></i>&nbsp;道路段
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # 添加点
    for _, row in df.iterrows():
        color = 'red' if row['source_type'] == 'intersection' else 'blue'
        popup = f"""
        <b>{row['name']}</b><br>
        ID: {row['camera_id']}<br>
        区: {row['district']}<br>
        类型: {row['source_type']}<br>
        朝向: {row['direction_bearing']:.1f}°
        """
        CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=4,
            color=color,
            fill=True,
            fillOpacity=0.7,
            popup=folium.Popup(popup, max_width=300)
        ).add_to(m)

    m.save(str(output_path))
    logger.success(f"地图保存成功: {output_path}")

if __name__ == "__main__":
    # 1. 加载数据
    logger.info(f"加载路网: {INPUT_GPKG}")
    nodes_gdf = gpd.read_file(INPUT_GPKG, layer='nodes')
    edges_gdf = gpd.read_file(INPUT_GPKG, layer='edges')

    # 2. 计算密度
    nodes_gdf = compute_road_density(edges_gdf, nodes_gdf)

    # 3. 部署相机
    intersection_cams = deploy_intersection_cameras(nodes_gdf, edges_gdf)
    road_cams = deploy_road_cameras(edges_gdf, nodes_gdf)
    all_cams = intersection_cams + road_cams

    # 4. 转为 DataFrame
    df = pd.DataFrame(all_cams)
    df = df[[
        'latitude', 'longitude', 'name', 'camera_id', 'district', 'source_type'
    ]]

    # 5. 保存 Excel
    df.to_excel(OUTPUT_EXCEL, index=False)
    logger.success(f"Excel 保存成功: {OUTPUT_EXCEL}，共 {len(df)} 条记录")

    # 6. 生成 Folium 地图
    create_folium_map(df, OUTPUT_HTML)

    # 7. 打印统计
    print("\n" + "="*50)
    print("部署统计")
    print("="*50)
    print(df['source_type'].value_counts())
    print("\n各区相机数量:")
    print(df['district'].value_counts())
    print(f"\n总相机数: {len(df)}")
    print(f"Excel: {OUTPUT_EXCEL}")
    print(f"地图: {OUTPUT_HTML}")