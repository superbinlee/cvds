import folium
import pandas as pd
import re
from folium import plugins
from shapely.wkt import loads
from shapely.geometry import Polygon, MultiPolygon


def extract_coordinates(wkt_str):
    """从WKT字符串中提取坐标点列表"""
    # 移除WKT类型前缀
    coord_str = re.sub(r'^MULTIPOLYGON \(\(\(|\)\)\)$|^POLYGON \(\(|\)\)$', '', wkt_str)
    # 分割坐标对
    coord_pairs = coord_str.split(',')
    # 转换为经纬度元组列表
    coordinates = []
    for pair in coord_pairs:
        lon, lat = pair.strip().split()
        coordinates.append((float(lat), float(lon)))  # folium使用(lat, lon)格式
    return coordinates


def main():
    # 读取区域边界数据
    districts_df = pd.read_csv('../districts/chengdu_districts_boundary.csv')

    # 读取相机点位数据
    cameras_df = pd.read_excel('../cameras_all.xlsx')

    # 确定成都的大致中心坐标（用于初始地图定位）
    chengdu_center = [30.6570, 104.0650]  # 成都大致中心点

    # 创建地图对象
    m = folium.Map(location=chengdu_center, zoom_start=11, tiles='CartoDB positron')

    # 添加区域边界
    for idx, row in districts_df.iterrows():
        district_name = row['区域名称']
        boundary_wkt = row['区域边界']

        try:
            # 尝试解析WKT格式
            geom = loads(boundary_wkt)

            # 根据几何类型添加到地图
            if isinstance(geom, Polygon):
                coords = [(y, x) for x, y in geom.exterior.coords]
                folium.Polygon(
                    locations=coords,
                    color='blue',
                    fill=True,
                    fill_color='blue',
                    fill_opacity=0.1,
                    tooltip=district_name
                ).add_to(m)
            elif isinstance(geom, MultiPolygon):
                for poly in geom.geoms:
                    coords = [(y, x) for x, y in poly.exterior.coords]
                    folium.Polygon(
                        locations=coords,
                        color='blue',
                        fill=True,
                        fill_color='blue',
                        fill_opacity=0.1,
                        tooltip=district_name
                    ).add_to(m)
        except:
            # 如果WKT解析失败，尝试使用自定义解析
            coords = extract_coordinates(boundary_wkt)
            if coords:
                folium.Polygon(
                    locations=coords,
                    color='blue',
                    fill=True,
                    fill_color='blue',
                    fill_opacity=0.1,
                    tooltip=district_name
                ).add_to(m)

    # 添加相机点位
    for idx, row in cameras_df.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.7,
            tooltip=f"{row['name']} ({row['camera_id']})"
        ).add_to(m)

    # 添加图层控制
    folium.LayerControl().add_to(m)

    # 保存为HTML文件
    m.save('chengdu_cameras_map.html')
    print("地图已保存为 'chengdu_cameras_map.html'")


if __name__ == "__main__":
    main()