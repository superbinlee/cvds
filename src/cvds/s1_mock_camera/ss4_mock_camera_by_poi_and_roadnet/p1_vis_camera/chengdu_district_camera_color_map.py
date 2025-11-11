import folium
import pandas as pd
import re
import logging
from shapely.wkt import loads
from shapely.geometry import Polygon, MultiPolygon

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 定义区域颜色列表（可根据需要扩展）
DISTRICT_COLORS = [
    '#FF5733', '#33FF57', '#3357FF', '#F3FF33', '#FF33F3',
    '#33FFF3', '#F333FF', '#FF9933', '#33FF99', '#9933FF'
]

# 定义相机source对应的颜色
CAMERA_COLOR_MAP = {
    'poi': 'red',
    'road': 'blue'
}
DEFAULT_CAMERA_COLOR = 'gray'  # 未知source的默认颜色


def extract_coordinates(wkt_str):
    """从WKT字符串中提取坐标点列表"""
    try:
        coord_str = re.sub(r'^MULTIPOLYGON \(\(\(|\)\)\)$|^POLYGON \(\(|\)\)$', '', wkt_str)
        coord_pairs = coord_str.split(',')
        coordinates = []
        for pair in coord_pairs:
            lon, lat = pair.strip().split()
            coordinates.append((float(lat), float(lon)))
        logger.info(f"成功解析坐标，共提取{len(coordinates)}个坐标点")
        return coordinates
    except Exception as e:
        logger.error(f"坐标解析失败：{str(e)}")
        return []


def main():
    logger.info("开始执行成都区域边界+相机点位地图绘制程序（带颜色区分）")

    # 读取区域边界数据
    try:
        logger.info("正在读取区域边界数据：../districts/chengdu_districts_boundary.csv")
        districts_df = pd.read_csv('../districts/chengdu_districts_boundary.csv')
        logger.info(f"成功读取区域边界数据，共{len(districts_df)}个区域")
    except Exception as e:
        logger.error(f"区域边界数据读取失败：{str(e)}")
        return

    # 读取相机点位数据
    try:
        logger.info("正在读取相机点位数据：../cameras_all.xlsx")
        cameras_df = pd.read_excel('../cameras_all.xlsx')
        logger.info(f"成功读取相机点位数据，共{len(cameras_df)}个相机点位")
    except Exception as e:
        logger.error(f"相机点位数据读取失败：{str(e)}")
        return

    # 初始化地图
    chengdu_center = [30.6570, 104.0650]
    try:
        m = folium.Map(location=chengdu_center, zoom_start=11, tiles='CartoDB positron')
        logger.info("成功创建地图对象")
    except Exception as e:
        logger.error(f"地图对象创建失败：{str(e)}")
        return

    # 添加区域边界（不同区域不同颜色）
    logger.info("开始添加区域边界（不同区域不同颜色）...")
    success_districts = 0
    fail_districts = 0

    for idx, row in districts_df.iterrows():
        district_name = row['区域名称']
        boundary_wkt = row['区域边界']
        # 循环使用颜色列表（取模避免索引越界）
        color_idx = idx % len(DISTRICT_COLORS)
        district_color = DISTRICT_COLORS[color_idx]
        logger.info(f"正在处理区域：{district_name}，分配颜色：{district_color}")

        try:
            geom = loads(boundary_wkt)
            logger.info(f"区域{district_name} WKT格式解析成功")

            if isinstance(geom, Polygon):
                coords = [(y, x) for x, y in geom.exterior.coords]
                folium.Polygon(
                    locations=coords,
                    color=district_color,
                    fill=True,
                    fill_color=district_color,
                    fill_opacity=0.1,
                    tooltip=district_name
                ).add_to(m)
                logger.info(f"成功添加区域{district_name}（Polygon类型）")
                success_districts += 1
            elif isinstance(geom, MultiPolygon):
                poly_count = 0
                for poly in geom.geoms:
                    coords = [(y, x) for x, y in poly.exterior.coords]
                    folium.Polygon(
                        locations=coords,
                        color=district_color,
                        fill=True,
                        fill_color=district_color,
                        fill_opacity=0.1,
                        tooltip=district_name
                    ).add_to(m)
                    poly_count += 1
                logger.info(f"成功添加区域{district_name}（MultiPolygon类型，包含{poly_count}个多边形）")
                success_districts += 1
        except Exception as e1:
            logger.warning(f"区域{district_name} WKT解析失败：{str(e1)}，尝试自定义解析")
            try:
                coords = extract_coordinates(boundary_wkt)
                if coords:
                    folium.Polygon(
                        locations=coords,
                        color=district_color,
                        fill=True,
                        fill_color=district_color,
                        fill_opacity=0.1,
                        tooltip=district_name
                    ).add_to(m)
                    logger.info(f"成功添加区域{district_name}（自定义解析）")
                    success_districts += 1
                else:
                    logger.error(f"区域{district_name}自定义解析失败，未提取到有效坐标")
                    fail_districts += 1
            except Exception as e2:
                logger.error(f"区域{district_name}自定义解析失败：{str(e2)}")
                fail_districts += 1

    logger.info(f"区域边界添加完成 - 成功：{success_districts}个，失败：{fail_districts}个")

    # 添加相机点位（按source区分颜色）
    logger.info("开始添加相机点位（按source区分颜色）...")
    success_cameras = 0
    fail_cameras = 0
    # 统计各source类型数量
    source_count = {}

    for idx, row in cameras_df.iterrows():
        try:
            lat = row['latitude']
            lon = row['longitude']
            name = row['name']
            camera_id = row['camera_id']
            source = row['source']

            # 统计source数量
            source_count[source] = source_count.get(source, 0) + 1

            # 确定相机颜色
            camera_color = CAMERA_COLOR_MAP.get(source, DEFAULT_CAMERA_COLOR)

            # 验证经纬度有效性
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                logger.warning(f"相机{name}({camera_id}) 经纬度无效：lat={lat}, lon={lon}")
                fail_cameras += 1
                continue

            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color=camera_color,
                fill=True,
                fill_color=camera_color,
                fill_opacity=0.7,
                tooltip=f"{name}（{camera_id}）\nsource: {source}"
            ).add_to(m)
            success_cameras += 1
            if (idx + 1) % 10 == 0:
                logger.info(f"已添加{idx + 1}个相机点位")
        except Exception as e:
            logger.error(f"相机点位添加失败（行号：{idx}）：{str(e)}")
            fail_cameras += 1

    logger.info(f"相机点位添加完成 - 成功：{success_cameras}个，失败：{fail_cameras}个")
    logger.info(f"相机source类型分布：{source_count}")

    # 添加图例（说明颜色含义）
    try:
        legend_html = '''
        <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; 
                    padding: 10px; border: 1px solid grey; border-radius: 5px;">
            <p style="margin: 0; font-weight: bold;">图例</p>
        '''
        # 添加区域图例（只显示已使用的颜色）
        used_colors = [DISTRICT_COLORS[i % len(DISTRICT_COLORS)] for i in range(success_districts)]
        unique_colors = list(set(used_colors))
        for color in unique_colors:
            legend_html += f'''
            <p style="margin: 2px 0;"><span style="display: inline-block; width: 12px; height: 12px; 
                        background-color: {color}; margin-right: 5px;"></span>区域</p>
            '''
        # 添加相机图例
        for source, color in CAMERA_COLOR_MAP.items():
            legend_html += f'''
            <p style="margin: 2px 0;"><span style="display: inline-block; width: 12px; height: 12px; 
                        background-color: {color}; border-radius: 50%; margin-right: 5px;"></span>
                        相机（{source}）</p>
            '''
        legend_html += '''
            <p style="margin: 2px 0;"><span style="display: inline-block; width: 12px; height: 12px; 
                        background-color: gray; border-radius: 50%; margin-right: 5px;"></span>
                        相机（未知类型）</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        logger.info("成功添加图例说明")
    except Exception as e:
        logger.warning(f"图例添加失败：{str(e)}")

    # 保存地图
    try:
        output_file = 'chengdu_cameras_map_with_colors.html'
        m.save(output_file)
        logger.info(f"带颜色区分的地图已成功保存为：{output_file}")
        logger.info("程序执行完成！可直接用浏览器打开HTML文件查看")
    except Exception as e:
        logger.error(f"地图保存失败：{str(e)}")


if __name__ == "__main__":
    main()