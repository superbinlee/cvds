import folium
import pandas as pd
from shapely.wkt import loads
from shapely.geometry import Polygon, MultiPolygon
import warnings

warnings.filterwarnings('ignore')  # 忽略 shapely 无关警告


def draw_district_boundaries(csv_path, output_html="district_boundaries.html"):
    # 1. 读取 CSV 文件
    df = pd.read_csv(csv_path)
    print(f"成功读取 {len(df)} 个区域数据")

    # 校验必要列是否存在
    required_cols = ["区域名称", "区域边界"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV 文件必须包含列：{required_cols}（列名严格匹配）")

    # 2. 初始化地图（以数据中心坐标为中心点）
    centroids = []
    for wkt in df["区域边界"]:
        try:
            # 尝试解析 WKT（兼容 POLYGON/MULTIPOLYGON）
            geom = loads(wkt)
            if isinstance(geom, (Polygon, MultiPolygon)):
                centroid = geom.centroid
                centroids.append([centroid.y, centroid.x])  # folium: [lat, lon]
        except Exception as e:
            print(f"暂无法解析边界：{str(e)[:50]}...")
            continue

    # 初始化地图（优先用数据中心，否则用默认成都坐标）
    if centroids:
        avg_lat = sum([c[0] for c in centroids]) / len(centroids)
        avg_lon = sum([c[1] for c in centroids]) / len(centroids)
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=12)
    else:
        m = folium.Map(location=[30.67, 104.06], zoom_start=11)  # 成都默认中心

    # 3. 定义颜色列表（区分不同区域）
    colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FECA57",
        "#FF9FF3", "#54A0FF", "#5F27CD", "#00D2D3", "#FF9F43",
        "#10AC84", "#EE5A24", "#0984e3", "#a29bfe", "#fd79a8"
    ]

    # 4. 遍历区域，绘制边界（兼容单个/多个多边形）
    for idx, row in df.iterrows():
        district_name = row["区域名称"].strip()  # 去除名称前后空格
        wkt_boundary = row["区域边界"].strip()

        try:
            # 解析 WKT（支持 POLYGON/MULTIPOLYGON）
            geom = loads(wkt_boundary)
            color = colors[idx % len(colors)]  # 同一区域的所有子多边形用同一种颜色

            # 处理单个多边形（POLYGON）
            if isinstance(geom, Polygon):
                # 提取外环坐标（转换为 folium 要求的 [lat, lon]）
                coordinates = [[lat, lon] for lon, lat in geom.exterior.coords]
                # 绘制单个多边形
                add_polygon_to_map(m, coordinates, color, district_name)

            # 处理多个多边形（MULTIPOLYGON，如飞地）
            elif isinstance(geom, MultiPolygon):
                # 遍历每个子多边形，分别绘制
                for sub_polygon in geom.geoms:
                    coordinates = [[lat, lon] for lon, lat in sub_polygon.exterior.coords]
                    add_polygon_to_map(m, coordinates, color, district_name)
                print(f"成功绘制（多多边形）：{district_name}")

            else:
                print(f"跳过 {district_name}：不支持的几何类型（{type(geom).__name__}）")
                continue

            # 标注区域名称（放在整个几何图形的中心）
            centroid = geom.centroid
            folium.Marker(
                location=[centroid.y, centroid.x],
                icon=folium.DivIcon(
                    html=f'''<div style="font-size: 12px; color: #333; font-weight: bold; 
                                  background: rgba(255,255,255,0.8); padding: 2px 5px; 
                                  border-radius: 3px; border: 1px solid #ddd;">{district_name}</div>''',
                    icon_size=(120, 30)  # 文本框大小（宽度自适应）
                )
            ).add_to(m)

        except Exception as e:
            # 捕获 WKT 不完整、格式错误等异常
            error_msg = str(e)[:80]  # 截取部分错误信息
            print(f"处理 {district_name} 失败：{error_msg}...")
            continue

    # 5. 添加图层控制（可开关单个区域）
    folium.LayerControl(collapsed=False).add_to(m)

    # 6. 保存地图
    m.save(output_html)
    print(f"\n地图已保存到：{output_html}（用浏览器打开即可查看）")
    return output_html


def add_polygon_to_map(map_obj, coordinates, color, district_name):
    """辅助函数：向地图添加多边形边界"""
    folium.Polygon(
        locations=coordinates,
        color=color,
        weight=3,  # 边界线宽度
        fill=True,
        fill_color=color,
        fill_opacity=0.3,  # 填充透明度（避免遮挡其他区域）
        popup=folium.Popup(district_name, max_width=200),  # 点击显示名称
        name=district_name  # 图层名称
    ).add_to(map_obj)


# ------------------- 调用函数 -------------------
if __name__ == "__main__":
    # 替换为你的 CSV 文件路径（例如："成都各区边界.csv"）
    CSV_FILE_PATH = "chengdu_districts_boundary.csv"
    # 生成地图
    draw_district_boundaries(CSV_FILE_PATH)
