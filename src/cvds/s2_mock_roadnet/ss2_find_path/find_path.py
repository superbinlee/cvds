import os
import geopandas as gpd
import networkx as nx
import folium
from haversine import haversine


def load_road_network(gpkg_path):
    """加载路网并构建无向图（忽略方向，提高路径可达性）"""
    nodes = gpd.read_file(gpkg_path, layer="nodes")
    edges = gpd.read_file(gpkg_path, layer="edges")

    source_col = "u" if "u" in edges.columns else "source"
    target_col = "v" if "v" in edges.columns else "target"

    # 构建无向图（忽略单行道限制，仅用于解决路径不可达问题）
    G = nx.from_pandas_edgelist(
        edges,
        source=source_col,
        target=target_col,
        edge_attr=["length", "name", "geometry"],
        create_using=nx.MultiGraph()  # 关键：无向图，双向均可通行
    )

    # 添加节点坐标
    node_id_col = "osmid" if "osmid" in nodes.columns else "id"
    for _, row in nodes.iterrows():
        node_id = row[node_id_col]
        if node_id in G.nodes:
            G.nodes[node_id]["x"] = row["x"]
            G.nodes[node_id]["y"] = row["y"]

    return G


def find_nearest_node(G, lon, lat):
    """查找最近的节点"""
    min_dist = float("inf")
    nearest_node = None
    for node, data in G.nodes(data=True):
        if "x" in data and "y" in data:
            dist = haversine((lat, lon), (data["y"], data["x"]), unit="m")
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
    return nearest_node, min_dist


def calculate_and_visualize_path(gpkg_path, start_lon, start_lat, end_lon, end_lat):
    # 1. 加载路网
    if not os.path.exists(gpkg_path):
        print(f"文件不存在: {gpkg_path}")
        return

    G = load_road_network(gpkg_path)
    print(f"路网加载完成: {len(G.nodes)}个节点, {len(G.edges)}条边")

    # 2. 匹配节点
    start_node, start_dist = find_nearest_node(G, start_lon, start_lat)
    end_node, end_dist = find_nearest_node(G, end_lon, end_lat)

    if not start_node or not end_node:
        print("未找到匹配的起点或终点节点")
        return

    print(f"匹配节点: 起点距离{start_dist:.2f}米, 终点距离{end_dist:.2f}米")

    # 3. 计算路径（使用Dijkstra算法，兼容性更好）
    try:
        path = nx.shortest_path(G, start_node, end_node, weight="length")
    except nx.NetworkXNoPath:
        print("路径不可达！尝试显示起点终点位置...")
        # 即使路径不可达，也展示起点终点
        m = folium.Map(location=[(start_lat + end_lat) / 2, (start_lon + end_lon) / 2], zoom_start=13)
        folium.Marker([start_lat, start_lon], icon=folium.Icon(color="green"), popup="起点").add_to(m)
        folium.Marker([end_lat, end_lon], icon=folium.Icon(color="red"), popup="终点").add_to(m)
        m.save("unreachable_path.html")
        print("已保存起点终点位置至 unreachable_path.html")
        return

    # 4. 提取路径几何信息
    path_geoms = []
    total_length = 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edge_data = G.get_edge_data(u, v)[0]  # 取第一条边
        total_length += edge_data.get("length", 0)
        if "geometry" in edge_data:
            path_geoms.append(edge_data["geometry"])

    # 5. 可视化路径（确保执行到这一步）
    m = folium.Map(location=[(start_lat + end_lat) / 2, (start_lon + end_lon) / 2], zoom_start=13)
    folium.Marker([start_lat, start_lon], icon=folium.Icon(color="green"), popup="起点").add_to(m)
    folium.Marker([end_lat, end_lon], icon=folium.Icon(color="red"), popup="终点").add_to(m)

    # 确保路径几何信息存在
    if path_geoms:
        for geom in path_geoms:
            folium.GeoJson(geom, style_function=lambda x: {"color": "blue", "weight": 4}).add_to(m)
        print(f"路径计算完成: 总长度 {total_length / 1000:.2f} 公里")
    else:
        print("路径几何信息缺失，但已标记起点终点")

    m.save("route_with_path.html")
    print("路径已保存至 route_with_path.html")


# 主程序：使用杭州实际可达的坐标
if __name__ == "__main__":
    # 选择杭州城区内较近的两个点（确保路网覆盖）
    start_lon, start_lat = 120.20511, 30.206485  # 海康威视
    end_lon, end_lat = 120.204549, 30.169539  # 白马湖

    calculate_and_visualize_path(
        "../ss2_download_roadnet/hangzhou_road_network.gpkg",
        start_lon, start_lat,
        end_lon, end_lat
    )
