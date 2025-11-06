import os
import time
import osmnx as ox
import geopandas as gpd
from loguru import logger
from pathlib import Path


def download_road_network(place_name: str, network_type: str, save_path: str) -> bool:
    """
    下载路网数据并保存（若文件已存在则跳过）。

    Args:
        place_name: 完整的地名（OSM 查询字符串）
        network_type: 路网类型，如 "all"、"drive"、"walk"
        save_path: 保存的 .gpkg 文件路径

    Returns:
        bool: 下载/读取成功返回 True，否则 False
    """
    if os.path.exists(save_path):
        logger.info(f"文件已存在，跳过下载 → {save_path}")
        return True

    try:
        logger.info(f"开始下载 {place_name} 的 {network_type} 路网...")
        start = time.time()

        G = ox.graph_from_place(
            place_name,
            network_type=network_type,
            simplify=True,
            retain_all=False,   # 只保留最大连通子图，防止碎块
            truncate_by_edge=False
        )
        ox.save_graph_geopackage(G, filepath=save_path)

        elapsed = time.time() - start
        logger.success(f"保存成功 → {save_path}，耗时 {elapsed:.2f}s")
        return True

    except Exception as e:
        logger.error(f"下载失败 [{place_name}]：{e}")
        return False


def merge_geopackages(gpkg_paths: list[str], output_path: str) -> None:
    """
    将多个 GeoPackage（每个包含 nodes、edges 两层）合并为一个。

    Args:
        gpkg_paths: 待合并的 .gpkg 文件路径列表
        output_path: 合并后文件的完整路径
    """
    if not gpkg_paths:
        logger.warning("没有可合并的文件")
        return

    logger.info(f"开始合并 {len(gpkg_paths)} 个 GeoPackage → {output_path}")

    # 用于累计节点和边
    nodes_list = []
    edges_list = []

    for p in gpkg_paths:
        try:
            # 每个 gpkg 里都有两层：'nodes' 和 'edges'
            nodes = gpd.read_file(p, layer="nodes")
            edges = gpd.read_file(p, layer="edges")

            # 可选：为避免 ID 冲突，给每个区的节点/边加上区名前缀
            district = Path(p).stem.split("_")[0]
            nodes["osmid"] = nodes["osmid"].astype(str) + f"_{district}"
            edges["u"] = edges["u"].astype(str) + f"_{district}"
            edges["v"] = edges["v"].astype(str) + f"_{district}"
            edges["key"] = edges.index.astype(str) + f"_{district}"

            nodes_list.append(nodes)
            edges_list.append(edges)
        except Exception as e:
            logger.error(f"读取 {p} 失败：{e}")

    if not nodes_list or not edges_list:
        logger.error("没有成功读取任何图层，合并中止")
        return

    all_nodes = gpd.pd.concat(nodes_list, ignore_index=True)
    all_edges = gpd.pd.concat(edges_list, ignore_index=True)

    # 保存为新的 GeoPackage（两层）
    all_nodes.to_file(output_path, layer="nodes", driver="GPKG")
    all_edges.to_file(output_path, layer="edges", driver="GPKG")

    logger.success(f"合并完成 → {output_path}（nodes: {len(all_nodes)}, edges: {len(all_edges)}）")


if __name__ == "__main__":
    # ------------------- 配置区 -------------------
    NETWORK_TYPE = "all"          # 可选: "drive", "walk", "bike", "all"
    BASE_DIR = Path("./road")     # 保存目录
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    districts = [
        # "温江区, 成都市, 四川省, 中国",
        "金牛区, 成都市, 四川省, 中国",
        "成华区, 成都市, 四川省, 中国",
        "郫都区, 成都市, 四川省, 中国",
        "新都区, 成都市, 四川省, 中国",
        "青羊区, 成都市, 四川省, 中国",
    ]
    # ------------------------------------------------

    success_cnt = 0
    downloaded_files = []                     # 记录成功下载的文件路径

    for idx, place in enumerate(districts, start=1):
        safe_name = place.split(",")[0]
        save_path = BASE_DIR / f"{safe_name}_road_network.gpkg"

        logger.info(f"[{idx}/{len(districts)}] 处理 {place}")
        if download_road_network(place, NETWORK_TYPE, str(save_path)):
            success_cnt += 1
            downloaded_files.append(str(save_path))
        else:
            logger.warning(f"[{place}] 下载失败，稍后可手动重试")

        time.sleep(2)                         # 避免触发 OSM 频率限制

    logger.info(f"全部下载完成！成功 {success_cnt}/{len(districts)} 个区")

    # ------------------- 合并阶段 -------------------
    if success_cnt > 0:
        merged_path = BASE_DIR / "Chengdu_all_road_network.gpkg"
        merge_geopackages(downloaded_files, str(merged_path))
    else:
        logger.error("没有任何区下载成功，跳过合并")