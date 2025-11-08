# -*- coding: utf-8 -*-
"""
下载成都市行政区边界 → 导出为 Excel
- 输入: districts 列表
- 输出: ./districts_excel/每个区.xlsx (每行一个边界点)
- 字段: longitude, latitude
- 来源: OSM Nominatim API
"""

import geopandas as gpd
import pandas as pd
import requests
from loguru import logger
from pathlib import Path
from tqdm import tqdm
import time
from shapely.geometry import Polygon, MultiPolygon

# ==================== 配置 ====================
districts = [
    "金牛区, 成都市, 四川省, 中国",
    "成华区, 成都市, 四川省, 中国",
    "郫都区, 成都市, 四川省, 中国",
    "新都区, 成都市, 四川省, 中国",
    "青羊区, 成都市, 四川省, 中国",
]

BASE_DIR = Path("./districts_excel")
BASE_DIR.mkdir(exist_ok=True)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
HEADERS = {"User-Agent": "district-excel-downloader/1.0"}


# ==================== 提取多边形所有坐标点 ====================
def extract_coords_from_geometry(geom):
    """
    从 Polygon 或 MultiPolygon 提取所有坐标点
    返回: List[(lon, lat)]
    """
    coords = []
    if isinstance(geom, Polygon):
        # 外边界
        for lon, lat in geom.exterior.coords:
            coords.append((lon, lat))
        # 内环（洞）
        for interior in geom.interiors:
            for lon, lat in interior.coords:
                coords.append((lon, lat))
    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            coords.extend(extract_coords_from_geometry(poly))
    return coords


# ==================== 下载并导出 Excel ====================
def download_and_export_district(place: str, save_path: Path) -> bool:
    params = {
        "q": place,
        "format": "geojson",
        "polygon_geojson": "1",
        "limit": "1",
        "countrycodes": "cn"
    }

    try:
        response = requests.get(NOMINATIM_URL, params=params, headers=HEADERS, timeout=30)
        response.raise_for_status()

        gdf = gpd.read_file(response.text, driver="GeoJSON")

        if gdf.empty or len(gdf.geometry) == 0:
            logger.warning(f"未找到边界: {place}")
            return False

        # 取第一个特征（最佳匹配）
        geom = gdf.geometry.iloc[0]
        if geom is None or geom.is_empty:
            logger.warning(f"边界为空: {place}")
            return False

        # 提取所有坐标点
        points = extract_coords_from_geometry(geom)
        if not points:
            logger.warning(f"无坐标点: {place}")
            return False

        # 转为 DataFrame
        df = pd.DataFrame(points, columns=['longitude', 'latitude'])

        # 保存为 Excel
        df.to_excel(save_path, index=False)

        logger.success(f"导出成功: {save_path.name} → {len(df):,} 个点")
        return True

    except Exception as e:
        logger.error(f"处理失败 {place}: {e}")
        return False


# ==================== 主流程 ====================
if __name__ == "__main__":
    success_count = 0
    total_points = 0

    for place in tqdm(districts, desc="下载并导出行政区", unit="区"):
        district_name = place.split(",")[0].strip()
        save_path = BASE_DIR / f"{district_name}.xlsx"

        if download_and_export_district(place, save_path):
            success_count += 1
            # 统计点数
            try:
                df_temp = pd.read_excel(save_path)
                total_points += len(df_temp)
            except:
                pass

        time.sleep(1.1)  # Nominatim 限流

    logger.info(f"全部完成！成功 {success_count}/{len(districts)} 个区")
    logger.info(f"总计导出点数: {total_points:,}")
    logger.info(f"文件保存在: {BASE_DIR.resolve()}")

    # 打印文件列表
    print("\n" + "=" * 60)
    print("导出完成！")
    print("=" * 60)
    for file in sorted(BASE_DIR.glob("*.xlsx")):
        try:
            df = pd.read_excel(file)
            print(f"  {file.name:<15} → {len(df):>6} 个点")
        except:
            print(f"  {file.name:<15} → 读取失败")
    print(f"\n总点数: {total_points:,}")
    print("=" * 60)
