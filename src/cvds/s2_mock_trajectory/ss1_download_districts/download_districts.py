# -*- coding: utf-8 -*-
"""
下载成都市行政区边界 → 导出为一个 CSV 文件
- 输出: ./districts_excel/chengdu_districts_boundary.csv
- 每行一个区，包含：区域名称 + 区域边界（WKT Polygon 格式）
- 来源: OSM Nominatim API
"""
import time
from pathlib import Path

import pandas as pd
import requests
from loguru import logger
from shapely.geometry import shape
from tqdm import tqdm

# ==================== 配置 ====================
districts = [
    "金牛区, 成都市, 四川省, 中国",
    "成华区, 成都市, 四川省, 中国",
    "郫都区, 成都市, 四川省, 中国",
    "新都区, 成都市, 四川省, 中国",
    "青羊区, 成都市, 四川省, 中国",
]

OUTPUT_DIR = Path("./districts_excel")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "chengdu_districts_boundary.csv"   # ← 改成 .csv

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
HEADERS = {"User-Agent": "district-boundary-csv/1.0"}

# ==================== 下载并提取 WKT 边界 ====================
def download_district_boundary(place: str):
    params = {
        "q": place,
        "format": "geojson",
        "polygon_geojson": "1",
        "limit": "1",
        "countrycodes": "cn",
    }
    try:
        response = requests.get(NOMINATIM_URL, params=params, headers=HEADERS, timeout=30)
        response.raise_for_status()
        data = response.json()

        if not data.get("features"):
            logger.warning(f"未找到边界: {place}")
            return None, None

        feature = data["features"][0]
        geom_geojson = feature["geometry"]
        district_name = place.split(",")[0].strip()

        # GeoJSON → Shapely → WKT
        geom = shape(geom_geojson)
        if geom.is_empty:
            logger.warning(f"边界为空: {place}")
            return None, None

        wkt = geom.wkt          # 自动处理 Polygon / MultiPolygon
        return district_name, wkt

    except Exception as e:
        logger.error(f"处理失败 {place}: {e}")
        return None, None


# ==================== 主流程 ====================
if __name__ == "__main__":
    results = []
    success_count = 0

    logger.info("开始下载成都市各区边界（WKT Polygon 格式）...")
    for place in tqdm(districts, desc="下载行政区", unit="区"):
        name, wkt = download_district_boundary(place)
        if name and wkt:
            results.append({"区域名称": name, "区域边界": wkt})
            success_count += 1
        time.sleep(1.1)   # Nominatim 限流

    if not results:
        logger.error("所有区均下载失败！")
        exit(1)

    # ---------- 保存为 CSV ----------
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")   # ← CSV

    logger.success(f"全部完成！成功导出 {success_count}/{len(districts)} 个区")
    logger.info(f"CSV 文件已保存: {OUTPUT_FILE.resolve()}")

    # ---------- 打印预览 ----------
    print("\n" + "=" * 80)
    print("导出完成！每个区一个 Polygon（WKT 格式）")
    print("=" * 80)
    for _, row in df.iterrows():
        wkt_preview = (
            row["区域边界"][:70] + "..." if len(row["区域边界"]) > 70 else row["区域边界"]
        )
        print(f" {row['区域名称']:<6} → {wkt_preview}")
    print(f"\n文件路径: {OUTPUT_FILE.resolve()}")
    print(f"总行数: {len(df)} 行")
    print("=" * 80)