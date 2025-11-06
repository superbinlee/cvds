import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Union

import geopandas as gpd
import osmnx as ox
import pandas as pd
from loguru import logger
from shapely import to_wkt

warnings.filterwarnings("ignore", category=UserWarning)


def download_pois_for_tag(place_name: str, tag_key: str, tag_value: Union[bool, str, List[str]], retries: int = 3, delay: int = 5) -> gpd.GeoDataFrame:
    """
    Helper function to download POIs for a single tag key-value pair with retry mechanism.

    Args:
        place_name (str): The name of the place to query.
        tag_key (str): OSM tag key (e.g., 'amenity').
        tag_value (Union[bool, str, List[str]]): OSM tag value(s) or True for all values.
        retries (int): Number of retry attempts for failed queries.
        delay (int): Delay between retries in seconds.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame of POIs for the given tag.
    """
    for attempt in range(retries):
        try:
            logger.info(f"Downloading POIs for {place_name}, {tag_key}={tag_value} (Attempt {attempt + 1}/{retries})")
            pois = ox.features_from_place(place_name, tags={tag_key: tag_value})
            logger.info(f"Identified {len(pois)} geometries for {place_name}, {tag_key}={tag_value}")
            if not pois.empty:
                pois['tag_key'] = tag_key
                pois['tag_value'] = str(tag_value)
                pois['district'] = place_name.split(',')[0]  # Add district name for tracking
            return pois
        except Exception as e:
            logger.warning(f"Error downloading POIs for {place_name}, {tag_key}={tag_value}: {str(e)}")
            if attempt < retries - 1:
                logger.info(f"Retrying after {delay} seconds...")
                time.sleep(delay)
            else:
                logger.warning(f"Failed after {retries} attempts for {place_name}, {tag_key}={tag_value}")
                return gpd.GeoDataFrame()


def download_chengdu_poi(
        districts: List[str],
        tags: Dict[str, Union[bool, str, List[str]]],
        batch_size: int = 1000,
        output_dir: str = "./",
        max_workers: int = 4,
        keep_columns: List[str] = None,
        retries: int = 3,
        retry_delay: int = 5
) -> gpd.GeoDataFrame:
    """
    Download POIs for specified districts in Chengdu, save to Excel files with geometry in WKT format,
    and support parallel downloading for efficiency.

    Args:
        districts (List[str]): List of district names to query (e.g., ["锦江区, 成都市, 四川省, 中国"]).
        tags (dict): OSM tags to filter POIs (e.g., {'amenity': True}).
        batch_size (int): Number of features to process per batch (for logging purposes).
        output_dir (str): Directory to save the output Excel files.
        max_workers (int): Number of parallel workers for downloading POIs.
        keep_columns (list): List of columns to keep in the output (e.g., ['name', 'addr:street']).
        retries (int): Number of retry attempts for failed queries.
        retry_delay (int): Delay between retries in seconds.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing all downloaded POIs with a 'category' column.
    """
    try:
        logger.info(f"Starting POI download for {len(districts)} districts in Chengdu...")
        all_pois = gpd.GeoDataFrame()

        for district in districts:
            logger.info(f"Processing district: {district}")
            district_pois = gpd.GeoDataFrame()
            batch_count = 1

            # Prepare tasks for parallel downloading
            tasks = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for tag_key, tag_value in tags.items():
                    tasks.append(executor.submit(download_pois_for_tag, district, tag_key, tag_value, retries, retry_delay))

                # Collect results for the district
                for future in as_completed(tasks):
                    pois = future.result()
                    if not pois.empty:
                        district_pois = pd.concat([district_pois, pois], ignore_index=True)
                        logger.info(f"Batch {batch_count} for {district}: Added {len(pois)} features")
                        logger.info(f"类别 {pois['tag_key'].iloc[0]} 批次 {batch_count} 下载 {len(pois)} 个POI")
                    else:
                        logger.info(f"Batch {batch_count} for {district}: No POIs found")
                    batch_count += 1

            # Remove duplicates for the district
            if not district_pois.empty:
                district_pois = district_pois.drop_duplicates(subset=['geometry', 'name', 'tag_key', 'tag_value'])
                logger.info(f"After deduplication for {district}: {len(district_pois)} POIs remain")

                # Define possible category columns
                category_cols = [
                    'amenity', 'shop', 'tourism', 'leisure', 'highway', 'building', 'healthcare',
                    'public_transport', 'access', 'office', 'craft', 'sport', 'religion', 'emergency',
                    'landuse', 'natural', 'man_made', 'military', 'aeroway', 'railway', 'waterway'
                ]

                # Filter only columns that exist
                existing_cols = [col for col in category_cols if col in district_pois.columns]
                logger.info(f"Columns in {district} POIs: {list(district_pois.columns)}")
                logger.info(f"Using category columns: {existing_cols}")

                # Assign category column
                if existing_cols:
                    district_pois["category"] = district_pois[existing_cols].bfill(axis=1).iloc[:, 0].fillna("unknown")
                else:
                    logger.warning(f"No category columns found for {district}. Assigning 'unknown'.")
                    district_pois["category"] = "unknown"

                # Keep specified columns
                if keep_columns:
                    keep_cols = [col for col in keep_columns if col in district_pois.columns] + ['geometry', 'category', 'tag_key', 'tag_value', 'district']
                    district_pois = district_pois[keep_cols]
                    logger.info(f"Retained columns for {district}: {keep_cols}")

                # Save district-specific Excel
                try:
                    export_df = district_pois.copy()
                    export_df['geometry'] = export_df['geometry'].apply(to_wkt)
                    district_name = district.split(',')[0].strip()
                    output_excel = f"{output_dir}/chengdu_{district_name}_pois.xlsx"
                    export_df.to_excel(output_excel, index=False)
                    logger.info(f"Successfully saved POIs for {district} to {output_excel}")
                except Exception as e:
                    logger.error(f"Failed to save Excel for {district}: {str(e)}")
                    raise

                # Append to all_pois
                all_pois = pd.concat([all_pois, district_pois], ignore_index=True)

        # Save combined Excel
        if not all_pois.empty:
            logger.info(f"总计下载 {len(all_pois)} 个POI数据（未去重）")
            all_pois = all_pois.drop_duplicates(subset=['geometry', 'name', 'tag_key', 'tag_value', 'district'])
            logger.info(f"After final deduplication: {len(all_pois)} POIs remain")
            try:
                export_df = all_pois.copy()
                export_df['geometry'] = export_df['geometry'].apply(to_wkt)
                combined_output = f"{output_dir}/chengdu_all_pois.xlsx"
                export_df.to_excel(combined_output, index=False)
                logger.info(f"Successfully saved combined POIs to {combined_output}")
            except Exception as e:
                logger.error(f"Failed to save combined Excel: {str(e)}")
                raise
        else:
            logger.warning("No POIs to save for any district.")

        return all_pois

    except Exception as e:
        logger.error(f"===== 下载失败 =====\n错误详细信息: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        # Define the six specified districts in Chengdu
        districts = [
            # "温江区, 成都市, 四川省, 中国",
            "成华区, 成都市, 四川省, 中国",
            "郫都区, 成都市, 四川省, 中国",
            "新都区, 成都市, 四川省, 中国",
            "金牛区, 成都市, 四川省, 中国",
            "青羊区, 成都市, 四川省, 中国",
        ]

        # OSM tags (unchanged) https://wiki.openstreetmap.org/wiki/Map_features
        tags = {
            'amenity': True,          # 设施标签
            'shop': True,             # 商店标签
            'tourism': True,          # 旅游标签
            'leisure': True,          # 休闲标签
            'highway': True,          # 道路标签
            'building': True,         # 建筑标签
            'healthcare': True,       # 医疗保健标签
            'public_transport': True, # 公共交通标签
            'office': True,           # 办公室标签
            'craft': True,            # 工艺标签
            'sport': True,            # 体育标签
            'religion': True,         # 宗教标签
            'emergency': True,        # 紧急服务标签
            'landuse': True,          # 土地利用标签
            'natural': True,          # 自然标签
            'man_made': True,         # 人造设施标签
            'military': True,         # 军事标签
            'aeroway': True,          # 航空标签
            'railway': True,          # 铁路标签
            'waterway': True          # 水路标签
        }

        # Columns to keep in the output
        keep_columns = [
            'name', 'addr:street', 'addr:city', 'addr:country', 'phone', 'website',
            'opening_hours', 'access', 'destination', 'brand', 'operator'
        ]

        # Download POIs and save to Excel
        output_dir = "./poi"
        chengdu_pois = download_chengdu_poi(
            districts=districts,
            tags=tags,
            output_dir=output_dir,
            max_workers=4,
            keep_columns=keep_columns,
            retries=3,
            retry_delay=5
        )
        logger.info(f"Successfully downloaded and processed {len(chengdu_pois)} POIs across all districts")

    except Exception as e:
        logger.error(f"===== 下载失败 =====\nError: {str(e)}")
        exit(1)
