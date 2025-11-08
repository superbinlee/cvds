import folium
import geopandas as gpd
import pandas as pd
from loguru import logger


def visualize_road_network_folium(gpkg_path: str, layer: str = "edges", output_html: str = "hangzhou_roads.html"):
    """
    Read road network from GeoPackage and visualize using folium.

    Args:
        gpkg_path (str): Path to the GeoPackage file.
        layer (str): Layer to read (default: 'edges' for road segments).
        output_html (str): Path to save the interactive Folium map.
    """
    try:
        # Read the edges layer from GeoPackage
        logger.info(f"Reading {layer} layer from {gpkg_path}...")
        gdf_edges = gpd.read_file(gpkg_path, layer=layer)
        logger.info(f"Loaded {len(gdf_edges)} road segments")

        # Check available columns
        logger.info(f"Columns in GeoDataFrame: {list(gdf_edges.columns)}")

        # Initialize map centered on Binjiang District (approximate coordinates)
        logger.info("Creating interactive road network map with folium...")

        # 自动计算地图中心（取所有边的质心均值）
        gdf_centroids = gdf_edges.geometry.centroid
        center_lat = gdf_centroids.y.mean()
        center_lon = gdf_centroids.x.mean()
        logger.info(f"地图中心自动定位：[{center_lat:.6f}, {center_lon:.6f}]（成都）")

        m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="cartodbpositron")

        # Add roads to the map
        for idx, row in gdf_edges.iterrows():
            # Convert geometry to GeoJSON for the current row
            geo_j = gdf_edges.loc[[idx], ['geometry']].to_json()

            # Prepare popup content
            popup_content = ""
            if 'name' in row and pd.notna(row['name']):
                popup_content += f"Name: {row['name']}<br>"
            else:
                popup_content += "Name: N/A<br>"
            if 'highway' in row and pd.notna(row['highway']):
                popup_content += f"Highway Type: {row['highway']}<br>"
            else:
                popup_content += "Highway Type: N/A<br>"
            popup_content += f"Length: {row.get('length', 'N/A')} meters"

            # Define style based on highway type
            def style_function(feature, highway_type=row.get('highway', None)):
                # Default style
                style = {'color': 'blue', 'weight': 2}
                if highway_type:
                    if highway_type in ['motorway', 'primary']:
                        style = {'color': 'red', 'weight': 4}
                    elif highway_type in ['secondary', 'tertiary']:
                        style = {'color': 'orange', 'weight': 3}
                    elif highway_type == 'residential':
                        style = {'color': 'green', 'weight': 2}
                return style

            # Add GeoJSON layer with popup
            folium.GeoJson(
                geo_j,
                style_function=style_function,
                popup=folium.Popup(popup_content, max_width=300)
            ).add_to(m)

        # Save interactive map
        m.save(output_html)
        logger.info(f"Interactive map saved as {output_html}")

        return gdf_edges

    except Exception as e:
        logger.error(f"===== 可视化失败 =====\n错误详细信息: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        # 配置参数
        GPKG_PATH = "Chengdu_all_road_network.gpkg"
        LAYER = "edges"
        OUTPUT_HTML = "./vis/vis_roads.html"

        # 执行可视化
        gdf = visualize_road_network_folium(GPKG_PATH, LAYER, OUTPUT_HTML)
        logger.info(f"Visualization completed for {len(gdf)} road segments")

    except Exception as e:
        logger.error(f"===== 可视化失败 =====\nError: {str(e)}")
        exit(1)
