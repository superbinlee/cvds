<xaiArtifact artifact_id="c8a99b98-e155-4ef0-a5c9-a805cb40a5a0" artifact_version_id="abff0e9b-b069-45b6-b8ae-5250550ee9e3" title="README.md" contentType="text/markdown">

# 成都POI数据下载脚本说明

## 概述
`download_chengdu_pois.py` 是一个使用 Python 编写的脚本，用于从 OpenStreetMap (OSM) 下载成都市六个指定行政区（锦江区、青羊区、金牛区、武侯区、成华区、龙泉驿区）的兴趣点 (POI) 数据。脚本支持并行下载、数据去重、分类整理，并将结果保存为 Excel 文件。

## 主要功能
1. **多区域POI下载**：针对成都市的六个区（锦江区、青羊区、金牛区、武侯区、成华区、龙泉驿区）批量下载 OSM 数据。
2. **支持多种OSM标签**：通过指定的 OSM 标签（如 `amenity`, `shop`, `tourism` 等）获取丰富的 POI 数据。
3. **并行处理**：使用多线程（`ThreadPoolExecutor`）加速数据下载，提高效率。
4. **数据清洗**：
   - 去除重复的 POI（基于几何位置、名称和标签）。
   - 根据 OSM 标签自动分配类别（如 `amenity`, `shop` 等）。
   - 保留用户指定的字段（如名称、地址、电话等）。
5. **输出到Excel**：
   - 为每个区生成单独的 Excel 文件（例如 `chengdu_锦江区_pois.xlsx`）。
   - 生成包含所有区域数据的合并 Excel 文件（`chengdu_all_pois.xlsx`）。
   - 几何数据以 WKT 格式保存，方便后续分析。
6. **错误重试机制**：支持下载失败时自动重试（默认3次，每次间隔5秒）。
7. **日志记录**：使用 `loguru` 记录下载和处理过程，便于调试和监控。

## 使用方法
1. 确保安装依赖库：`geopandas`, `osmnx`, `pandas`, `loguru`, `shapely`。
2. 运行脚本，脚本会自动下载六个区的 POI 数据，并保存为 Excel 文件。
3. 可通过修改 `districts`, `tags`, `keep_columns`, 和 `output_dir` 参数自定义下载范围、标签、输出字段和保存路径。

## 输出文件
- **单区文件**：每个区的 POI 数据保存为单独的 Excel 文件，文件名格式为 `chengdu_[区名]_pois.xlsx`。
- **合并文件**：所有区的 POI 数据合并保存为 `chengdu_all_pois.xlsx`。
- 每行数据包含 POI 的几何信息（WKT格式）、名称、地址、类别等字段。

## 注意事项
- 确保网络连接稳定，因为 OSM 数据需要在线下载。
- 检查输出目录的写权限以确保 Excel 文件能正常保存。
- 可根据需要调整 `max_workers`（并行线程数）以平衡速度和系统资源。

</xaiArtifact>