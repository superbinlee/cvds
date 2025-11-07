# 成都市道路网络数据下载工具

## 概述

[download_roadnet.py](file://D:\root\projects\python\cvds\src\cvds\s2_mock_roadnet\ss1_download_roadnet\download_roadnet.py) 是一个专门用于下载成都市各区道路网络数据的Python脚本。该工具通过调用OpenStreetMap (OSM) API获取道路网络数据，并支持按行政区划分批下载和自动合并功能。

## 主要功能

1. **批量下载**：支持成都市多个行政区的道路网络数据批量下载
2. **数据格式**：下载数据保存为GeoPackage ([.gpkg](file://D:\root\projects\python\cvds\src\cvds\s2_mock_roadnet\ss2_vis_roadnet\Chengdu_all_road_network.gpkg)) 格式，便于GIS软件处理
3. **自动合并**：下载完成后自动将各区数据合并为一个完整的成都市道路网络文件
4. **错误处理**：具备下载失败重试机制和错误日志记录
5. **频率控制**：自动添加延迟避免触发OSM API频率限制

## 使用方法

直接运行脚本即可开始下载：

```bash
python download_roadnet.py
```


脚本会自动处理以下步骤：
1. 依次下载各行政区的道路网络数据
2. 将每个区的数据保存为独立的GeoPackage文件
3. 合并所有成功下载的区域数据为一个完整文件

## 配置说明

### 下载区域配置
在脚本中可以修改 `districts` 列表来指定需要下载的行政区：

```python
districts = [
    "锦江区, 成都市, 四川省, 中国",
    "青羊区, 成都市, 四川省, 中国",
    # ... 其他区域
]
```


### 道路网络类型
通过 `NETWORK_TYPE` 参数控制下载的道路类型，支持OSM标准的道路标签过滤。

## 输出文件

脚本会在运行目录下生成以下文件：
- 各区独立的道路网络文件：`[区名]_road_network.gpkg`
- 合并后的全市道路网络文件：[Chengdu_all_road_network.gpkg](file://D:\root\projects\python\cvds\src\cvds\s2_mock_roadnet\ss2_vis_roadnet\Chengdu_all_road_network.gpkg)

## 技术细节

### 下载机制
- 使用OSM Overpass API进行数据查询
- 每次下载间隔2秒避免频率限制
- 支持下载失败后的重试机制

### 数据处理
- 采用GeoPackage格式存储，支持空间数据和属性数据
- 自动处理坐标系转换
- 合并过程中保持数据完整性

## 依赖库

- `osmnx` - OSM数据获取和处理
- `geopandas` - 空间数据处理
- `loguru` - 日志记录

## 注意事项

1. 确保网络连接稳定，道路网络数据量较大
2. 首次运行可能需要较长时间，取决于网络速度和区域大小
3. 下载过程中请勿频繁中断，可能导致数据不完整
4. 如遇下载失败，可单独重新运行失败的区域
5. 生成的GeoPackage文件可直接在QGIS、ArcGIS等GIS软件中打开使用