# 成都城市数据模拟系统 (CVDS)

## 概述

成都城市数据模拟系统 (Chengdu Virtual Data System, CVDS) 是一个完整的城市数据模拟平台，旨在通过真实地理数据构建虚拟城市环境。系统包含POI数据处理、道路网络分析和路径规划三大核心模块，为城市模拟、监控部署和交通分析提供数据支撑。

## 项目结构

```
cvds/
├── s1_mock_camera/           # 相机模拟模块
│   ├── ss1_download_poi/     # POI数据下载
│   ├── ss2_vis_poi/          # POI数据可视化
│   └── ss3_process_poi/      # 相机部署模拟
├── s2_mock_roadnet/          # 道路网络模块
│   ├── ss1_download_roadnet/ # 道路网络下载
│   ├── ss2_vis_roadnet/      # 道路网络可视化
│   └── ss3_find_path/        # 路径规划分析
└── __init__.py
```

## 相机模拟模块设计思路

### 1. 数据采集层 (ss1_download_poi)

#### 核心设计理念

构建真实城市POI数据基础，为后续模拟提供数据支撑。

#### 技术实现

- **多区域并行下载**：使用 `ThreadPoolExecutor` 并行下载成都市多个行政区数据
- **OSM标签体系**：支持 `amenity`, `shop`, `tourism` 等19种POI标签类型
- **数据清洗机制**：基于几何位置、名称和标签的去重策略
- **缓存优化**：本地缓存机制避免重复请求

```python
# 多标签并行下载策略
tags = {
    'amenity': True, 'shop': True, 'tourism': True,
    'leisure': True, 'highway': True, 'building': True,
    # ... 共19种标签
}
```

### 2. 数据可视化层 (ss2_vis_poi)

#### 核心设计理念

将抽象的POI数据转化为直观的地理可视化，便于分析和验证。

#### 技术实现

- **分区地图生成**：按行政区生成独立交互式地图
- **分类着色系统**：不同POI类型使用不同颜色标识
- **图层控制**：支持按类别显示/隐藏POI
- **信息弹窗**：点击POI查看详细属性信息

### 3. 相机部署模拟层 (ss3_process_poi)

#### 核心设计理念

基于真实POI数据模拟监控相机部署方案，提供三种不同策略满足不同需求。

#### 三套部署策略

##### 3.1 基础版 (process_poi.py) - 全量覆盖策略

**设计思路**：对所有POI几何类型进行全面覆盖部署

**几何处理策略**：

- **Point**：1:1部署，每个点部署1个相机
- **LineString**：按150米间距部署相机
- **Polygon/MultiPolygon**：根据面积确定相机数量，沿边界均匀部署

**适用场景**：需要全面监控覆盖的场景

##### 3.2 终极版 (process_poi_.py) - 大规模优化策略

**设计思路**：针对海量相机数据的性能优化

**关键技术**：

- **FastMarkerCluster**：使用folium插件优化22万+点渲染
- **动态聚类**：地图缩放时自动聚类显示
- **回调优化**：动态生成标记样式提升交互体验

**适用场景**：大规模数据展示和分析

##### 3.3 智能版 (process_poi__.py) - 智能筛选策略

**设计思路**：贴近实际监控部署需求的智能筛选

**核心算法**：

```python
def is_building(geom) -> bool:
    """智能识别并剔除楼栋（面积 < 0.05km²）"""
    area_m2 = geom.area * 111320 ** 2
    return area_m2 / 1e6 < BUILDING_AREA_THRESHOLD


def get_boundary_camera_count(perimeter_m: float) -> int:
    """根据边界周长智能确定相机数量"""
    if perimeter_m < 200: return 4
    if perimeter_m < 500: return 8
    # ... 智能分级
```

**部署策略**：

- 智能剔除小型建筑
- 仅在小区外边界部署相机
- 根据边界周长确定相机密度

**适用场景**：实际监控部署方案模拟

## 数据流转设计

```
OSM数据 → POI下载 → 数据清洗 → 相机部署 → 可视化展示
   ↓         ↓         ↓          ↓          ↓
网络层   文件层    内存层     业务层     展示层
```

## 核心技术栈

- **数据处理**：geopandas, pandas, shapely
- **可视化**：folium, leaflet.js
- **并发处理**：ThreadPoolExecutor
- **日志系统**：loguru
- **进度显示**：tqdm

## 系统特色

### 1. 模块化设计

各模块独立运行，可单独使用也可组合使用

### 2. 策略多样化

针对不同需求提供多种处理策略

### 3. 性能优化

大规模数据处理时采用聚类、缓存等优化技术

### 4. 智能化处理

自动识别建筑类型、智能命名、自适应密度部署

## 使用流程

1. **数据准备**：
   ```bash
   cd s1_mock_camera/ss1_download_poi
   python download_poi.py
   ```


2. **数据验证**：
   ```bash
   cd ../ss2_vis_poi
   python vis_poi.py
   ```


3. **相机部署**（选择任一策略）：
   ```bash
   cd ../ss3_process_poi
   python process_poi__.py  # 推荐智能版
   ```

## 输出成果

- **POI数据**：各行政区Excel文件 + 合并文件
- **可视化地图**：交互式HTML地图文件
- **相机部署方案**：Excel坐标文件 + 可视化地图
- **日志记录**：详细处理过程记录

该系统为城市模拟、安防规划、交通分析等领域提供了完整的数据基础和分析工具。