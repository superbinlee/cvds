# POI数据处理与相机模拟系统

## 概述

该文件夹包含三套不同的POI数据处理方案，用于将成都市POI数据转换为模拟监控相机部署方案。每套方案针对不同场景和需求设计，具有不同的处理策略和优化目标。

## 文件结构

```
.
├── process_poi.py      # 基础版：全量相机生成方案
├── process_poi_.py     # 终极版：22万相机流畅展示方案
└── process_poi__.py    # 智能版：边界相机智能部署方案
```

## 三套方案对比

### 1. 基础版 (process_poi.py)

#### 核心设计思路

- 对所有几何类型进行相机部署
- 根据几何特征采用不同密度策略
- 生成完整的相机部署方案

#### 处理逻辑

- **Point类型**：直接在点位置部署1个相机
- **LineString类型**：按150米间距部署相机
- **Polygon类型**：根据面积大小确定相机数量，沿边界均匀部署
- **MultiPolygon类型**：统一处理为边界点部署

#### 相机密度策略

```python
def get_camera_count(area_km2: float) -> int:
    if area_km2 < 0.01: return 4  # 小于100m²: 4个相机
    if area_km2 < 0.1: return 8  # 100m²-0.1km²: 8个相机
    if area_km2 < 1.0: return 16  # 0.1km²-1km²: 16个相机
    if area_km2 < 5.0: return 32  # 1km²-5km²: 32个相机
    return 64  # 大于5km²: 64个相机
```

#### 输出特点

- 生成完整的相机部署方案
- 包含所有几何类型的处理
- 适用于全面覆盖需求

### 2. 终极版 (process_poi_.py)

#### 核心设计思路

- 专为处理大规模相机数据优化
- 使用FastMarkerCluster提升地图渲染性能
- 保持基础版的处理逻辑但优化展示效果

#### 性能优化策略

- **FastMarkerCluster**：使用folium插件优化22万+点的渲染性能
- **回调函数**：动态生成标记样式，提升交互体验
- **聚类显示**：在地图缩放时自动聚类显示，避免浏览器卡顿

#### 输出特点

- 针对大规模数据优化
- 地图渲染流畅
- 适用于展示海量相机部署方案

### 3. 智能版 (process_poi__.py)

#### 核心设计思路

- 智能识别并剔除楼栋等小型建筑
- 仅在小区外边界部署相机
- 更贴近实际监控部署需求

#### 智能筛选机制

```python
def is_building(geom) -> bool:
    """识别并剔除楼栋（面积小于阈值的多边形）"""
    if not isinstance(geom, (Polygon, MultiPolygon)):
        return False
    area_m2 = geom.area * 111320 ** 2
    return area_m2 / 1e6 < BUILDING_AREA_THRESHOLD  # 默认0.05km²
```

#### 边界提取策略

- **外边界提取**：仅提取Polygon/MultiPolygon的外边界
- **周长计算**：根据边界周长确定相机数量
- **智能部署**：沿小区边界均匀部署相机

#### 相机部署策略

```python
def get_boundary_camera_count(perimeter_m: float) -> int:
    if perimeter_m < 200: return 4  # 小于200米: 4个相机
    if perimeter_m < 500: return 8  # 200-500米: 8个相机
    if perimeter_m < 1000: return 12  # 500-1000米: 12个相机
    if perimeter_m < 2000: return 16  # 1-2公里: 16个相机
    return 20  # 大于2公里: 20个相机
```

#### 输出特点

- 智能剔除无关建筑
- 仅在重要边界部署相机
- 更符合实际监控需求

## 共同特性

### 智能命名系统

所有方案都采用统一的智能命名策略：

```python
def smart_fill_name(row, idx, camera_id) -> str:
# 优先级依次为: name > brand+operator > brand+street > operator+street > category+street > category+district > category+id
```

### 数据输入输出

- **输入**：`./poi/chengdu_all_pois.xlsx` (WKT格式几何数据)
- **输出**：
    - Excel文件：包含相机坐标、名称、ID等信息
    - HTML地图：交互式可视化展示

### 地图可视化

- 支持多种底图切换(CartoDB, OpenStreetMap)
- 提供详细的信息弹窗
- 支持图层控制

## 使用建议

1. **全面覆盖需求**：使用 [process_poi.py](file://D:\root\projects\python\cvds\src\cvds\s1_mock_camera\ss3_process_poi\process_poi.py) 基础版
2. **大规模数据展示**：使用 [process_poi_.py](file://D:\root\projects\python\cvds\src\cvds\s1_mock_camera\ss3_process_poi\process_poi_.py) 终极版
3. **实际部署模拟**：使用 [process_poi__.py](file://D:\root\projects\python\cvds\src\cvds\s1_mock_camera\ss3_process_poi\process_poi__.py) 智能版

## 配置参数

各方案均支持自定义配置：

- [INPUT_DIR](file://D:\root\projects\python\cvds\src\cvds\s1_mock_camera\ss3_process_poi\process_poi.py#L17-L17)：输入目录
- [EXCEL_OUTPUT](file://D:\root\projects\python\cvds\src\cvds\s1_mock_camera\ss3_process_poi\process_poi.py#L18-L18)：Excel输出文件名
- [OUTPUT_MAP](file://D:\root\projects\python\cvds\src\cvds\s1_mock_camera\ss3_process_poi\process_poi.py#L19-L19)：地图输出文件名
- 各种阈值参数可根据实际需求调整

## 技术依赖

- geopandas：空间数据处理
- folium：地图可视化
- shapely：几何操作
- loguru：日志记录
- tqdm：进度显示