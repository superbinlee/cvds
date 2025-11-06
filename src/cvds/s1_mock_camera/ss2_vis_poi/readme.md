# POI数据可视化工具

## 概述

这是一个用于将成都市POI数据可视化到交互式地图上的Python工具集。工具使用`folium`库生成基于Leaflet的HTML地图文件，支持按行政区划分展示POI数据，并提供统一的全区域视图。

## 文件结构

```
.
├── vis_poi.py          # POI数据可视化主程序
├── vis_poi_label.py    # POI数据标注可视化程序
└── README.md           # 本说明文档
```

## 功能特性

1. **分区地图生成**：为每个行政区生成独立的交互式地图
2. **全域地图整合**：生成包含所有POI的成都市总览地图
3. **分类着色显示**：不同类别的POI使用不同颜色标识
4. **图层控制**：支持按类别显示/隐藏POI
5. **行政区边界叠加**：可选显示行政区划边界
6. **信息弹窗**：点击POI可查看详细信息

## 使用方法

### 安装依赖

```bash
pip install pandas geopandas folium openpyxl loguru
```

### 运行程序

1. 确保有符合格式要求的POI数据文件（Excel格式）
2. 修改代码中的文件路径：
   ```python
   POI_FILE = "./vis/chengdu_all_pois.xlsx"  # 输入POI文件路径
   OUTPUT_DIR = "./vis/maps"                 # 输出地图文件目录
   ```

3. 运行任一可视化脚本：
   ```bash
   python vis_poi.py
   # 或
   python vis_poi_label.py
   ```

## 输入数据格式

输入文件应为Excel格式，至少包含以下列：

- `geometry`: POI的几何信息（WKT格式）
- `category`: POI类别（可选）
- `district`: 所属行政区（可选，若缺失则自动推断）

## 输出结果

生成的地图文件将保存在指定的输出目录中：

- 各行政区独立地图文件：`map_<区名>_pois.html`
- 全市汇总地图文件：[map_Chengdu_ALL_pois.html](file://D:\root\projects\python\cvds\src\cvds\s1_mock_camera\ss1_vis_poi\vis\maps\map_Chengdu_ALL_pois.html)

## 颜色编码

不同类别的POI使用特定颜色显示：

- 公园相关（park, playground等）: 绿色系
- 商业设施（hotel, supermarket等）: 红橙色系
- 体育设施: 蓝紫色系
- 未知类别: 灰色

## 注意事项

1. 确保输入Excel文件存在且格式正确
2. `geometry`列必须为WKT格式字符串
3. 网络连接有助于地图底图加载
4. 生成的HTML文件可在浏览器中打开查看交互式地图