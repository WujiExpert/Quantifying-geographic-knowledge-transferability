# 地理知识可迁移性分析项目使用说明

## 环境准备

### 1. 安装Python依赖

```bash
# 安装主要依赖包
pip install numpy pandas matplotlib seaborn scikit-learn scipy tqdm openpyxl

# 或者使用requirements.txt
pip install -r requirements.txt
```

### 2. 项目结构

确保项目目录结构如下：
```
geoTransferAnalysis/
├── data/                    # 数据文件
├── model/                   # 核心代码
│   ├── main_opt.py         # 主程序
│   └── ...
├── results/                 # 结果输出
└── README.md               # 项目文档
```

## 快速开始

### 步骤1：准备数据

将您的数据文件放入 `data/` 目录，当前已有的示例数据如下：

- **土壤数据**: `soil_train.csv`
- **犯罪数据**: `community_crime_stats.xlsx`
- **SDP数据**: `ecpr_case.xlsx` 和 `region.xlsx`

### 步骤2：运行基础地理知识可迁移性评估分析

编辑 `model/main_opt.py`，在 `if __name__ == "__main__":` 部分取消注释相应的分析函数：

```python
# 土壤数据分析
print("Running basic analyses...")
soil_result = run_soil_analysis()
# crime_result = run_crime_analysis()
# sdp_result = run_sdp_analysis()
```

然后运行：
```bash
python model/main_opt.py
```

### 步骤3：地理知识可迁移关键影响因子分析

编辑 `model/main_opt.py`，在 `if __name__ == "__main__":` 部分取消注释相应的分析函数：

```python
# 土壤数据关键因子分析
# soil_key_result = run_soil_key_factor_analysis('climate_type', 'categorical')

# 犯罪数据关键因子分析
# crime_key_result = run_crime_key_factor_analysis('TotIncome', 'numerical', 3, 'kmeans')

# SDP数据关键因子分析
# sdp_key_result = run_sdp_key_factor_analysis('climate_type_final', 'categorical', top_k=5)
```

### 步骤4：查看结果

分析完成后，结果将保存在 `results/` 目录中：

- **预测结果**: `prediction_sdp_with_groups_{key_factor}.xlsx`
- **统计检验**: `significance_test_{key_factor}.json`
- **可视化图表**: `figures/` 目录下的各种图表



## 常用参数配置

### 函数调用示例

#### 基础分析
```python
# 土壤分析
soil_result = run_soil_analysis()

# 犯罪分析
crime_result = run_crime_analysis()

# SDP分析
sdp_result = run_sdp_analysis()
```

#### 关键因子分析
```python
# 土壤关键因子分析 (使用默认参数)
soil_key_result = run_soil_key_factor_analysis()  # 默认: clay_S, numerical, 3, kmeans

# 土壤关键因子分析 (自定义参数)
soil_key_result = run_soil_key_factor_analysis(
    key_factor='climate_type',      # 关键因子名称
    factor_type='categorical',      # 因子类型: 'numerical' 或 'categorical'
    num_classes=3,                  # 分组数量 (仅数值型)
    group_method='kmeans'           # 分组方法: 'kmeans', 'quantile', 'jenks'
)

# 犯罪关键因子分析 (使用默认参数)
crime_key_result = run_crime_key_factor_analysis()  # 默认: RentPct, numerical, 3, kmeans

# 犯罪关键因子分析 (自定义参数)
crime_key_result = run_crime_key_factor_analysis(
    key_factor='TotIncome',
    factor_type='numerical',
    num_classes=3,
    group_method='kmeans'
)

# SDP关键因子分析 (使用默认参数)
sdp_key_result = run_sdp_key_factor_analysis()  # 默认: climate_type_final, categorical, 4, kmeans, 5

# SDP关键因子分析 (自定义参数)
sdp_key_result = run_sdp_key_factor_analysis(
    key_factor='climate_type_final',
    factor_type='categorical',
    top_k=5                         # TOP-K评估参数
)
```

### 分组设置

```python
# 数值型因子分组
factor_type='numerical'
num_classes=3  # 或 4, 5
group_method='kmeans'  # 'kmeans', 'quantile', 'jenks'

# 分类型因子分组
factor_type='categorical'
```

### 评估设置

```python
# TOP-K评估 (多分类)
top_k=5  # 或 3, 10

# Bootstrap设置
n_bootstrap=1000  # 或 500, 2000
alpha=0.05  # 显著性水平
```

## 数据格式示例

### 土壤数据格式
```csv
id,clay_S,sand_S,Elevation,OC,LU
1,0.25,0.45,120.5,1.2,Forest
2,0.30,0.40,150.2,1.5,Agriculture
...
```

### 多分类数据格式
```csv
id,feature1,feature2,patterns
1,0.5,0.3,"[1,2,3]"
2,0.7,0.4,"[2,4]"
...
```

## 结果解释

### 评估指标含义

- **RMSE**: 均方根误差，越小越好
- **R²**: 决定系数，越接近1越好
- **TOP-K准确率**: 前K个预测中包含真实标签的比例
- **F1分数**: 精确率和召回率的调和平均
- **Cohen's f²**: 决定系数效应量
- **SF1G**: F1分数的效应量

### 基于环境因子样本分组对地理知识迁移影响评估

- **组内预测**: 同一组内的预测性能
- **组间预测**: 不同组间的预测性能
- **迁移性影响程度**: 组间与组内性能的差异，以及统计检验结果

### 统计显著性

- **p值 < 0.05**: 差异具有统计显著性
- **置信区间**: 不包含0表示差异显著

## 快速使用示例

### 土壤分析示例

1. 编辑 `model/main_opt.py`：
```python
if __name__ == "__main__":
    # 基础分析
    print("Running basic analyses...")
    soil_result = run_soil_analysis()
    # crime_result = run_crime_analysis()
    # sdp_result = run_sdp_analysis()
    
    # 关键因子分析
    # print("\nRunning key factor analyses...")
    # soil_key_result = run_soil_key_factor_analysis('climate_type', 'categorical')
    # crime_key_result = run_crime_key_factor_analysis('TotIncome', 'numerical', 3, 'kmeans')
    # sdp_key_result = run_sdp_key_factor_analysis('climate_type_final', 'categorical', top_k=5)
    
    # 自定义多分类配置示例
    # custom_multiclass_config = create_multiclass_config(
    #     data_paths=['../data/region.xlsx', '../data/ecpr_case.xlsx'],
    #     target_variable='patterns',
    #     feature_variables=['precipitation', 'average_altitude'],
    #     top_k=3
    # )
    # custom_multiclass_result = run_multiclass_key_factor_analysis(
    #     custom_multiclass_config, 'climate_type_final', 'categorical', 3, 'jenks'
    # )
    
    # 寻找最优阈值
    # optimal_kappa = find_optimal_kappa()
    
    print("\nAll analyses completed successfully!")
```

2. 运行分析：
```bash
python model/main_opt.py
```

3. 查看结果：
- 基础分析结果：`results/basic_analysis_results.csv`
- 关键因子分析结果：`results/{key_factor}_group_analysis_results.csv`
- 统计检验结果：`results/significance_test_{key_factor}.json`