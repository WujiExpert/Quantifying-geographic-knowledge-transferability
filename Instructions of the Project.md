# Geographic Knowledge Transferability Analysis Project - User Guide

## Environment Setup

### 1. Install Python Dependencies

```bash
# Install main dependency packages
pip install numpy pandas matplotlib seaborn scikit-learn scipy tqdm openpyxl

# Or use requirements.txt
pip install -r requirements.txt
```

### 2. Project Structure

Ensure the project directory structure is as follows:
```
geoTransferAnalysis/
├── data/                    # Data files
├── model/                   # Core code
│   ├── main_opt.py         # Main program
│   └── ...
├── results/                 # Results output
└── README.md               # Project documentation
```

## Quick Start

### Step 1: Prepare Data

Place your data files in the `data/` directory. Current sample data includes:

- **Soil Data**: `soil_train.csv`
- **Crime Data**: `community_crime_stats.xlsx`
- **SDP Data**: `ecpr_case.xlsx` and `region.xlsx`

### Step 2: Run Basic Geographic Knowledge Transferability Assessment Analysis

Edit `model/main_opt.py`, uncomment the corresponding analysis functions in the `if __name__ == "__main__":` section:

```python
# Soil data analysis
print("Running basic analyses...")
soil_result = run_soil_analysis()
# crime_result = run_crime_analysis()
# sdp_result = run_sdp_analysis()
```

Then run:
```bash
python model/main_opt.py
```

### Step 3: Geographic Knowledge Transfer Key Impact Factor Analysis

Edit `model/main_opt.py`, uncomment the corresponding analysis functions in the `if __name__ == "__main__":` section:

```python
# Soil data key factor analysis
# soil_key_result = run_soil_key_factor_analysis('climate_type', 'categorical')

# Crime data key factor analysis
# crime_key_result = run_crime_key_factor_analysis('TotIncome', 'numerical', 3, 'kmeans')

# SDP data key factor analysis
# sdp_key_result = run_sdp_key_factor_analysis('climate_type_final', 'categorical', top_k=5)
```

### Step 4: View Results

After analysis is complete, results will be saved in the `results/` directory:

- **Prediction Results**: `prediction_sdp_with_groups_{key_factor}.xlsx`
- **Statistical Tests**: `significance_test_{key_factor}.json`
- **Visualization Charts**: Various charts in the `figures/` directory



## Common Parameter Configuration

### Function Call Examples

#### Basic Analysis
```python
# Soil analysis
soil_result = run_soil_analysis()

# Crime analysis
crime_result = run_crime_analysis()

# SDP analysis
sdp_result = run_sdp_analysis()
```

#### Key Factor Analysis
```python
# Soil key factor analysis (using default parameters)
soil_key_result = run_soil_key_factor_analysis()  # Default: clay_S, numerical, 3, kmeans

# Soil key factor analysis (custom parameters)
soil_key_result = run_soil_key_factor_analysis(
    key_factor='climate_type',      # Key factor name
    factor_type='categorical',      # Factor type: 'numerical' or 'categorical'
    num_classes=3,                  # Number of groups (numerical only)
    group_method='kmeans'           # Grouping method: 'kmeans', 'quantile', 'jenks'
)

# Crime key factor analysis (using default parameters)
crime_key_result = run_crime_key_factor_analysis()  # Default: RentPct, numerical, 3, kmeans

# Crime key factor analysis (custom parameters)
crime_key_result = run_crime_key_factor_analysis(
    key_factor='TotIncome',
    factor_type='numerical',
    num_classes=3,
    group_method='kmeans'
)

# SDP key factor analysis (using default parameters)
sdp_key_result = run_sdp_key_factor_analysis()  # Default: climate_type_final, categorical, 4, kmeans, 5

# SDP key factor analysis (custom parameters)
sdp_key_result = run_sdp_key_factor_analysis(
    key_factor='climate_type_final',
    factor_type='categorical',
    top_k=5                         # TOP-K evaluation parameter
)
```

### Grouping Settings

```python
# Numerical factor grouping
factor_type='numerical'
num_classes=3  # or 4, 5
group_method='kmeans'  # 'kmeans', 'quantile', 'jenks'

# Categorical factor grouping
factor_type='categorical'
```

### Evaluation Settings

```python
# TOP-K evaluation (multiclass)
top_k=5  # or 3, 10

# Bootstrap settings
n_bootstrap=1000  # or 500, 2000
alpha=0.05  # Significance level
```

## Data Format Examples

### Soil Data Format
```csv
id,clay_S,sand_S,Elevation,OC,LU
1,0.25,0.45,120.5,1.2,Forest
2,0.30,0.40,150.2,1.5,Agriculture
...
```

### Multi-class Data Format
```csv
id,feature1,feature2,patterns
1,0.5,0.3,"[1,2,3]"
2,0.7,0.4,"[2,4]"
...
```

## Result Interpretation

### Evaluation Metric Meanings

- **RMSE**: Root Mean Square Error, smaller is better
- **R²**: Coefficient of Determination, closer to 1 is better
- **TOP-K Accuracy**: Proportion of true labels in top K predictions
- **F1 Score**: Harmonic mean of precision and recall
- **Cohen's f²**: Effect size of coefficient of determination
- **SF1G**: Effect size of F1 score

### Assessment of Geographic Knowledge Transfer Impact Based on Environmental Factor Sample Grouping

- **Within-group Prediction**: Prediction performance within the same group
- **Cross-group Prediction**: Prediction performance across different groups
- **Transferability Impact Level**: Difference between cross-group and within-group performance, along with statistical test results

### Statistical Significance

- **p-value < 0.05**: Difference is statistically significant
- **Confidence Interval**: Not containing 0 indicates significant difference

## Quick Usage Example

### Soil Analysis Example

1. Edit `model/main_opt.py`:
```python
if __name__ == "__main__":
    # Basic analysis
    print("Running basic analyses...")
    soil_result = run_soil_analysis()
    # crime_result = run_crime_analysis()
    # sdp_result = run_sdp_analysis()
    
    # Key factor analysis
    # print("\nRunning key factor analyses...")
    # soil_key_result = run_soil_key_factor_analysis('climate_type', 'categorical')
    # crime_key_result = run_crime_key_factor_analysis('TotIncome', 'numerical', 3, 'kmeans')
    # sdp_key_result = run_sdp_key_factor_analysis('climate_type_final', 'categorical', top_k=5)
    
    # Example of using unified framework for any multiclass configuration
    # custom_multiclass_config = create_multiclass_config(
    #     data_paths=['../data/region.xlsx', '../data/ecpr_case.xlsx'],
    #     target_variable='patterns',
    #     feature_variables=['precipitation', 'average_altitude'],
    #     top_k=3
    # )
    # custom_multiclass_result = run_multiclass_key_factor_analysis(
    #     custom_multiclass_config, 'climate_type_final', 'categorical', 3, 'jenks'
    # )
    
    # Find optimal threshold
    # optimal_kappa = find_optimal_kappa()
    
    print("\nAll analyses completed successfully!")
```

2. Run analysis:
```bash
python model/main_opt.py
```

3. View results:
- Basic analysis results: `results/basic_analysis_results.csv`
- Key factor analysis results: `results/{key_factor}_group_analysis_results.csv`
- Statistical test results: `results/significance_test_{key_factor}.json` 