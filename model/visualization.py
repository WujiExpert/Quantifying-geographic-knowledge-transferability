import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

from scipy import stats


plt.rcParams['font.family'] = 'Times New Roman'
# Fix negative sign display issue
plt.rcParams['axes.unicode_minus'] = False

def plot_violin_charts(data_path):
    # Load data
    data = pd.read_csv(data_path)

    # Set Chinese font
    plt.rcParams['font.family'] = 'Hiragino Sans GB'

    # Create output directory
    output_dir = r"D:\dev\python\geoKnowledgeTransferValidate\results\figures"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Violin plot of prediction errors by land use type (LU)
    plt.figure(figsize=(12, 8))
    sns.violinplot(x='LU', y='diff', data=data, inner='box', palette='Set3')
    plt.title('Prediction Error Distribution by Land Use Type')
    plt.xlabel('Land Use Type')
    plt.ylabel('Prediction Error Percentage (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lu_error_violin.png"), dpi=300, bbox_inches='tight')
    plt.show()

    # 2. Violin plot of prediction errors grouped by uncertainty level
    # Divide uncertainty into 4 groups
    data['uncertainty_level'] = pd.qcut(data['uncertainty'],
                                        q=4,
                                        labels=['Low Uncertainty', 'Medium-Low Uncertainty', 'Medium-High Uncertainty', 'High Uncertainty'])

    plt.figure(figsize=(12, 8))
    sns.violinplot(x='uncertainty_level', y='diff', data=data, inner='box', palette='viridis')
    plt.title('Prediction Error Distribution by Uncertainty Level')
    plt.xlabel('Uncertainty Level')
    plt.ylabel('Prediction Error Percentage (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "uncertainty_error_violin.png"), dpi=300, bbox_inches='tight')
    plt.show()

    # 3. Violin plot of prediction errors grouped by elevation
    # Divide elevation into 5 groups
    data['elevation_group'] = pd.qcut(data['Elevation'],
                                      q=5,
                                      labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

    plt.figure(figsize=(12, 8))
    sns.violinplot(x='elevation_group', y='diff', data=data, inner='box', palette='coolwarm')
    plt.title('Prediction Error Distribution by Elevation')
    plt.xlabel('Elevation Group')
    plt.ylabel('Prediction Error Percentage (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "elevation_error_violin.png"), dpi=300, bbox_inches='tight')
    plt.show()

    # 4. Combined violin plot: comparison of predicted vs actual values by land use type
    plt.figure(figsize=(15, 10))

    # Create a long-format DataFrame for seaborn plotting
    long_data = pd.DataFrame({
        'LU': np.concatenate([data['LU'], data['LU']]),
        'value': np.concatenate([data['OC'], data['predicted']]),
        'type': np.concatenate([['Actual'] * len(data), ['Predicted'] * len(data)])
    })

    sns.violinplot(x='LU', y='value', hue='type', data=long_data,
                   split=True, inner='quart', palette={'Actual': 'lightblue', 'Predicted': 'lightgreen'})
    plt.title('Actual vs Predicted Value Distribution Comparison by Land Use Type')
    plt.xlabel('Land Use Type')
    plt.ylabel('Value (Log Scale)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Data Type')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lu_prediction_violin.png"), dpi=300, bbox_inches='tight')
    plt.show()

def visualize_correlation_scatter(data_path, target_colum):
    # Load data
    data = pd.read_csv(data_path)

    # 1. Scatter plot of predicted vs. observed values (with uncertainty)
    plt.figure(figsize=(10, 8)) # Adjusted figure size for better aesthetics

    # Set plot style for a cleaner look
    plt.style.use('seaborn-v0_8-whitegrid') # Using a modern seaborn style
    plt.rcParams['font.family'] = 'Times New Roman'

    # Set point color based on uncertainty
    sc = plt.scatter(data[target_colum], data['predicted'],
                     c=data['uncertainty'], cmap='viridis', # 'viridis' is a good perceptually uniform colormap
                     alpha=0.7, s=35, edgecolors='none') # Increased point size, removed edgecolors

    # Add linear fit line - filter NaN values
    # Create mask for valid data
    valid_mask = ~(np.isnan(data[target_colum]) | np.isnan(data['predicted']))
    valid_x = data.loc[valid_mask, target_colum]
    valid_y = data.loc[valid_mask, 'predicted']

    # Perform linear regression using valid data
    slope, intercept, r_value, p_value, std_err = stats.linregress(valid_x, valid_y)

    # Calculate R²
    ss_res = np.sum((valid_x - valid_y) ** 2)
    ss_tot = np.sum((valid_x - np.mean(valid_x)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Calculate RMSE and MAE
    rmse = np.sqrt(np.mean((valid_y - valid_x) ** 2))

    # Plot regression line
    min_val = min(data[target_colum].min(), data['predicted'].min())
    max_val = max(data[target_colum].max(), data['predicted'].max())
    x_vals = np.array([min_val, max_val])
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, 'b-', linewidth=2.5, label='Regression Line') # Thicker line, added label

    # Plot 1:1 line
    plt.plot([min_val, max_val],
             [min_val, max_val], 'r--', linewidth=1.5, label='1:1 Line') # Adjusted line style and width

    # Add evaluation metrics text box to the plot
    textstr = '\n'.join((
        r'$R² = %.3f$' % (r_squared,),
        r'$RMSE = %.3f$' % (rmse,)
    ))

    # Add text box with improved styling
    props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.85, edgecolor='gray', linewidth=0.8)
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12, # Smaller font for metrics
                   verticalalignment='top', bbox=props, fontweight='bold')

    # Set color bar
    cbar = plt.colorbar(sc, label='Prediction Uncertainty', pad=0.02) # Changed label to English, adjusted padding
    cbar.ax.tick_params(labelsize=10) # Smaller tick labels for colorbar
    cbar.set_label('Prediction Uncertainty', fontsize=12, fontweight='bold') # Changed label to English, adjusted font size

    # Set axis labels and title
    plt.xlabel('Observed Values', fontsize=14, fontweight='bold') # Changed label to English
    plt.ylabel('Predicted Values', fontsize=14, fontweight='bold') # Changed label to English
    plt.title('Predicted vs. Observed Values (Color by Uncertainty)', fontsize=16, fontweight='bold', pad=15) # Changed title to English

    # Set grid lines
    plt.grid(True, linestyle=':', alpha=0.7, color='gray') # Lighter grid lines

    # Set tick font size
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Add legend for regression and 1:1 lines
    plt.legend(fontsize=12, loc='lower right')

    # Set plot boundaries
    plt.tight_layout()

    # Save image
    output_dir = r"../results/figures"
    # Save as PDF format (vector graphic)
    plt.savefig(os.path.join(output_dir, "prediction_scatter.pdf"), bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    data_path = r"../results/basic_analysis_results.csv"
    # for SOC
    visualize_correlation_scatter(data_path, 'OC')

    #for Crime rate
    #visualize_correlation_scatter(data_path, 'CrimeRate')