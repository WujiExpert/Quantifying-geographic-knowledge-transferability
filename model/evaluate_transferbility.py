import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import ast

from utils import calculate_metrics, evaluate_multiclass_prediction, calculate_random_f1_baseline

plt.rcParams['font.family'] = 'Times New Roman'
# Fix negative sign display issue
plt.rcParams['axes.unicode_minus'] = False

def evaluate_model_performance_r2(y_true, y_pred, n_bootstrap=1000, alpha=0.05):
    """Evaluate model performance, including R² Bootstrap confidence intervals and Cohen's f² effect size analysis

    Parameters:
        y_true: numpy.ndarray, true values
        y_pred: numpy.ndarray, predicted values
        n_bootstrap: int, number of Bootstrap resampling iterations, default 1000
        alpha: float, significance level, default 0.05 (95% CI)

    Returns:
        dict: dictionary containing evaluation results
    """

    # Calculate original metrics
    metrics = calculate_metrics(y_true, y_pred)
    original_r2 = metrics['r2']

    # Calculate Cohen's f²
    f2 = original_r2 / (1 - original_r2)

    # Bootstrap analysis
    r2_boots = []
    f2_boots = []
    n_samples = len(y_true)

    for _ in tqdm(range(n_bootstrap), desc="Executing Bootstrap analysis"):
        # Sampling with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_true = y_true[indices]
        boot_pred = y_pred[indices]

        # Calculate R² for this sample
        boot_metrics = calculate_metrics(boot_true, boot_pred)
        boot_r2 = boot_metrics['r2']
        r2_boots.append(boot_r2)

        # Calculate Cohen's f² for this sample
        boot_f2 = boot_r2 / (1 - boot_r2)
        f2_boots.append(boot_f2)

    # Calculate confidence intervals
    r2_ci = np.percentile(r2_boots, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    f2_ci = np.percentile(f2_boots, [alpha / 2 * 100, (1 - alpha / 2) * 100])

    # Visualize Bootstrap distribution
    plt.figure(figsize=(5, 8))
    axis_color = '#CCCCCC'
    line_width = 1  # Originally 2, reduced line width

    # R² Distribution
    ax1 = plt.subplot(2, 1, 1)
    plt.hist(r2_boots, bins=50, alpha=0.7, color='skyblue')
    plt.axvline(original_r2, color='red', linestyle='--', linewidth=line_width, label=f'Original R² = {original_r2:.3f}')
    plt.axvline(r2_ci[0], color='green', linestyle=':', linewidth=line_width, label=f'95% CI: [{r2_ci[0]:.3f}, {r2_ci[1]:.3f}]')
    plt.axvline(r2_ci[1], color='green', linestyle=':', linewidth=line_width)
    plt.xlabel('R²', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('R² Bootstrap Distribution', fontsize=14, fontweight='bold', pad=10)
    plt.legend(fontsize=10)
    ax1.spines['top'].set_color(axis_color)
    ax1.spines['right'].set_color(axis_color)
    ax1.spines['bottom'].set_color(axis_color)
    ax1.spines['left'].set_color(axis_color)

    # Cohen's f² Distribution
    ax2 = plt.subplot(2, 1, 2)
    plt.hist(f2_boots, bins=50, alpha=0.7, color='lightgreen')
    plt.axvline(f2, color='red', linestyle='--', linewidth=line_width, label=f'Original f² = {f2:.3f}')
    plt.axvline(f2_ci[0], color='green', linestyle=':', linewidth=line_width, label=f'95% CI: [{f2_ci[0]:.3f}, {f2_ci[1]:.3f}]')
    plt.axvline(f2_ci[1], color='green', linestyle=':', linewidth=line_width)
    plt.xlabel("Cohen's f²", fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title("Cohen's f² Bootstrap Distribution", fontsize=14, fontweight='bold', pad=10)
    plt.legend(fontsize=10)
    ax2.spines['top'].set_color(axis_color)
    ax2.spines['right'].set_color(axis_color)
    ax2.spines['bottom'].set_color(axis_color)
    ax2.spines['left'].set_color(axis_color)

    plt.tight_layout()
    plt.savefig('../results/figures/bootstrap_analysis_r2.pdf', bbox_inches='tight')
    plt.close()
    print("Save figures to /results/figures/bootstrap_analysis_r2.pdf")

    # Determine transferability based on Cohen's f²
    if f2 < 0.02:
        transferability = "Effect size is negligible. Geographic knowledge transferability is very weak."
    elif 0.02 <= f2 < 0.15:
        transferability = "Small effect size. Geographic knowledge transferability is weak."
    elif 0.15 <= f2 < 0.35:
        transferability = "Medium effect size. Geographic knowledge transferability is moderate."
    else:
        transferability = "Large effect size. Geographic knowledge transferability is strong."

    return transferability


def evaluate_model_performance_f1(y_true, y_pred, n_bootstrap=1000, alpha=0.05, k_value=5):
    """Evaluate multiclass model performance, including F1-score Bootstrap confidence intervals and SF1G effect size analysis

    Parameters:
        y_true: list, true category list
        y_pred: list, predicted category list
        n_bootstrap: int, number of Bootstrap resampling iterations
        alpha: float, significance level
        k_value: int, K value for TOP-K evaluation

    Returns:
        str: knowledge transferability evaluation result
    """

    y_true = [ast.literal_eval(s) for s in y_true]
    y_pred = [ast.literal_eval(s) for s in y_pred]
    # Calculate original F1-score
    original_f1 = evaluate_multiclass_prediction(y_true, y_pred, k_value=k_value)[f'f1@{k_value}']
    f1_random = calculate_random_f1_baseline(y_true, k=k_value)['random_f1']
    original_sf1g = (original_f1 - f1_random) / (1 - f1_random)

    # Bootstrap analysis
    f1_boots = []
    sf1g_boots = []
    n_samples = len(y_true)

    for _ in tqdm(range(n_bootstrap), desc="Executing Bootstrap analysis (F1)"):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_true = [y_true[i] for i in indices]
        boot_pred = [y_pred[i] for i in indices]

        boot_f1 = evaluate_multiclass_prediction(boot_true, boot_pred, k_value=k_value)[f'f1@{k_value}']
        boot_sf1g = (boot_f1 - f1_random) / (1 - f1_random)
        f1_boots.append(boot_f1)
        sf1g_boots.append(boot_sf1g)

    # Calculate confidence intervals
    f1_ci = np.percentile(f1_boots, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    sf1g_ci = np.percentile(sf1g_boots, [alpha / 2 * 100, (1 - alpha / 2) * 100])

    # Visualization
    plt.figure(figsize=(10, 5))
    axis_color = '#CCCCCC'
    line_width = 1

    # F1-score Distribution
    ax1 = plt.subplot(1, 2, 1)
    plt.hist(f1_boots, bins=50, alpha=0.7, color='skyblue')
    plt.axvline(original_f1, color='red', linestyle='--', linewidth=line_width, label=f'Original F1 = {original_f1:.3f}')
    plt.axvline(f1_ci[0], color='green', linestyle=':', linewidth=line_width, label=f'95% CI: [{f1_ci[0]:.3f}, {f1_ci[1]:.3f}]')
    plt.axvline(f1_ci[1], color='green', linestyle=':', linewidth=line_width)
    plt.xlabel('F1-score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('F1-score Bootstrap Distribution', fontsize=14, fontweight='bold', pad=10)
    plt.legend(fontsize=10)
    ax1.spines['top'].set_color(axis_color)
    ax1.spines['right'].set_color(axis_color)
    ax1.spines['bottom'].set_color(axis_color)
    ax1.spines['left'].set_color(axis_color)

    # SF1G Distribution
    ax2 = plt.subplot(1, 2, 2)
    plt.hist(sf1g_boots, bins=50, alpha=0.7, color='lightgreen')
    plt.axvline(original_sf1g, color='red', linestyle='--', linewidth=line_width, label=f'Original SF1G = {original_sf1g:.3f}')
    plt.axvline(sf1g_ci[0], color='green', linestyle=':', linewidth=line_width, label=f'95% CI: [{sf1g_ci[0]:.3f}, {sf1g_ci[1]:.3f}]')
    plt.axvline(sf1g_ci[1], color='green', linestyle=':', linewidth=line_width)
    plt.xlabel('SF1G', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('SF1G Bootstrap Distribution', fontsize=14, fontweight='bold', pad=10)
    plt.legend(fontsize=10)
    ax2.spines['top'].set_color(axis_color)
    ax2.spines['right'].set_color(axis_color)
    ax2.spines['bottom'].set_color(axis_color)
    ax2.spines['left'].set_color(axis_color)

    plt.tight_layout()
    plt.savefig('../results/figures/bootstrap_analysis_f1.pdf', bbox_inches='tight')
    plt.close()
    print("Save figures to /results/figures/bootstrap_analysis_f1.pdf")

    # Determine transferability based on SF1G (thresholds can be adjusted as needed)
    if original_sf1g < 0.1:
        transferability = "Negligible effect size. Geographic knowledge transferability is very weak."
    elif 0.1 <= original_sf1g < 0.3:
        transferability = "Small effect size. Geographic knowledge transferability is weak."
    elif 0.3 <= original_sf1g < 0.5:
        transferability = "Medium effect size. Geographic knowledge transferability is moderate."
    else:
        transferability = "Large effect size. Geographic knowledge transferability is strong."

    return transferability


if __name__ == "__main__":
    # using example of soil organic carbon prediction
    # data_path = r"../results/basic_analysis_results.csv"
    # data = pd.read_csv(data_path)
    # result = evaluate_model_performance_r2(data['OC'], data['predicted'])

    # using example of soil organic carbon prediction
    # data_path = r"../results/basic_analysis_results.csv"
    # data = pd.read_csv(data_path)
    # result = evaluate_model_performance_r2(data['CrimeRate'], data['predicted'])

    # using example of sdp recommendation
    data_path = r"../results/basic_analysis_results.csv"
    data = pd.read_csv(data_path)
    result = evaluate_model_performance_f1(data['patterns'].tolist(), data['predicted'].tolist())

    print(result)