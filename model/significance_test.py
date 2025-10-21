import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

from utils import evaluate_multiclass_prediction, calculate_metrics

plt.rcParams['font.family'] = 'Times New Roman'
# Fix negative sign display issue
plt.rcParams['axes.unicode_minus'] = False

def bootstrap_f1_test(within_predictions, within_actuals, cross_predictions, cross_actuals, 
                      n_bootstrap=1000, k=5, alpha=0.05):
    """
    Use Bootstrap method to test statistical significance of F1 score differences between two groups
    
    Parameters:
    within_predictions (list): List of within-group prediction results, each element is a list of predicted labels
    within_actuals (list): List of within-group actual labels, each element is a list of actual labels
    cross_predictions (list): List of cross-group prediction results, each element is a list of predicted labels
    cross_actuals (list): List of cross-group actual labels, each element is a list of actual labels
    n_bootstrap (int): Number of Bootstrap resampling iterations
    k (int): TOP-K value
    alpha (float): Significance level
    
    Returns:
    dict: Dictionary containing test results
    """
    print(f"Executing Bootstrap F1 score difference significance test (n={n_bootstrap})...")
    
    # Calculate original F1 scores - using evaluate_multiclass_prediction function
    within_results = evaluate_multiclass_prediction(within_actuals, within_predictions, k)
    cross_results = evaluate_multiclass_prediction(cross_actuals, cross_predictions, k)
    
    within_f1 = within_results[f'f1@{k}']
    cross_f1 = cross_results[f'f1@{k}']
    observed_diff = within_f1 - cross_f1
    
    # Create index arrays
    within_indices = np.arange(len(within_predictions))
    cross_indices = np.arange(len(cross_predictions))
    
    # Store F1 score differences of Bootstrap samples
    bootstrap_diffs = []
    
    # Execute Bootstrap resampling
    for _ in tqdm(range(n_bootstrap), desc="Bootstrap progress"):
        # Random sampling with replacement
        within_bootstrap_indices = np.random.choice(within_indices, size=len(within_indices), replace=True)
        cross_bootstrap_indices = np.random.choice(cross_indices, size=len(cross_indices), replace=True)
        
        # Get Bootstrap samples
        within_bootstrap_preds = [within_predictions[i] for i in within_bootstrap_indices]
        within_bootstrap_acts = [within_actuals[i] for i in within_bootstrap_indices]
        cross_bootstrap_preds = [cross_predictions[i] for i in cross_bootstrap_indices]
        cross_bootstrap_acts = [cross_actuals[i] for i in cross_bootstrap_indices]
        
        # Calculate F1 scores of Bootstrap samples - using evaluate_multiclass_prediction function
        within_bootstrap_results = evaluate_multiclass_prediction(within_bootstrap_acts, within_bootstrap_preds, k)
        cross_bootstrap_results = evaluate_multiclass_prediction(cross_bootstrap_acts, cross_bootstrap_preds, k)
        
        within_bootstrap_f1 = within_bootstrap_results[f'f1@{k}']
        cross_bootstrap_f1 = cross_bootstrap_results[f'f1@{k}']
        
        # Calculate and store differences
        bootstrap_diffs.append(within_bootstrap_f1 - cross_bootstrap_f1)
    
    # Calculate confidence intervals
    lower_bound = np.percentile(bootstrap_diffs, alpha/2 * 100)
    upper_bound = np.percentile(bootstrap_diffs, (1 - alpha/2) * 100)
    
    # Calculate p-value (two-tailed test)
    # If 0 is not in the distribution range of bootstrap_diffs, then p-value is 0
    min_p = 1 / n_bootstrap
    if min(bootstrap_diffs) > 0 or max(bootstrap_diffs) < 0:
        p_value = min_p
    else:
        # Calculate how many bootstrap samples have difference values with opposite signs to the observed difference
        opposite_sign_count = sum(1 for diff in bootstrap_diffs if np.sign(diff) != np.sign(observed_diff))
        p_value = max(opposite_sign_count / n_bootstrap, min_p)

    # Determine if significant
    is_significant = p_value < alpha

    #visualization
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(bootstrap_diffs, bins=30, kde=False, color='#A6CEE3', edgecolor='white', alpha=0.85, ax=ax)
    #sns.kdeplot(bootstrap_diffs, color='#1F78B4', linewidth=2, ax=ax)

    ax.axvline(x=observed_diff, color='#E31A1C', linestyle='--', linewidth=1.5, label=f'Observed: {observed_diff:.3f}')
    ax.axvline(x=lower_bound, color='#33A02C', linestyle=':', linewidth=1, label=f'CI Lower: {lower_bound:.3f}')
    ax.axvline(x=upper_bound, color='#33A02C', linestyle=':', linewidth=1, label=f'CI Upper: {upper_bound:.3f}')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.2, label='Zero')
    ax.set_xlabel('F1 Score Difference (Intragroup - Intergroup)', fontsize=13, fontname='Times New Roman')
    ax.set_ylabel('Count', fontsize=13, fontname='Times New Roman')
    #ax.set_title('Bootstrap Distribution of F1 Score Difference by GDP Group', fontsize=15, weight='bold', pad=12, fontname='Times New Roman')
    ax.legend(fontsize=12, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=4, prop={'family': 'Times New Roman'})
    for label in ax.get_xticklabels():
        label.set_fontname('Times New Roman')
    for label in ax.get_yticklabels():
        label.set_fontname('Times New Roman')
    sns.despine()
    plt.tight_layout()
    plt.savefig('../results/figures/bootstrap_f1_diff.pdf', bbox_inches='tight')
    plt.close()
    
    return {
        'within_f1': within_f1,
        'cross_f1': cross_f1,
        'observed_diff': observed_diff,
        'confidence_interval': (lower_bound, upper_bound),
        'p_value': p_value,
        'is_significant': is_significant,
        'n_bootstrap': n_bootstrap,
        'alpha': alpha,
    }

def permutation_f1_test(within_predictions, within_actuals, cross_predictions, cross_actuals, 
                        n_permutations=1000, k=5):
    """
    Use permutation test method to test statistical significance of F1 score differences between two groups
    
    Parameters:
    within_predictions (list): List of within-group prediction results, each element is a list of predicted labels
    within_actuals (list): List of within-group actual labels, each element is a list of actual labels
    cross_predictions (list): List of cross-group prediction results, each element is a list of predicted labels
    cross_actuals (list): List of cross-group actual labels, each element is a list of actual labels
    n_permutations (int): Number of permutations
    k (int): TOP-K value
    
    Returns:
    dict: Dictionary containing test results
    """
    print(f"Executing permutation test F1 score difference significance test (n={n_permutations})...")
    
    # Calculate original F1 scores - using evaluate_multiclass_prediction function
    within_results = evaluate_multiclass_prediction(within_actuals, within_predictions, k)
    cross_results = evaluate_multiclass_prediction(cross_actuals, cross_predictions, k)
    
    within_f1 = within_results[f'f1@{k}']
    cross_f1 = cross_results[f'f1@{k}']
    observed_diff = within_f1 - cross_f1
    
    # Merge data
    all_predictions = within_predictions + cross_predictions
    all_actuals = within_actuals + cross_actuals
    n_within = len(within_predictions)
    n_cross = len(cross_predictions)
    n_total = n_within + n_cross
    
    # Store F1 score differences of permutation test
    permutation_diffs = []
    
    # Execute permutation test
    for _ in tqdm(range(n_permutations), desc="Permutation test progress"):
        # Randomly shuffle indices
        permuted_indices = np.random.permutation(n_total)
        
        # Allocate shuffled data
        permuted_within_indices = permuted_indices[:n_within]
        permuted_cross_indices = permuted_indices[n_within:]
        
        # Get permuted samples
        permuted_within_preds = [all_predictions[i] for i in permuted_within_indices]
        permuted_within_acts = [all_actuals[i] for i in permuted_within_indices]
        permuted_cross_preds = [all_predictions[i] for i in permuted_cross_indices]
        permuted_cross_acts = [all_actuals[i] for i in permuted_cross_indices]
        
        # Calculate F1 scores of permuted samples - using evaluate_multiclass_prediction function
        permuted_within_results = evaluate_multiclass_prediction(permuted_within_acts, permuted_within_preds, k)
        permuted_cross_results = evaluate_multiclass_prediction(permuted_cross_acts, permuted_cross_preds, k)
        
        permuted_within_f1 = permuted_within_results[f'f1@{k}']
        permuted_cross_f1 = permuted_cross_results[f'f1@{k}']
        
        # Calculate and store differences
        permutation_diffs.append(permuted_within_f1 - permuted_cross_f1)
    
    # Calculate p-value (two-tailed test)
    min_p = 1 / n_permutations
    p_value = max(sum(1 for diff in permutation_diffs if abs(diff) >= abs(observed_diff)) / n_permutations, min_p)
    is_significant = p_value < 0.05
    
    #visualization
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(3.46, 2.23))
    sns.histplot(permutation_diffs, bins=30, kde=False, color='#A6CEE3', edgecolor='white', alpha=0.85, ax=ax)
    sns.kdeplot(permutation_diffs, color='#1F78B4', linewidth=2, ax=ax)
    mean_diff = np.mean(permutation_diffs)
    ax.axvline(x=observed_diff, color='#E31A1C', linestyle='--', linewidth=2.5, label=f'Observed: {observed_diff:.3f}')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, label='Zero')
    ax.axvline(x=mean_diff, color='#FF7F00', linestyle='-', linewidth=2, label=f'Mean: {mean_diff:.3f}')
    ax.set_xlabel('F1 Score Difference (Within - Cross)', fontsize=13, fontname='Times New Roman')
    ax.set_ylabel('Count', fontsize=13, fontname='Times New Roman')
    ax.set_title('Permutation Test Distribution of F1 Score Difference', fontsize=15, weight='bold', pad=12, fontname='Times New Roman')
    ax.legend(fontsize=11, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3, prop={'family': 'Times New Roman'})
    for label in ax.get_xticklabels():
        label.set_fontname('Times New Roman')
    for label in ax.get_yticklabels():
        label.set_fontname('Times New Roman')
    sns.despine()
    plt.tight_layout(rect=[0, 0, 1, 1])
    fig.text(0.98, 0.98, f'p-value = {p_value:.3g}\nSignificant: {"Yes" if is_significant else "No"}',
             ha='right', va='top', fontsize=12, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'), fontname='Times New Roman')
    output_path = '../results/permutation_f1_diff.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'within_f1': within_f1,
        'cross_f1': cross_f1,
        'observed_diff': observed_diff,
        'p_value': p_value,
        'is_significant': is_significant,
        'n_permutations': n_permutations,
        'plot_path': output_path
    }


def bootstrap_r2_test(within_predictions, within_actuals, cross_predictions, cross_actuals,
                      n_bootstrap=1000, alpha=0.05):
    """
    Use Bootstrap method to test statistical significance of R² score differences between two groups

    Parameters:
    within_predictions (list/array): Within-group prediction results
    within_actuals (list/array): Within-group actual values
    cross_predictions (list/array): Cross-group prediction results
    cross_actuals (list/array): Cross-group actual values
    n_bootstrap (int): Number of Bootstrap resampling iterations
    alpha (float): Significance level

    Returns:
    dict: Dictionary containing test results
    """
    print(f"Executing Bootstrap R² score difference significance test (n={n_bootstrap})...")

    # Calculate original R² scores
    within_r2 = calculate_metrics(within_actuals, within_predictions)['r2']
    cross_r2 = calculate_metrics(cross_actuals, cross_predictions)['r2']
    observed_diff = within_r2 - cross_r2

    # Create index arrays
    within_indices = np.arange(len(within_predictions))
    cross_indices = np.arange(len(cross_predictions))

    # Store R² score differences of Bootstrap samples
    bootstrap_diffs = []

    # Execute Bootstrap resampling
    for _ in tqdm(range(n_bootstrap), desc="Bootstrap progress"):
        # Random sampling with replacement
        within_bootstrap_indices = np.random.choice(within_indices, size=len(within_indices), replace=True)
        cross_bootstrap_indices = np.random.choice(cross_indices, size=len(cross_indices), replace=True)

        # Get Bootstrap samples
        within_bootstrap_preds = np.array(within_predictions)[within_bootstrap_indices]
        within_bootstrap_acts = np.array(within_actuals)[within_bootstrap_indices]
        cross_bootstrap_preds = np.array(cross_predictions)[cross_bootstrap_indices]
        cross_bootstrap_acts = np.array(cross_actuals)[cross_bootstrap_indices]

        # Calculate R² scores of Bootstrap samples
        try:
            within_bootstrap_r2 = calculate_metrics(within_bootstrap_acts, within_bootstrap_preds)['r2']
        except:
            within_bootstrap_r2 = 0  # Handle possible error cases

        try:
            cross_bootstrap_r2 = calculate_metrics(cross_bootstrap_acts, cross_bootstrap_preds)['r2']
        except:
            cross_bootstrap_r2 = 0  # Handle possible error cases

        # Calculate and store differences
        bootstrap_diffs.append(within_bootstrap_r2 - cross_bootstrap_r2)

    # Calculate confidence intervals
    lower_bound = np.percentile(bootstrap_diffs, alpha / 2 * 100)
    upper_bound = np.percentile(bootstrap_diffs, (1 - alpha / 2) * 100)

    # Calculate p-value (two-tailed test)
    # If 0 is not in the distribution range of bootstrap_diffs, then p-value is 0
    min_p = 1 / n_bootstrap
    if min(bootstrap_diffs) > 0 or max(bootstrap_diffs) < 0:
        p_value = min_p
    else:
        # Calculate how many bootstrap samples have difference values with opposite signs to the observed difference
        opposite_sign_count = sum(1 for diff in bootstrap_diffs if np.sign(diff) != np.sign(observed_diff))
        p_value = max(opposite_sign_count / n_bootstrap, min_p)

    # Determine if significant
    is_significant = p_value < alpha

    #visualization
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(bootstrap_diffs, bins=30, kde=False, color='#A6CEE3', edgecolor='white', alpha=0.85, ax=ax)
    #sns.kdeplot(bootstrap_diffs, color='#1F78B4', linewidth=2, ax=ax)
    #mean_diff = np.mean(bootstrap_diffs)
    ax.axvline(x=observed_diff, color='#E31A1C', linestyle='--', linewidth=1.5, label=f'Observed: {observed_diff:.3f}')
    ax.axvline(x=lower_bound, color='#33A02C', linestyle=':', linewidth=1, label=f'CI Lower: {lower_bound:.3f}')
    ax.axvline(x=upper_bound, color='#33A02C', linestyle=':', linewidth=1, label=f'CI Upper: {upper_bound:.3f}')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.2, label='Zero')
    #ax.axvline(x=mean_diff, color='#FF7F00', linestyle='-', linewidth=2, label=f'Mean: {mean_diff:.3f}')
    ax.set_xlabel('R² Score Difference (Intragroup - Intergroup)', fontsize=13, fontname='Times New Roman')
    ax.set_ylabel('Count', fontsize=13, fontname='Times New Roman')
    #ax.set_title('Bootstrap Distribution of R² Score Difference', fontsize=15, weight='bold', pad=12, fontname='Times New Roman')
    ax.legend(fontsize=12, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=4, prop={'family': 'Times New Roman'})
    for label in ax.get_xticklabels():
        label.set_fontname('Times New Roman')
    for label in ax.get_yticklabels():
        label.set_fontname('Times New Roman')
    sns.despine()
    plt.tight_layout()
    #fig.text(0.98, 0.98, f'p-value = {p_value:.3g}\nSignificant: {"Yes" if is_significant else "No"}',
     #        ha='right', va='top', fontsize=12, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'), fontname='Times New Roman')
    plt.savefig('../results/figures/bootstrap_r2_diff.pdf', bbox_inches='tight')
    plt.close()

    return {
        'within_r2': within_r2,
        'cross_r2': cross_r2,
        'observed_diff': observed_diff,
        'confidence_interval': (lower_bound, upper_bound),
        'p_value': p_value,
        'is_significant': is_significant,
        'n_bootstrap': n_bootstrap,
        'alpha': alpha,
    }


def permutation_r2_test(within_predictions, within_actuals, cross_predictions, cross_actuals,
                        n_permutations=1000):
    """
    Use permutation test method to test statistical significance of R² score differences between two groups

    Parameters:
    within_predictions (list/array): Within-group prediction results
    within_actuals (list/array): Within-group actual values
    cross_predictions (list/array): Cross-group prediction results
    cross_actuals (list/array): Cross-group actual values
    n_permutations (int): Number of permutations

    Returns:
    dict: Dictionary containing test results
    """
    print(f"Executing permutation test R² score difference significance test (n={n_permutations})...")

    # Calculate original R² scores
    within_r2 = calculate_metrics(within_actuals, within_predictions)['r2']
    cross_r2 = calculate_metrics(cross_actuals, cross_predictions)['r2']
    observed_diff = within_r2 - cross_r2

    # Merge data
    all_predictions = np.concatenate([within_predictions, cross_predictions])
    all_actuals = np.concatenate([within_actuals, cross_actuals])
    n_within = len(within_predictions)
    n_cross = len(cross_predictions)
    n_total = n_within + n_cross

    # Store R² score differences of permutation test
    permutation_diffs = []

    # Execute permutation test
    for _ in tqdm(range(n_permutations), desc="Permutation test progress"):
        # Randomly shuffle indices
        permuted_indices = np.random.permutation(n_total)

        # Allocate shuffled data
        permuted_within_indices = permuted_indices[:n_within]
        permuted_cross_indices = permuted_indices[n_within:]

        # Get permuted samples
        permuted_within_preds = all_predictions[permuted_within_indices]
        permuted_within_acts = all_actuals[permuted_within_indices]
        permuted_cross_preds = all_predictions[permuted_cross_indices]
        permuted_cross_acts = all_actuals[permuted_cross_indices]

        # Calculate R² scores of permuted samples
        try:
            permuted_within_r2 = calculate_metrics(permuted_within_acts, permuted_within_preds)['r2']
        except:
            permuted_within_r2 = 0  # Handle possible error cases

        try:
            permuted_cross_r2 = calculate_metrics(permuted_cross_acts, permuted_cross_preds)['r2']
        except:
            permuted_cross_r2 = 0  # Handle possible error cases

        # Calculate and store differences
        permutation_diffs.append(permuted_within_r2 - permuted_cross_r2)

    # Calculate p-value (two-tailed test)
    min_p = 1 / n_permutations
    p_value = max(sum(1 for diff in permutation_diffs if abs(diff) >= abs(observed_diff)) / n_permutations, min_p)
    is_significant = p_value < 0.05
    #visualization
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(3.46, 2.23))
    sns.histplot(permutation_diffs, bins=30, kde=False, color='#A6CEE3', edgecolor='white', alpha=0.85, ax=ax)
    sns.kdeplot(permutation_diffs, color='#1F78B4', linewidth=2, ax=ax)
    mean_diff = np.mean(permutation_diffs)
    ax.axvline(x=observed_diff, color='#E31A1C', linestyle='--', linewidth=2.5, label=f'Observed: {observed_diff:.3f}')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, label='Zero')
    ax.axvline(x=mean_diff, color='#FF7F00', linestyle='-', linewidth=2, label=f'Mean: {mean_diff:.3f}')
    ax.set_xlabel('R² Score Difference (Within - Cross)', fontsize=13, fontname='Times New Roman')
    ax.set_ylabel('Count', fontsize=13, fontname='Times New Roman')
    ax.set_title('Permutation Test Distribution of R² Score Difference', fontsize=15, weight='bold', pad=12, fontname='Times New Roman')
    ax.legend(fontsize=11, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3, prop={'family': 'Times New Roman'})
    for label in ax.get_xticklabels():
        label.set_fontname('Times New Roman')
    for label in ax.get_yticklabels():
        label.set_fontname('Times New Roman')
    sns.despine()
    plt.tight_layout(rect=[0, 0, 1, 1])
    fig.text(0.98, 0.98, f'p-value = {p_value:.3g}\nSignificant: {"Yes" if is_significant else "No"}',
             ha='right', va='top', fontsize=12, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'), fontname='Times New Roman')
    output_path = '../results/permutation_r2_diff.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'within_r2': within_r2,
        'cross_r2': cross_r2,
        'observed_diff': observed_diff,
        'p_value': p_value,
        'is_significant': is_significant,
        'n_permutations': n_permutations,
        'plot_path': output_path
    }


def run_significance_tests(results_data, key_factor_column, target_variable, k=5, regression=False):
    """
    Run statistical significance tests, supporting F1 score (classification) or R² value (regression) difference tests

    Parameters:
    results_data (DataFrame): DataFrame containing prediction results
    key_factor_column (str): Key factor column name
    target_variable: Target observation variable
    k (int): TOP-K value (only for classification tasks)
    regression (bool): Whether it's a regression task, True means test R² values, False means test F1 values

    Returns:
    dict: Dictionary containing test results
    """

    # Check if required columns exist
    required_columns = ['predicted_within', 'predicted_cross', target_variable]
    if not all(col in results_data.columns for col in required_columns):
        print(f"Error: Missing required columns in data. Required columns: {required_columns}")
        print(f"Available columns: {results_data.columns.tolist()}")
        return None

    if regression:
        # Regression task - test R² value differences
        # Extract within-group and cross-group prediction results
        valid_within_indices = results_data[results_data['predicted_within'].notna()].index
        within_predictions = results_data.loc[valid_within_indices, 'predicted_within'].to_numpy()
        within_actual_values = results_data.loc[valid_within_indices, target_variable].to_numpy()

        valid_cross_indices = results_data[results_data['predicted_cross'].notna()].index
        cross_predictions = results_data.loc[valid_cross_indices, 'predicted_cross'].to_numpy()
        cross_actual_values = results_data.loc[valid_cross_indices, target_variable].to_numpy()

        # Run Bootstrap test
        bootstrap_results = bootstrap_r2_test(
            within_predictions=within_predictions,
            within_actuals=within_actual_values,
            cross_predictions=cross_predictions,
            cross_actuals=cross_actual_values,
            n_bootstrap=1000
        )

        # Run permutation test
        permutation_results = permutation_r2_test(
            within_predictions=within_predictions,
            within_actuals=within_actual_values,
            cross_predictions=cross_predictions,
            cross_actuals=cross_actual_values,
            n_permutations=1000
        )

        # Print results
        print("\nStatistical significance test results:")
        print(f"Within-group prediction R²: {bootstrap_results['within_r2']:.4f}")
        print(f"Cross-group prediction R²: {bootstrap_results['cross_r2']:.4f}")
        print(f"Observed difference: {bootstrap_results['observed_diff']:.4f}")

        print("\nBootstrap test results:")
        print(
            f"  {(1 - bootstrap_results['alpha']) * 100}% confidence interval: ({bootstrap_results['confidence_interval'][0]:.4f}, {bootstrap_results['confidence_interval'][1]:.4f})")
        print(f"  p-value: {bootstrap_results['p_value']:.4f}")
        print(f"  Significant: {'Yes' if bootstrap_results['is_significant'] else 'No'}")

        print("\nPermutation test results:")
        print(f"  p-value: {permutation_results['p_value']:.4f}")
        print(f"  Significant: {'Yes' if permutation_results['is_significant'] else 'No'}")

        # Return comprehensive results
        return {
            'bootstrap': bootstrap_results,
            'permutation': permutation_results,
            'key_factor': key_factor_column,
            'metric': 'r2'
        }
    else:
        # Classification task - test F1 value differences
        # Extract within-group and cross-group prediction results
        valid_within_indices = results_data[results_data['predicted_within'].apply(lambda x: len(x) > 0)].index
        within_predictions = results_data.loc[valid_within_indices, 'predicted_within'].tolist()
        within_actual_classes = results_data.loc[valid_within_indices, target_variable].tolist()
        valid_cross_indices = results_data[results_data['predicted_cross'].apply(lambda x: len(x) > 0)].index
        cross_predictions = results_data.loc[valid_cross_indices, 'predicted_cross'].tolist()
        cross_actual_classes = results_data.loc[valid_cross_indices, target_variable].tolist()

        # Run Bootstrap test
        bootstrap_results = bootstrap_f1_test(
            within_predictions=within_predictions,
            within_actuals=within_actual_classes,
            cross_predictions=cross_predictions,
            cross_actuals=cross_actual_classes,
            n_bootstrap=1000,
            k=k
        )

        # Run permutation test
        permutation_results = permutation_f1_test(
            within_predictions=within_predictions,
            within_actuals=within_actual_classes,
            cross_predictions=cross_predictions,
            cross_actuals=cross_actual_classes,
            n_permutations=1000,
            k=k
        )

        # Print results
        print("\nStatistical significance test results:")
        print(f"Within-group prediction TOP-{k} F1 score: {bootstrap_results['within_f1']:.4f}")
        print(f"Cross-group prediction TOP-{k} F1 score: {bootstrap_results['cross_f1']:.4f}")
        print(f"Observed difference: {bootstrap_results['observed_diff']:.4f}")

        print("\nBootstrap test results:")
        print(
            f"  {(1 - bootstrap_results['alpha']) * 100}% confidence interval: ({bootstrap_results['confidence_interval'][0]:.4f}, {bootstrap_results['confidence_interval'][1]:.4f})")
        print(f"  p-value: {bootstrap_results['p_value']:.4f}")
        print(f"  Significant: {'Yes' if bootstrap_results['is_significant'] else 'No'}")

        print("\nPermutation test results:")
        print(f"  p-value: {permutation_results['p_value']:.4f}")
        print(f"  Significant: {'Yes' if permutation_results['is_significant'] else 'No'}")

        # Return comprehensive results
        return {
            'bootstrap': bootstrap_results,
            'permutation': permutation_results,
            'key_factor': key_factor_column,
            'k_value': k,
            'metric': 'f1'
        }

if __name__ == "__main__":
    # Example usage
    results_file = '../results/prediction_sdp_with_groups.xlsx'
    key_factor = 'climate_type_final_group'  # or other key factor column name
    
    # Run significance tests
    test_results = run_significance_tests(results_file, key_factor, k=5)
    
    # Save results as JSON file
    if test_results:
        import json
        output_path = '../results/significance_test_results.json'
        
        # Convert non-serializable objects to strings
        serializable_results = {}
        for test_type, results in test_results.items():
            if isinstance(results, dict):
                serializable_results[test_type] = {k: str(v) if not isinstance(v, (int, float, str, bool, list, dict, type(None))) else v 
                                                for k, v in results.items()}
            else:
                serializable_results[test_type] = str(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=4)
        
        print(f"\nResults saved to: {output_path}")