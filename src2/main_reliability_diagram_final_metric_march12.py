import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import pickle

###########################
# 1. Utility Functions for ECE and Plotting
###########################

def compute_ece(confidences, accuracies, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        confidences: 1D array-like of confidence scores in [0,1]
        accuracies:  1D array-like of 0/1 correctness
        n_bins:      Number of bins for ECE
    Returns:
        Float representing the ECE
    """
    confidences = np.array(confidences)
    accuracies = np.array(accuracies)
    bins = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences >= bins[i]) & (confidences < bins[i + 1])
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / len(confidences)) * abs(bin_acc - bin_conf)
    return ece


def plot_multiple_calibration_curve(
    df, confidence_cols, correctness_col,
    nbins=10, plot_title='temp', save_path=None
):
    """
    Plots multiple reliability diagrams (confidence vs. accuracy) on the same plot,
    calculates ECE for each, and optionally saves the figure to save_path.
    
    Args:
        df: DataFrame containing the data
        confidence_cols: List of column names for confidence scores
        correctness_col: Column name for accuracy (0/1)
        nbins: Number of bins for the reliability diagram
        plot_title: Title of the plot
        save_path: Path to save the plot (if None, displays the plot)
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for confidence_col in confidence_cols:
        confidences = df[confidence_col].values
        correctness = df[correctness_col].values

        # Binning
        bins = np.linspace(0, 1, nbins + 1)
        bin_indices = np.digitize(confidences, bins) - 1

        bin_conf_sums = np.zeros(nbins, dtype=float)
        bin_acc_sums = np.zeros(nbins, dtype=float)
        bin_counts = np.zeros(nbins, dtype=int)

        for i in range(len(confidences)):
            b = bin_indices[i]
            if 0 <= b < nbins:
                bin_conf_sums[b] += confidences[i]
                bin_acc_sums[b] += correctness[i]
                bin_counts[b] += 1

        # Compute bin-wise means and ECE
        bin_conf_means = []
        bin_acc_means = []
        for b in range(nbins):
            if bin_counts[b] > 0:
                mean_conf = bin_conf_sums[b] / bin_counts[b]
                mean_acc = bin_acc_sums[b] / bin_counts[b]
                bin_conf_means.append(mean_conf)
                bin_acc_means.append(mean_acc)

        total = float(len(confidences))
        ece = 0.0
        for b in range(nbins):
            if bin_counts[b] > 0:
                mean_conf_b = bin_conf_sums[b] / bin_counts[b]
                mean_acc_b = bin_acc_sums[b] / bin_counts[b]
                fraction_b = bin_counts[b] / total
                ece += fraction_b * abs(mean_acc_b - mean_conf_b)

        # Clean up the label for display
        display_label = confidence_col

        # If the entire column name is exponent_gd_sam_mean_score_processed_num_clusters,
        # then collapse it to "grounding sam"
        if 'gd_sam' in display_label:
            display_label = 'grounding sam'
        # If the entire column name is exponent_grounding_llama32_11b_processed_score_num_clusters,
        # then collapse it to "llama3.2v 11 b"
        elif 'llama32_11b' in display_label:
            display_label = 'llama3.2v 11 b'
        elif 'llama32_70b' in display_label:
            display_label = 'llama3.2v 70b'
        elif 'biomedclip' in display_label:
            display_label = 'BiomedCLIP'
        else:
            # For other columns, do lighter replacements (if needed)
            for r in [
                'exponent gd sammean_score_processed_',
                'exponent_grounding_llama32_11b_processed_score_',
                'final_conf_', '_min_max_scaled', 'confidence_from_', 'calibrated_'
            ]:
                display_label = display_label.replace(r, '')
            display_label = display_label.replace('_', ' ')

        # Plot
        ax.plot(
            bin_conf_means, bin_acc_means,
            marker='o', linestyle='-', linewidth=2,
            label=f'{display_label} (ECE={ece:.3f})'
        )

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('Confidence', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title(f'Reliability Diagram - {plot_title}', fontsize=16)
    ax.legend(loc='best')

    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


###########################
# 2. Calibration Methods
###########################

def exponent_calibration(conf, gr_score, T):
    """
    Exponent-based calibration: new_conf = conf * (gr_score ** (1 / T))
    
    Args:
        conf: Baseline confidence scores
        gr_score: Grounding scores
        T: Temperature parameter
    Returns:
        Calibrated confidence scores
    """
    return conf * (gr_score ** (1.0 / T))


def logistic_temperature_scaling(conf, T):
    """
    Logistic temperature scaling: conf_scaled = sigmoid(logit(conf) / T)
    
    Args:
        conf: Confidence scores
        T: Temperature parameter
    Returns:
        Calibrated confidence scores
    """
    eps = 1e-10
    conf = np.clip(conf, eps, 1 - eps)
    logit_conf = np.log(conf / (1 - conf))
    scaled_logit = logit_conf / T
    conf_scaled = 1.0 / (1.0 + np.exp(-scaled_logit))
    return conf_scaled


def find_best_exponent_T(calib_conf, calib_gr, calib_acc, T_candidates):
    """
    Find the best T for exponent-based calibration by minimizing ECE.
    
    Args:
        calib_conf: Calibration set confidence scores
        calib_gr: Calibration set grounding scores
        calib_acc: Calibration set accuracies
        T_candidates: Array of T values to search over
    Returns:
        Tuple of (best T, lowest ECE)
    """
    best_T = None
    lowest_ece = float('inf')
    for T in T_candidates:
        temp_conf = exponent_calibration(calib_conf, calib_gr, T)
        ece_val = compute_ece(temp_conf, calib_acc)
        if ece_val < lowest_ece:
            lowest_ece = ece_val
            best_T = T
    return best_T, lowest_ece


def isotonic_fit_transform(calib_conf, calib_acc, eval_conf):
    """
    Fit isotonic regression on calibration data and transform evaluation data.
    
    Args:
        calib_conf: Calibration set confidence scores
        calib_acc: Calibration set accuracies
        eval_conf: Evaluation set confidence scores
    Returns:
        Calibrated confidence scores for evaluation set
    """
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(calib_conf, calib_acc)
    return iso_reg.predict(eval_conf)


def logistic_regression_temperature_scaling(calib_conf, calib_acc, eval_conf):
    """
    Use logistic regression to learn the temperature parameter T.
    
    Args:
        calib_conf: Calibration set confidence scores
        calib_acc: Calibration set accuracies (0 or 1)
        eval_conf: Evaluation set confidence scores
    Returns:
        Tuple of (calibrated confidence scores for evaluation set, learned T)
    """
    eps = 1e-10
    calib_conf = np.clip(calib_conf, eps, 1 - eps)
    eval_conf = np.clip(eval_conf, eps, 1 - eps)
    
    # Convert continuous 0./1. to integer class labels
    y_calib = np.rint(calib_acc).astype(int)
    
    # Compute logits
    logit_calib_conf = np.log(calib_conf / (1 - calib_conf)).reshape(-1, 1)
    logit_eval_conf = np.log(eval_conf / (1 - eval_conf)).reshape(-1, 1)
    
    # Fit logistic regression (coefficient = 1/T)
    model = LogisticRegression(fit_intercept=False)
    model.fit(logit_calib_conf, y_calib)
    
    beta = model.coef_[0][0]
    T = 1 / beta if beta != 0 else 1.0
    
    # Apply temperature scaling
    scaled_logit_eval = logit_eval_conf / T
    conf_scaled = 1.0 / (1.0 + np.exp(-scaled_logit_eval))
    return conf_scaled.flatten(), T


###########################
# 3. Main Script with 5 Random Shuffles for CV and Aggregated CSV Results
###########################

if __name__ == '__main__':
    
    # base_dataset = 'vqa'
    base_dataset = 'slake'
    
    # **Load Data**
    # data_path = '/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/my_outputs/vqa/merged_with_grounding_processed_with_variance_march7.csv'
    data_path = '/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/my_outputs/merged_with_grounding_processed.csv'
    df_vqa = pd.read_csv(data_path)

    # **Define Columns**
    gr_cols = [
        'grounding_biomedclip_processed_scaled_average',
        'grounding_llama32_11b_processed_score',
        'grounding_llama32_70b_processed_score'
    ]
    uq_cols = ['semantic_entropy_min_max_scaled', 'predictive_entropy_min_max_scaled']
    
    confidence_cols = ['lexical_similarity_min_max_scaled', 'num_clusters_min_max_scaled']

    # Convert uncertainty -> confidence
    for col in uq_cols:
        df_vqa[f'confidence_from_{col}'] = 1 - df_vqa[col]
        confidence_cols.append(f'confidence_from_{col}')

    all_cols = gr_cols + confidence_cols + ['accuracy']
    df = df_vqa[all_cols].copy()

    # **Define T candidates for grid search (for exponent calibration)**
    exponent_T_candidates = np.linspace(0.1, 100, 500)
    
    # Dictionary to store cross-validation (CV) results
    cv = {}

    # Perform calibration with 5 random shuffles (CV iterations)
    for cv_iter in range(5):
        print(f"\n=== CV Iteration {cv_iter + 1} ===")
        
        # **Split into Calibration and Evaluation Sets** (using a different random seed each time)
        df_calib = df.sample(frac=0.2, random_state=42 + cv_iter)
        df_eval = df.drop(df_calib.index).copy()

        print(f"Calibration set size: {len(df_calib)}")
        print(f"Evaluation set size: {len(df_eval)}")
        
        cv[cv_iter] = {}  # store results for this iteration

        # First, compute and store the baseline ECE for the uncalibrated confidence columns
        for conf_col in confidence_cols:
            baseline_ece = compute_ece(df_calib[conf_col].values, df_calib["accuracy"].values)
            cv[cv_iter].setdefault(conf_col, {})["baseline_ece"] = baseline_ece

            print(f"\nProcessing baseline: {conf_col}")
            
            # Exponent Calibration (for each grounding model)
            for gr_col in gr_cols:
                best_T_exp, best_ece_exp = find_best_exponent_T(
                    df_calib[conf_col].values,
                    df_calib[gr_col].values,
                    df_calib["accuracy"].values,
                    exponent_T_candidates
                )
                print(f"  Exponent best T for {gr_col}: {best_T_exp:.2f} (calib ECE={best_ece_exp:.4f})")
                
                # Save calibrated results in cv
                cv[cv_iter][conf_col][gr_col] = {
                    "best_T": best_T_exp,
                    "calib_ece": best_ece_exp
                }
                
                # Apply exponent calibration on evaluation set
                df_eval[f'exponent_calibrated_{gr_col}_{conf_col}'] = exponent_calibration(
                    df_eval[conf_col].values,
                    df_eval[gr_col].values,
                    best_T_exp
                )
            
            # Only plot for the first CV iteration
            if cv_iter == 0:
                # Plot baseline + exponent calibrations (plotting code is unchanged)
                cols_to_plot = [conf_col] + [
                    f'exponent_calibrated_{gr_col}_{conf_col}' for gr_col in gr_cols
                ]

                # Build the plot title (upper-casing the main column name)
                plot_title_col = conf_col.replace('_min_max_scaled', '').replace('_', ' ').upper()
                plot_title = f'{base_dataset.upper()} - {plot_title_col} - CV Iteration 1'
                save_path = f'{base_dataset.upper()}_calibration_{plot_title_col}_CV1.png'

                plot_multiple_calibration_curve(
                    df_eval,
                    confidence_cols=cols_to_plot,
                    correctness_col='accuracy',
                    nbins=10,
                    plot_title=plot_title,
                    save_path=save_path
                )
                print(f"Plot saved to {save_path}")

    # Save the raw CV results to a pickle file for later use
    cv_pickle_file = f'{base_dataset.upper()}_cv_results.pkl'
    with open(cv_pickle_file, 'wb') as f:
        pickle.dump(cv, f)
    print(f"\nCV results saved to {cv_pickle_file}")
    
    # Aggregate ECE results and compute mean and variance for each combination
    summary_rows = []
    for conf_col in confidence_cols:
        # Aggregate baseline ECE for this confidence column
        baseline_ece_values = [cv[i][conf_col]["baseline_ece"] for i in range(5)]
        summary_rows.append({
            "baseline": conf_col,
            "grounding_model": "baseline",
            "mean_ece": np.mean(baseline_ece_values),
            "variance_ece": np.var(baseline_ece_values)
        })
        # Aggregate calibrated ECE for each grounding model
        for gr_col in gr_cols:
            ece_values = [cv[i][conf_col][gr_col]["calib_ece"] for i in range(5)]
            summary_rows.append({
                "baseline": conf_col,
                "grounding_model": gr_col,
                "mean_ece": np.mean(ece_values),
                "variance_ece": np.var(ece_values)
            })

    df_summary = pd.DataFrame(summary_rows)
    csv_file = f"{base_dataset.upper()}_ece_summary.csv"
    df_summary.to_csv(csv_file, index=False)
    print(f"\nAggregated ECE results and variance saved to {csv_file}")