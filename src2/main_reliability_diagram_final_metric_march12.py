import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import os # Added import
# mpl.rcParams['text.usetex'] = True

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
    # Set up figure with a larger size
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use a seaborn color palette for visually appealing, distinct colors
    colors = sns.color_palette('husl', len(confidence_cols))

    for i, confidence_col in enumerate(confidence_cols):
        confidences = df[confidence_col].values
        correctness = df[correctness_col].values

        # Binning
        bins = np.linspace(0, 1, nbins + 1)
        bin_indices = np.digitize(confidences, bins) - 1

        bin_conf_sums = np.zeros(nbins, dtype=float)
        bin_acc_sums = np.zeros(nbins, dtype=float)
        bin_counts = np.zeros(nbins, dtype=int)

        for j in range(len(confidences)):
            b = bin_indices[j]
            if 0 <= b < nbins:
                bin_conf_sums[b] += confidences[j]
                bin_acc_sums[b] += correctness[j]
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

        # Clean up the label for display (unchanged from original)
        display_label = confidence_col
        if 'sam' in display_label:
            display_label = 'With GM SAM'
        elif 'llama32_11b' in display_label:
            display_label = 'With GM LLaMA3.2V'
        elif 'llama32_70b' in display_label:
            display_label = 'LLaMA3.2v 70b'
        elif 'biomedclip' in display_label:
            display_label = 'BiomedCLIP'
        elif 'gemini' in display_label:
            display_label = 'Gemini'
        elif 'qwen_vl' in display_label:
            display_label = 'Qwen VL'
        elif 'clip' in display_label:
            display_label = 'With GM CLIP'
        elif 'temperature' in display_label:
            # display_label = 'Conf_baseline^(1/T)'
            # display_label = r'$Conf_{\text{baseline}}^{(1/T)}$'
            display_label = r'$Conf_{\mathrm{baseline}}^{(1/T)}$'
        elif 'lexical' or 'num_clusters' or 'semantic' or 'predictive' in display_label:
            # display_label = 'Conf_baseline' 
            display_label = r'$Conf_{\mathrm{baseline}}$'
        else:
            display_label = display_label.replace('_', ' ')

        # Plot with custom color
        ax.plot(
            bin_conf_means, bin_acc_means,
            marker='o', linestyle='-', color=colors[i],
            label=f'{display_label.strip()} (ECE={ece:.3f})'
        )
        
        # Add shading between the curve and the x==y line
        ax.fill_between(bin_conf_means, bin_acc_means, bin_conf_means, color=colors[i], alpha=0.1, zorder=1)

    # Perfect calibration line (thicker and more prominent)
    ax.plot([0, 1], [0, 1], '--', color='black', linewidth=2, label='Perfect Calibration')

    # Set axis limits and labels
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('Confidence', fontsize=40, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=40, fontweight='bold')
    ax.set_title(f'{plot_title}', fontsize=40, fontweight='bold')

    # Ensure proper spacing
    plt.tight_layout()
    
    # Move legend inside the plot (upper left)
    ax.legend(loc='best', fontsize=18)

    # Lighter grid
    ax.grid(True, linestyle='--', alpha=0.5)

    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
        
# def plot_multiple_calibration_curve(
#     df, confidence_cols, correctness_col,
#     nbins=10, plot_title='temp', save_path=None
# ):
#     """
#     Plots multiple reliability diagrams (confidence vs. accuracy) on the same plot,
#     calculates ECE for each, and optionally saves the figure to save_path.
    
#     Args:
#         df: DataFrame containing the data
#         confidence_cols: List of column names for confidence scores
#         correctness_col: Column name for accuracy (0/1)
#         nbins: Number of bins for the reliability diagram
#         plot_title: Title of the plot
#         save_path: Path to save the plot (if None, displays the plot)
#     """
#     fig, ax = plt.subplots(figsize=(8, 6))

#     for confidence_col in confidence_cols:
#         confidences = df[confidence_col].values
#         correctness = df[correctness_col].values

#         # Binning
#         bins = np.linspace(0, 1, nbins + 1)
#         bin_indices = np.digitize(confidences, bins) - 1

#         bin_conf_sums = np.zeros(nbins, dtype=float)
#         bin_acc_sums = np.zeros(nbins, dtype=float)
#         bin_counts = np.zeros(nbins, dtype=int)

#         for i in range(len(confidences)):
#             b = bin_indices[i]
#             if 0 <= b < nbins:
#                 bin_conf_sums[b] += confidences[i]
#                 bin_acc_sums[b] += correctness[i]
#                 bin_counts[b] += 1

#         # Compute bin-wise means and ECE
#         bin_conf_means = []
#         bin_acc_means = []
#         for b in range(nbins):
#             if bin_counts[b] > 0:
#                 mean_conf = bin_conf_sums[b] / bin_counts[b]
#                 mean_acc = bin_acc_sums[b] / bin_counts[b]
#                 bin_conf_means.append(mean_conf)
#                 bin_acc_means.append(mean_acc)

#         total = float(len(confidences))
#         ece = 0.0
#         for b in range(nbins):
#             if bin_counts[b] > 0:
#                 mean_conf_b = bin_conf_sums[b] / bin_counts[b]
#                 mean_acc_b = bin_acc_sums[b] / bin_counts[b]
#                 fraction_b = bin_counts[b] / total
#                 ece += fraction_b * abs(mean_acc_b - mean_conf_b)

#         # Clean up the label for display
#         display_label = confidence_col

#         # Example replacements:
#         if 'gd_sam' in display_label:
#             display_label = 'Grounding SAM'
#         elif 'llama32_11b' in display_label:
#             display_label = 'LLaMA3.2v 11 b'
#         elif 'llama32_70b' in display_label:
#             display_label = 'LLaMA3.2v 70b'
#         elif 'biomedclip' in display_label:
#             display_label = 'BiomedCLIP'
#         elif 'gemini' in display_label:
#             display_label = 'Gemini'
#         elif 'qwen_vl' in display_label:
#             display_label = 'Qwen VL'
#         else:
#             # Careful with replacements so we don't delete *all* text
#             # If your columns share these substrings, they’ll be removed in the label below
#             for r in [
#                 'exponent gd sammean_score_processed_',
#                 'exponent_grounding_llama32_11b_processed_score_',
#                 'final_conf_', '_min_max_scaled', 'confidence_from_', 'calibrated_'
#             ]:
#                 display_label = display_label.replace(r, '')
            
#             # Convert underscores to spaces
#             display_label = display_label.replace('_', ' ')
        
#         # Plot
#         ax.plot(
#             bin_conf_means, bin_acc_means,
#             marker='o', linestyle='-',
#             label=f'{display_label.strip()} (ECE={ece:.3f})'
#         )

#     # Perfect calibration line
#     ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')

#     ax.set_xlim([0, 1])
#     ax.set_ylim([0, 1])
#     ax.set_xlabel('Confidence', fontsize=14)
#     ax.set_ylabel('Accuracy', fontsize=14)
#     ax.set_title(f'Reliability Diagram - {plot_title}', fontsize=16)
#     ax.legend(loc='best')
#     ax.grid(True)  # Add grid to the plot

#     if save_path:
#         plt.savefig(save_path)
#         plt.close(fig)
#     else:
#         plt.show()


###########################
# 2. Calibration Methods (Updated)
###########################


def exponent_calibration_with_c(conf, gr_score, T, C):
    """
    New formula: new_conf = (conf * (gr_score^(1/T))) + C
    We also clip the result to [0,1] to ensure valid probability predictions.
    
    Args:
        conf: Baseline confidence scores
        gr_score: Grounding scores
        T: Temperature parameter
        C: Offset parameter in [0,1]
    Returns:
        Calibrated confidence scores in [0,1].
    """
    new_conf = conf * (gr_score ** (1.0 / T)) + C
    return np.clip(new_conf, 0.0, 1.0)


def find_best_exponent_TC(calib_conf, calib_gr, calib_acc, T_candidates, C_candidates):
    """
    Find the best (T, C) for exponent-based calibration by minimizing ECE:
        new_conf = (baseline_conf * (gr_score^(1/T))) + C
    
    Args:
        calib_conf: Calibration set confidence scores
        calib_gr:   Calibration set grounding scores
        calib_acc:  Calibration set accuracies
        T_candidates: Array of T values to search over
        C_candidates: Array of C values to search over [0,1]
    Returns:
        (best_T, best_C, lowest_ECE)
    """
    best_T = None
    best_C = None
    lowest_ece = float('inf')

    for T in T_candidates:
        for C in C_candidates:
            temp_conf = exponent_calibration_with_c(calib_conf, calib_gr, T, C)
            ece_val = compute_ece(temp_conf, calib_acc)
            if ece_val < lowest_ece:
                lowest_ece = ece_val
                best_T = T
                best_C = C
    
    return best_T, best_C, lowest_ece


def find_best_exponent_baseline_TC(calib_conf, calib_acc, T_candidates):
    """
    Find the best (T, C) for exponent-based calibration by minimizing ECE:
        new_conf = (baseline_conf * ^(1/T)) + C
    
    Args:
        calib_conf: Calibration set confidence scores
        calib_gr:   Calibration set grounding scores
        calib_acc:  Calibration set accuracies
        T_candidates: Array of T values to search over
        C_candidates: Array of C values to search over [0,1]
    Returns:
        (best_T, best_C, lowest_ECE)
    """
    best_T = None
    best_C = None
    lowest_ece = float('inf')

    for T in T_candidates:
        # for C in C_candidates:
        # temp_conf = exponent_calibration_with_c(calib_conf, T)
        temp_conf = calib_conf ** (1.0 / T)
        ece_val = compute_ece(temp_conf, calib_acc)
        if ece_val < lowest_ece:
            lowest_ece = ece_val
            best_T = T
            # best_C = C
    
    return best_T, lowest_ece


###########################
# 3. Main Script with 5 Random Shuffles for CV
###########################

if __name__ == '__main__':
    
    # base_dataset = 'slake'  # Change this to your desired dataset
    # base_dataset = 'slake_filtered'
    # base_dataset = 'slake_random'
    # base_dataset = 'vqa_random_march22' 
    base_dataset = 'vqa_random_march28'
    # base_dataset = 'slake_gemini_filtered_june1'
    # base_dataset = 'vqa_filtered_march22'
    print(f"=== Reliability Diagram Calibration for {base_dataset.upper()} ===")

    # Create directory for saving detailed evaluation DataFrames
    eval_dfs_dir = f'{base_dataset.upper()}_eval_dfs'
    os.makedirs(eval_dfs_dir, exist_ok=True)
    print(f"Evaluation DataFrames will be saved in: {eval_dfs_dir}")
    
    # **Load Data** (Adjust your path or DataFrame as needed)
    # data_path = 'my_outputs_slake/merged_scaled.csv' 
    data_path = f'my_outputs_{base_dataset}/merged_scaled.csv'
    df_vqa = pd.read_csv(data_path)

    # **Define Columns**
    # gr_cols = [
    #     'grounding_biomedclip_mean_min_max_scaled',
    #     'grounding_llama32_11b_mean_min_max_scaled',
    #     # 'grounding_llama32_11b_processed_score_min_max_scaled',
    #     'grounding_qwen_vl_mean_min_max_scaled',
    #     # 'grounding_qwen_vl_processed_score_min_max_scaled',
    #     # 'grounding_gemini_mean_min_max_scaled'
    # ] # for slake 
    gr_cols = [ 
    'grounding_grounding_sam_mean_min_max_scaled',
       'grounding_qwen_vl_mean_min_max_scaled',
       'grounding_llama32_11b_mean_min_max_scaled',
       'grounding_clip_mean_min_max_scaled',
    #    'grounding_gemini_mean_min_max_scaled'
    ] # for vqa
    # gr_cols = ['grounding_llama32_11b_mean_min_max_scaled'] # for plotting only one grounding model, since we saw llama32_11b is the best
    uq_cols = ['semantic_entropy_min_max_scaled', 'predictive_entropy_min_max_scaled']
    # uq_cols = []
    
    confidence_cols = ['lexical_similarity_min_max_scaled', 'num_clusters_min_max_scaled']
    # confidence_cols = [
    #     'lexical_similarity_min_max_scaled',
    # ]

    # Convert uncertainty -> confidence
    for col in uq_cols:
        df_vqa[f'confidence_from_{col}'] = 1 - df_vqa[col]
        confidence_cols.append(f'confidence_from_{col}')

    all_cols = gr_cols + confidence_cols + ['accuracy']
    df = df_vqa[all_cols].copy()

    # **Define parameter candidates for exponent + C calibration**
    exponent_T_candidates = np.linspace(0.1, 100, 500)
    exponent_C_candidates = np.linspace(0, 0.5, 51)  # e.g. 51 steps in [0,1]

    # Dictionary to store cross-validation (CV) results
    cv = {}

    no_of_cv_iterations = 5
    # Perform calibration with 5 random shuffles (CV iterations)
    for cv_iter in range(no_of_cv_iterations):
        print(f"\\n=== CV Iteration {cv_iter + 1} ===")
        
        # **Split into Calibration (20%) and Evaluation (80%) Sets**
        df_calib = df.sample(frac=0.2, random_state=42 + cv_iter)
        df_eval = df.drop(df_calib.index).copy()

        print(f"Calibration set size: {len(df_calib)}")
        print(f"Evaluation set size: {len(df_eval)}")
        
        cv[cv_iter] = {}  # store results for this iteration
        
        # Compute and store baseline ECE for uncalibrated columns
        for conf_col in confidence_cols:
            baseline_ece = compute_ece(df_eval[conf_col].values, df_eval["accuracy"].values)
            if conf_col not in cv[cv_iter]:
                cv[cv_iter][conf_col] = {}
            cv[cv_iter][conf_col]["baseline_ece"] = baseline_ece
                
            print(f"\nProcessing baseline: {conf_col}")
            
            # Find the best temperature scaling parameter for baseline
            best_T_exp, best_ece_exp = find_best_exponent_baseline_TC(
                df_calib[conf_col].values,
                df_calib["accuracy"].values,
                exponent_T_candidates
            )
            print(f"Baseline Best T={best_T_exp:.3f} (calib ECE={best_ece_exp:.4f})")
            
            
            
            # Store the best parameters and calibration ECE without losing baseline_ece
            cv[cv_iter][conf_col].update({
                "best_T": best_T_exp,
                "calib_ece": best_ece_exp
            })
            
            # Apply temperature scaling on evaluation set for baseline
            df_eval[f'temperature_scaled_{conf_col}'] = df_eval[conf_col].values ** (1.0 / best_T_exp)
            # Calculate and store eval ECE for temperature-scaled baseline for this fold
            eval_ece_ts_baseline = compute_ece(df_eval[f'temperature_scaled_{conf_col}'].values, df_eval["accuracy"].values)
            cv[cv_iter][conf_col]["eval_ece_temp_scaled_baseline"] = eval_ece_ts_baseline
            
            # For each grounding model, find the best (T, C)
            for gr_col in gr_cols:
                best_T_exp, best_C_exp, best_ece_exp = find_best_exponent_TC(
                    df_calib[conf_col].values,
                    df_calib[gr_col].values,
                    df_calib["accuracy"].values,
                    exponent_T_candidates,
                    exponent_C_candidates
                )
                print(f"Best T={best_T_exp:.3f}, C={best_C_exp:.3f} (calib ECE={best_ece_exp:.4f}) for {conf_col} with {gr_col}")
                
                # Store the best parameters and calibration ECE
                cv[cv_iter][conf_col][gr_col] = {
                    "best_T": best_T_exp,
                    "best_C": best_C_exp,
                    "calib_ece_exponent_calibrated": best_ece_exp # ECE on calibration set
                }
                
                # Apply exponent + C calibration on evaluation set
                df_eval[f'exponent_calibrated_{gr_col}_{conf_col}'] = exponent_calibration_with_c(
                    df_eval[conf_col].values,
                    df_eval[gr_col].values,
                    best_T_exp,
                    best_C_exp
                )
                # Compute and store ECE on evaluation set for this method
                eval_ece_exponent_calibrated = compute_ece(
                    df_eval[f'exponent_calibrated_{gr_col}_{conf_col}'].values,
                    df_eval["accuracy"].values
                )
                cv[cv_iter][conf_col][gr_col]["eval_ece_exponent_calibrated"] = eval_ece_exponent_calibrated
                
                # Calibrate grounding scores alone using temperature scaling
                # Corrected call to find_best_exponent_baseline_TC:
                # Calibrates df_calib[gr_col] against df_calib["accuracy"]
                current_gr_scores_calib = df_calib[gr_col].values
                current_accuracies_calib = df_calib["accuracy"].values
                
                best_T_exp_gr, calib_ece_gr_temp_scaled = find_best_exponent_baseline_TC(
                    current_gr_scores_calib,    # Grounding scores from calib set as confidences
                    current_accuracies_calib,   # Accuracies from calib set
                    exponent_T_candidates       # T candidates
                )
                print(f"Best T for {gr_col} alone ={best_T_exp_gr:.3f} (calib ECE={calib_ece_gr_temp_scaled:.4f})")
                
                # Store results for grounding-only temperature scaling
                cv[cv_iter][conf_col][gr_col]["best_T_exp_gr"] = best_T_exp_gr
                cv[cv_iter][conf_col][gr_col]["calib_ece_gr_temp_scaled"] = calib_ece_gr_temp_scaled # ECE on calib set

                # Apply temperature scaling for grounding scores on evaluation set
                # The column name f'temperature_scaled_{gr_col}_{conf_col}' might be slightly misleading
                # as it only scales gr_col based on best_T_exp_gr.
                df_eval[f'temperature_scaled_gr_only_{gr_col}'] = df_eval[gr_col].values ** (1.0 / best_T_exp_gr)
                
                # Compute and store ECE on evaluation set for grounding-only temperature scaling
                eval_ece_gr_temp_scaled = compute_ece(
                    df_eval[f'temperature_scaled_gr_only_{gr_col}'].values,
                    df_eval["accuracy"].values
                )
                cv[cv_iter][conf_col][gr_col]["eval_ece_gr_temp_scaled"] = eval_ece_gr_temp_scaled
            
            # Plot reliability diagrams (only for the first CV iteration as an example)
            if cv_iter == 0:
                cols_to_plot = [conf_col, f'temperature_scaled_{conf_col}'] + [
                    f'exponent_calibrated_{gr_col}_{conf_col}' for gr_col in gr_cols
                ] 

                plot_title_col = conf_col.replace('_min_max_scaled', '').replace('_', ' ').upper()
                if 'lexical' in plot_title_col.lower():
                    plot_title = 'LexSim'
                elif 'num' in plot_title_col.lower():
                    plot_title = 'NumSets'
                elif 'semantic' in plot_title_col.lower():
                    plot_title = 'SemEnt'
                elif 'predictive' in plot_title_col.lower():
                    plot_title = 'PredEnt'
                else:
                    plot_title = plot_title_col.lower()
                save_path = f'{base_dataset.upper()}_calibration_{plot_title_col}_CV1.png'

                # We use df_eval (80%) for plotting so there's no overlap with calibration
                plot_multiple_calibration_curve(
                    df_eval,
                    confidence_cols=cols_to_plot,
                    correctness_col='accuracy',
                    nbins=10,
                    plot_title=plot_title,
                    save_path=save_path
                )
                print(f"Plot saved to {save_path}")

        # Save the df_eval for the current fold
        eval_df_filename = os.path.join(eval_dfs_dir, f'eval_df_fold_{cv_iter + 1}.csv')
        df_eval.to_csv(eval_df_filename, index=False)
        print(f"Saved evaluation data for fold {cv_iter + 1} to {eval_df_filename}")

    # Save the raw CV results to a pickle file
    cv_pickle_file = f'{base_dataset.upper()}_cv_results.pkl'
    with open(cv_pickle_file, 'wb') as f:
        pickle.dump(cv, f)
    print(f"\nCV results saved to {cv_pickle_file}")
    
    # Aggregate ECE results and compute mean and variance for each combination
    summary_rows = []
    for conf_col in confidence_cols:
        # Aggregate baseline ECE for this confidence column (already uses eval ECE)
        baseline_ece_values = [cv[i][conf_col]["baseline_ece"] for i in range(no_of_cv_iterations)]
        summary_rows.append({
            "baseline_confidence": conf_col,
            "calibration_method": "None (Baseline)",
            "grounding_model_context": "N/A",
            "mean_eval_ece": np.mean(baseline_ece_values),
            "variance_eval_ece": np.var(baseline_ece_values),
            "mean_T": np.nan,
            "mean_C": np.nan,
            "mean_calib_ece": np.nan # Baseline ECE is on eval set
        })

        # Aggregate temperature-scaled baseline confidence ECE (already uses eval ECE)
        # Need to ensure df_eval is the one from the last CV iteration, or recompute ECEs for each fold.
        # The current code for temp_scaled_ece_values takes only the last df_eval. This should be fixed.
        
        # Re-calculating eval ECE for temperature-scaled baseline for each fold
        temp_scaled_baseline_eval_ece_values = []
        for i in range(no_of_cv_iterations):
            # Re-split to get the correct df_eval for fold i, or ensure df_eval used for ECE calc is per fold
            # For simplicity, let's assume cv[i][conf_col]['temperature_scaled_eval_ece'] was stored
            # We need to add this storage in the main loop.
            # Let's assume it's: cv[i][conf_col]["eval_ece_temp_scaled_baseline"]
            # This was not explicitly stored before for the summary, let's add it.
            # The column f'temperature_scaled_{conf_col}' is on df_eval which changes per iteration.
            # So, we must compute ECE per fold for summary.
            
            # Placeholder: This part needs df_eval from each fold to be accurate for summary.
            # The current df_eval is from the last fold.
            # For now, we'll use the stored best_T and apply to the last df_eval as an approximation,
            # or ideally, this ECE should be computed and stored per fold.
            # Let's assume we stored eval_ece_temp_scaled_baseline in the CV loop:
            # cv[cv_iter][conf_col]["eval_ece_temp_scaled_baseline"] = compute_ece(df_eval[f'temperature_scaled_{conf_col}'].values, df_eval["accuracy"].values)
            # This should be added after line 454.
            # For now, this part of summary might be slightly off if not handled carefully with per-fold df_eval.
            # The current code `temp_scaled_ece_values = [compute_ece(df_eval[f'temperature_scaled_{conf_col}'].values, df_eval["accuracy"].values)]`
            # is indeed using the LAST df_eval. This is a bug in original summary logic.

            # Corrected approach for temperature-scaled baseline for summary:
            # We need to retrieve or recompute ECE for each fold.
            # Let's assume 'eval_ece_temp_scaled_baseline' is stored per fold in cv results.
            # This requires adding its calculation in the main loop:
            # After df_eval[f'temperature_scaled_{conf_col}'] = ... (line 454)
            # Add: eval_ece_ts_baseline = compute_ece(df_eval[f'temperature_scaled_{conf_col}'].values, df_eval["accuracy"].values)
            #      cv[cv_iter][conf_col]["eval_ece_temp_scaled_baseline"] = eval_ece_ts_baseline

            # Assuming eval_ece_temp_scaled_baseline is now stored:
            temp_scaled_baseline_eval_ece_values.append(cv[i][conf_col].get("eval_ece_temp_scaled_baseline", np.nan))


        T_values_baseline_temp = [cv[i][conf_col]["best_T"] for i in range(no_of_cv_iterations)]
        calib_ece_baseline_temp = [cv[i][conf_col]["calib_ece"] for i in range(no_of_cv_iterations)] # This is calib ECE for baseline temp scaling
        
        summary_rows.append({
            "baseline_confidence": conf_col,
            "calibration_method": "Temperature Scaling (Baseline Conf)",
            "grounding_model_context": "N/A",
            "mean_eval_ece": np.nanmean(temp_scaled_baseline_eval_ece_values), # Use nanmean
            "variance_eval_ece": np.round(np.nanvar(temp_scaled_baseline_eval_ece_values), 7),
            "mean_T": np.mean(T_values_baseline_temp),
            "mean_C": np.nan,
            "mean_calib_ece": np.mean(calib_ece_baseline_temp)
        })

        # Aggregate exponent+C ECE for each grounding model
        for gr_col in gr_cols:
            eval_ece_values_exp_c = [cv[i][conf_col][gr_col]["eval_ece_exponent_calibrated"] for i in range(no_of_cv_iterations)]
            calib_ece_values_exp_c = [cv[i][conf_col][gr_col]["calib_ece_exponent_calibrated"] for i in range(no_of_cv_iterations)]
            T_values_exp_c = [cv[i][conf_col][gr_col]["best_T"] for i in range(no_of_cv_iterations)]
            C_values_exp_c = [cv[i][conf_col][gr_col]["best_C"] for i in range(no_of_cv_iterations)]
            summary_rows.append({
                "baseline_confidence": conf_col,
                "calibration_method": "Exponent + C",
                "grounding_model_context": gr_col,
                "mean_eval_ece": np.mean(eval_ece_values_exp_c),
                "variance_eval_ece": np.round(np.var(eval_ece_values_exp_c), 7),
                "mean_T": np.mean(T_values_exp_c),
                "mean_C": np.mean(C_values_exp_c),
                "mean_calib_ece": np.mean(calib_ece_values_exp_c)
            })

            # Add summary for temperature-scaled grounding scores (EVAL ECE)
            eval_ece_gr_temp = [cv[i][conf_col][gr_col]["eval_ece_gr_temp_scaled"] for i in range(no_of_cv_iterations)]
            calib_ece_gr_temp = [cv[i][conf_col][gr_col]["calib_ece_gr_temp_scaled"] for i in range(no_of_cv_iterations)]
            T_values_gr_temp = [cv[i][conf_col][gr_col]["best_T_exp_gr"] for i in range(no_of_cv_iterations)]
            summary_rows.append({
                "baseline_confidence": conf_col, # Context of the baseline confidence
                "calibration_method": "Temperature Scaling (Grounding Only)",
                "grounding_model_context": gr_col, # Specifies which grounding score was scaled
                "mean_eval_ece": np.mean(eval_ece_gr_temp),
                "variance_eval_ece": np.round(np.var(eval_ece_gr_temp), 7),
                "mean_T": np.mean(T_values_gr_temp),
                "mean_C": np.nan,
                "mean_calib_ece": np.mean(calib_ece_gr_temp) # This is the calib ECE for this method
            })
        
    df_summary = pd.DataFrame(summary_rows)
    # Adjust column names for clarity if needed, e.g., 'mean_eval_ece', 'variance_eval_ece', 'mean_calib_ece'
    df_summary.rename(columns={
        "baseline_confidence": "BaselineConfidence",
        "calibration_method": "CalibrationMethod",
        "grounding_model_context": "GroundingContext",
        "mean_eval_ece": "MeanEvalECE",
        "variance_eval_ece": "VarEvalECE",
        "mean_T": "MeanT",
        "mean_C": "MeanC",
        "mean_calib_ece": "MeanCalibECE"
    }, inplace=True)
    
    csv_file = f"{base_dataset.upper()}_ece_summary_revised.csv"
    df_summary.to_csv(csv_file, index=False)
    print(f"\nAggregated ECE results and variance saved to {csv_file}")