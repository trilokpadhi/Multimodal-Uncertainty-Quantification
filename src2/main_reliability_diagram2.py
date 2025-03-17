import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

def compute_ece(bin_centers, avg_accuracies):
    """
    ECE = mean of |avg_accuracy[i] - bin_center[i]|
    over all i where that bin is non-empty.
    """
    return np.mean(np.abs(avg_accuracies - bin_centers))


###############################################################################
# 1) Helper: min-max scale
###############################################################################
def min_max_scale(array):
    arr = np.array(array, dtype=float)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-12:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

###############################################################################
# 2) Helper: calibrate_model
###############################################################################
def calibrate_model(
    df,
    feature_cols,
    accuracy_col="accuracy",
    uncertainty_cols=None,
    val_percent=0.20,
    degree=2,
    alpha=3.0,
    random_state=123
):
    """
    1) Build X from feature_cols (optionally 1 - scaled if 'uncertainty_cols').
    2) Split into calibration vs test.
    3) Fit PolynomialFeatures+Ridge on calibration set.
    4) Predict on test => conf_test_scaled in [0..1].
    5) Return (conf_test_scaled, y_test).
    """
    if uncertainty_cols is None:
        uncertainty_cols = []

    # Construct X
    X_list = []
    for col in feature_cols:
        raw_values = df[col].values
        scaled = min_max_scale(raw_values)
        if col in uncertainty_cols:
            scaled = 1.0 - scaled
        X_list.append(scaled)
    X_all = np.column_stack(X_list)

    # Accuracy
    y_all = df[accuracy_col].values

    # Split => calibration vs test
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_all, y_all,
        test_size=(1 - val_percent),
        random_state=random_state,
        shuffle=True
    )

    # Build pipeline
    pipe = make_pipeline(
        PolynomialFeatures(degree=degree),
        Ridge(alpha=alpha)
    )
    pipe.fit(X_cal, y_cal)

    # Predict on test => scale
    y_pred_test = pipe.predict(X_test)
    conf_test_scaled = min_max_scale(y_pred_test)

    return conf_test_scaled, y_test

###############################################################################
# 3) Helper: compute_reliability_curve
###############################################################################
def compute_reliability_curve(confidences, accuracies, num_bins=10):
    """
    Bin confidences in [0,1] into num_bins intervals.
    Returns (bin_centers, avg_accuracies).
    """
    bins = np.linspace(0, 1, num_bins + 1)  # e.g. 11 edges => 10 bins
    bin_indices = np.digitize(confidences, bins) - 1

    avg_accuracies = []
    bin_centers = []
    for i in range(len(bins) - 1):
        mask = (bin_indices == i)
        if np.any(mask):
            avg_acc = np.mean(accuracies[mask])
            avg_accuracies.append(avg_acc)
            bin_center = 0.5 * (bins[i] + bins[i+1])
            bin_centers.append(bin_center)

    return np.array(bin_centers), np.array(avg_accuracies)

###############################################################################
# 4) Helper: Plot multiple reliability curves on one figure
###############################################################################
def plot_reliability_curves_in_one_figure(curves_data, save_path, title=None):
    """
    curves_data: list of dicts, each with:
       {
         "bin_centers": np.array(...),
         "avg_accuracies": np.array(...),
         "label": "some string"
       } 
    Plots them all on one figure. Saves to 'save_path'.
    """
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(8,6))

    # Perfect calibration
    ax.plot([0,1],[0,1], '--', color='gray', label='Perfect Calibration')

    # Plot each curve
    for entry in curves_data:
        bins = entry["bin_centers"]
        accs = entry["avg_accuracies"]
        lbl = entry["label"]
        ax.plot(bins, accs, marker='o', linestyle='-', linewidth=2, label=lbl)
        ax.fill_between(bins, bins, accs, alpha=0.1)

    ax.set_xlim([0,1])
    ax.set_ylim([0,1]) 
    ax.set_xlabel("Confidence", fontsize=14)
    ax.set_ylabel("Accuracy", fontsize=14)
    if title:
        ax.set_title(title, fontsize=16)
    ax.legend(loc='upper left', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Saved reliability diagram => {save_path}")

###############################################################################
# 5) MAIN EXAMPLE
###############################################################################

############################################
# 2) The main function to run each model N times
############################################
def evaluate_models_ece_n_runs(
    df,
    model_configs,
    n_runs=5,
    output_csv="all_ece_runs.csv",
    accuracy_col="accuracy",
    val_percent=0.20,
    degree=2,
    alpha=3.0
):
    """
    model_configs: list of dicts like:
      {
        "label": "Lexical Only",
        "feature_cols": [...],
        "uncertainty_cols": [...], # optional
      }
    Runs each config n_runs times with different random_state, 
    stores bin-level details & ECE in CSV, 
    plus summary rows for average ECE & variance of ECE.
    """
    all_rows = []

    for config in model_configs:
        label = config["label"]
        feat_cols = config["feature_cols"]
        unct_cols = config.get("uncertainty_cols", [])

        ece_values = []

        for run_idx in range(n_runs):
            random_state = 123 + run_idx
            # 1) Calibrate
            conf, acc = calibrate_model(
                df=df,
                feature_cols=feat_cols,
                accuracy_col=accuracy_col,
                uncertainty_cols=unct_cols,
                val_percent=val_percent,
                degree=degree,
                alpha=alpha,
                random_state=random_state
            )
            # 2) Compute reliability curve
            bin_centers, avg_accuracies = compute_reliability_curve(conf, acc, num_bins=10)
            # 3) ECE
            ece_val = compute_ece(bin_centers, avg_accuracies)
            ece_values.append(ece_val)

            # 4) Store bin-level details
            for b, a in zip(bin_centers, avg_accuracies):
                row = {
                    "model_label": label,
                    "run_idx": run_idx,
                    "bin_center": b,
                    "avg_accuracy": a,
                    "ece_run": ece_val
                }
                all_rows.append(row)

        # After n_runs => summary rows
        ece_mean = np.mean(ece_values)
        ece_var  = np.var(ece_values)

        # SUMMARY row => store the mean ECE
        summary_row = {
            "model_label": label,
            "run_idx": "SUMMARY",
            "bin_center": np.nan,
            "avg_accuracy": np.nan,
            "ece_run": ece_mean
        }
        all_rows.append(summary_row)

        # VARIANCE row => store the var or std
        variance_row = {
            "model_label": label,
            "run_idx": "VARIANCE",
            "bin_center": np.nan,
            "avg_accuracy": np.nan,
            "ece_run": ece_var
        }
        all_rows.append(variance_row)

    # Build and write the table
    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(output_csv, index=False)
    print(f"[INFO] Wrote details of {n_runs} runs for all models => {output_csv}")

    return df_out

if __name__ == "__main__":

    # Example: read your CSV
    # (Change the path as needed.)
    # df = pd.read_csv("/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/my_outputs/merged_with_grounding_processed_internal_external_g_entropy_feb19.csv") # for slake 
    df = pd.read_csv("/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/my_outputs/merged_with_grounding_processed_internal_external_g_feb20.csv") # for vqa

    # Suppose your "internal" grounding column is:
    internal_col = "grounding_internal_llama32V_cluster_wise"
    # internal_col = "grounding_internal_llama32V_with_cluster_wise_with_entropy"
    # Suppose your "external" grounding column is:
    external_col = "grounding_external"
    # external_col = "grounding_external_with_entropy"
    
    # Four baselines you mentioned
    baselines = [
        {"col": "predictive_entropy_min_max_scaled", "label": "Predictive Entropy", "uncertainty_col": "predictive_entropy_min_max_scaled"},
        {"col": "lexical_similarity_min_max_scaled", "label": "Lexical Similarity"},
        {"col": "semantic_entropy_min_max_scaled",  "label": "Semantic Entropy", "uncertainty_col": "semantic_entropy_min_max_scaled"},
        {"col": "num_clusters_min_max_scaled", "label": "Num Clusters"}
    ]

    # For each baseline, we want 3 curves on one plot:
    #   1) Baseline only
    #   2) Baseline + internal
    #   3) Baseline + external
    
    # We'll produce one CSV (bin data) + one PNG for each baseline.

    for baseline_info in baselines:
        base_col = baseline_info["col"]
        base_label = baseline_info["label"]
        if "uncertainty_col" in baseline_info:
            uncertainty_cols = baseline_info["uncertainty_col"]
        else:
            uncertainty_cols = None
        # 1) Baseline only
        conf_base, acc_base = calibrate_model(
            df=df,
            feature_cols=[base_col],
            uncertainty_cols=uncertainty_cols,  # or e.g. [base_col] if it is an 'uncertainty' needing (1 - scaled)
            accuracy_col="accuracy",
            val_percent=0.20,
            degree=2,
            alpha=3.0,
            random_state=123
        )
        bin_centers_base, avg_acc_base = compute_reliability_curve(conf_base, acc_base, num_bins=10)

        # 2) Baseline + internal
        conf_int, acc_int = calibrate_model(
            df=df,
            feature_cols=[base_col, internal_col],
            uncertainty_cols=uncertainty_cols,
            accuracy_col="accuracy",
            val_percent=0.20,
            degree=2,
            alpha=3.0,
            random_state=123
        )
        bin_centers_int, avg_acc_int = compute_reliability_curve(conf_int, acc_int, num_bins=10)

        # 3) Baseline + external
        conf_ext, acc_ext = calibrate_model(
            df=df,
            feature_cols=[base_col, external_col],
            uncertainty_cols=uncertainty_cols,
            accuracy_col="accuracy",
            val_percent=0.20,
            degree=2,
            alpha=3.0,
            random_state=123
        )
        bin_centers_ext, avg_acc_ext = compute_reliability_curve(conf_ext, acc_ext, num_bins=10)

        # Build a small DataFrame with these 3 curves => CSV
        rows = []
        # Baseline only
        for b, a in zip(bin_centers_base, avg_acc_base):
            rows.append({
                "model_label": f"{base_label} Only",
                "bin_center": b,
                "avg_accuracy": a
            })
        # Baseline + internal
        for b, a in zip(bin_centers_int, avg_acc_int):
            rows.append({
                "model_label": f"{base_label} + Internal",
                "bin_center": b,
                "avg_accuracy": a
            })
        # Baseline + external
        for b, a in zip(bin_centers_ext, avg_acc_ext):
            rows.append({
                "model_label": f"{base_label} + External",
                "bin_center": b,
                "avg_accuracy": a
            })

        df_out = pd.DataFrame(rows)
        csv_name = f"reliability_{base_label.replace(' ', '_')}.csv"
        df_out.to_csv(csv_name, index=False)
        print(f"[INFO] Wrote {csv_name}")

        # Now plot them all on one figure
        # We'll prepare a 'curves_data' list
        curves_data = [
            {
                "bin_centers": bin_centers_base,
                "avg_accuracies": avg_acc_base,
                "label": f"{base_label} Only"
            },
            {
                "bin_centers": bin_centers_int,
                "avg_accuracies": avg_acc_int,
                "label": f"{base_label} + Internal"
            },
            {
                "bin_centers": bin_centers_ext,
                "avg_accuracies": avg_acc_ext,
                "label": f"{base_label} + External"
            }
        ]

        png_name = f"slake_20_feb_reliability_{base_label.replace(' ', '_')}.png"
        plot_reliability_curves_in_one_figure(
            curves_data=curves_data,
            save_path=png_name,
            title=base_label
        )
        
    model_configs = [
        # Predictive Entropy
        {
            "label": "Predictive Only",
            "feature_cols": ["predictive_entropy_min_max_scaled"],
            # If it's truly an uncertainty measure, add "predictive_entropy_min_max_scaled" to 'uncertainty_cols'
            "uncertainty_cols": ["predictive_entropy_min_max_scaled"]
        },
        {
            "label": "Predictive + Internal",
            "feature_cols": ["predictive_entropy_min_max_scaled", internal_col],
            "uncertainty_cols": ["predictive_entropy_min_max_scaled"]
        },
        {
            "label": "Predictive + External",
            "feature_cols": ["predictive_entropy_min_max_scaled", external_col],
            "uncertainty_cols": ["predictive_entropy_min_max_scaled"]
        },

        # Lexical Similarity
        {
            "label": "Lexical Only",
            "feature_cols": ["lexical_similarity_min_max_scaled"]
        },
        {
            "label": "Lexical + Internal",
            "feature_cols": ["lexical_similarity_min_max_scaled", internal_col]
        },
        {
            "label": "Lexical + External",
            "feature_cols": ["lexical_similarity_min_max_scaled", external_col]
        },

        # Semantic Entropy
        {
            "label": "Semantic Only",
            "feature_cols": ["semantic_entropy_min_max_scaled"],
            "uncertainty_cols": ["semantic_entropy_min_max_scaled"]
        },
        {
            "label": "Semantic + Internal",
            "feature_cols": ["semantic_entropy_min_max_scaled", internal_col],
            "uncertainty_cols": ["semantic_entropy_min_max_scaled"]
        },
        {
            "label": "Semantic + External",
            "feature_cols": ["semantic_entropy_min_max_scaled", external_col],
            "uncertainty_cols": ["semantic_entropy_min_max_scaled"]
        },

        # Num Clusters
        {
            "label": "Clusters Only",
            "feature_cols": ["num_clusters_min_max_scaled"]
        },
        {
            "label": "Clusters + Internal",
            "feature_cols": ["num_clusters_min_max_scaled", internal_col]
        },
        {
            "label": "Clusters + External",
            "feature_cols": ["num_clusters_min_max_scaled", external_col]
        },
    ]
      
    # Run each model config for N=5 times => CSV
    output_csv = "slake_20_feb_all_ece_runs_for_12_variants.csv"
    df_ece = evaluate_models_ece_n_runs(
        df=df,
        model_configs=model_configs,
        n_runs=5,
        output_csv=output_csv,
        accuracy_col="accuracy",
        val_percent=0.20,
        degree=2,
        alpha=3.0
    )
    
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import math
# from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
# from sklearn.linear_model import Ridge
# from sklearn.pipeline import make_pipeline
# from sklearn.model_selection import train_test_split


# # df = pd.read_csv('/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/my_outputs/merged_with_grounding_processed.csv')

# df = pd.read_csv('/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/my_outputs/merged_with_grounding_processed_external_g_entropy.csv')

# # Below are the columns in the dataframe:
# # Index(['question_id', 'accuracy', 'predictive_entropy', 'lexical_similarity',
# #        'semantic_entropy', 'num_clusters', 'responses', 'resp_log_probs',
# #        'question', 'grounding_biomedclip', 'grounding_biomedclip_processed',
# #        'grounding_llama32_11b', 'grounding_llama32_11b_processed',
# #        'grounding_llama32_70b', 'grounding_llama32_70b_processed',
# #        'grounding_qwen_vl', 'grounding_qwen_vl_processed',
# #        'max_grounding_biomedclip', 'predictive_entropy_min_max_scaled',
# #        'lexical_similarity_min_max_scaled', 'semantic_entropy_min_max_scaled',
# #        'num_clusters_min_max_scaled', 'grounding_biomedclip_processed_scaled',
# #        'grounding_biomedclip_processed_scaled_average',
# #        'grounding_llama32_11b_processed_score',
# #        'grounding_llama32_70b_processed_score',
# #        'grounding_qwen_vl_processed_score',
# #        'grounding_llama32_11b_llama32_70b_agreement',
# #        'grounding_llama32_11b_qwen_vl_agreement',
# #        'grounding_llama32_70b_qwen_vl_agreement',
# #        'grounding_models_entropy_llama11b_llama70b_qwenvl',
# #        'grounding_models_entropy_llama11b_llama70b',
# #        'grounding_models_entropy_llama11b_qwenvl',
# #        'grounding_models_entropy_llama70b_qwenvl'],
# #       dtype='object')

# def plot_reliability_diagram(confidences, accuracies, confidence_type, save_path):
#     """
#     reliability diagram, saves the figure to 'save_path'.
#     """
#     plt.style.use('ggplot')
#     fig, ax = plt.subplots(figsize=(8, 6))

#     # Binning
#     bins = np.linspace(0, 1, 11)
#     bin_indices = np.digitize(confidences, bins) - 1

#     avg_accuracies = []
#     bin_centers = []
#     for i in range(len(bins) - 1):
#         mask = bin_indices == i
#         if mask.any():
#             avg_acc = np.mean(accuracies[mask])
#             avg_accuracies.append(avg_acc)
#             bin_center = (bins[i] + bins[i+1]) / 2
#             bin_centers.append(bin_center)

#     avg_accuracies = np.array(avg_accuracies)
#     bin_centers = np.array(bin_centers)

#     ax.plot(bin_centers, avg_accuracies, marker='o', linestyle='-', linewidth=2, label=f'{confidence_type}')
#     ax.plot([0,1],[0,1], '--', color='gray', label='Perfect')

#     ax.fill_between(bin_centers, bin_centers, avg_accuracies, alpha=0.1)
#     ax.set_xlim([0,1])
#     ax.set_ylim([0,1])
#     ax.set_xlabel("Confidence")
#     ax.set_ylabel("Accuracy")
#     ax.set_title(f"({confidence_type})")
#     ax.legend(loc='lower right')

#     # ECE + MCE
#     if len(bin_centers) > 0:
#         ece = np.mean(np.abs(avg_accuracies - bin_centers))
#         mce = np.max(np.abs(avg_accuracies - bin_centers))
#     else:
#         ece, mce = 0, 0
#     stats_text = f"ECE: {ece:.3f}\nMCE: {mce:.3f}"
#     ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12, 
#             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close(fig)
#     print(f"[INFO] Saved reliability diagram => {save_path}")
    
    
# def min_max_scale(array):
#     arr = np.array(array, dtype=float)
#     mn, mx = arr.min(), arr.max()
#     if mx - mn < 1e-12:
#         # Avoid dividing by zero if all values are the same
#         return np.zeros_like(arr)
#     return (arr - mn) / (mx - mn)



# def calibrate_and_plot_reliability(
#     df,
#     feature_cols,         # list of columns to use as features
#     accuracy_col,         # e.g., "accuracy"
#     uncertainty_cols=None,# which of these feature_cols are 'uncertainty' (apply 1 - scaled_value)
#     val_percent=0.20,     # fraction for calibration set
#     degree=2,
#     alpha=3.0,
#     save_path="reliability_diagram.png",
#     confidence_label="Calibrated Model"
# ):
#     """
#     Fits a multi-feature calibration model (PolynomialFeatures + Ridge)
#     using `val_percent` of data as calibration set. Then plots a reliability 
#     diagram with the predicted confidence vs. actual accuracy on the test set.
    
#     Steps:
#       1. For each feature_col:
#          - min-max scale the column.
#          - if it's in uncertainty_cols, do "confidence = 1 - scaled_value".
#       2. Split data into calibration vs. test sets.
#       3. Fit polynomial + Ridge on calibration set.
#       4. Predict on test set, then min-max scale predictions.
#       5. Plot reliability diagram of predicted (scaled) confidence vs. accuracy.
#     """
#     if uncertainty_cols is None:
#         uncertainty_cols = []

#     # 1) Construct the feature matrix X_all
#     #    For each column, scale it and possibly invert if it's an uncertainty metric.
#     X_list = []
#     for col in feature_cols:
#         raw_values = df[col].values
#         scaled = min_max_scale(raw_values)    # scaled in [0..1]
#         if col in uncertainty_cols:
#             # treat col as uncertainty => confidence = 1 - scaled
#             scaled = 1.0 - scaled
#         X_list.append(scaled)
#     # shape => (num_samples, num_features)
#     X_all = np.column_stack(X_list)

#     # 2) Accuracy vector
#     y_all = df[accuracy_col].values  # shape (N, ), e.g. 0/1 or float in [0,1]

#     # 3) Split: calibration vs test
#     X_cal, X_test, y_cal, y_test = train_test_split(
#         X_all, y_all,
#         test_size=(1 - val_percent),
#         random_state=42,
#         shuffle=True
#     )

#     # 4) Polynomial + Ridge pipeline
#     pipe = make_pipeline(
#         PolynomialFeatures(degree=degree),
#         Ridge(alpha=alpha)
#     )
#     pipe.fit(X_cal, y_cal)

#     # 5) Predict on the test set => raw predictions
#     y_pred_test = pipe.predict(X_test)

#     # 6) Min-max scale these predicted values to interpret as confidence
#     conf_test_scaled = min_max_scale(y_pred_test)

#     # 7) Plot reliability diagram
#     plot_reliability_diagram(
#         confidences=conf_test_scaled,
#         accuracies=y_test,
#         confidence_type=confidence_label,
#         save_path=save_path
#     )

#     return pipe

# # Let's say you have a DataFrame df with columns:
# #  - "accuracy"         (0 or 1) or maybe a float in [0..1]
# #  - "predictive_entropy_min_max_scaled"
# #  - "semantic_entropy_min_max_scaled"
# #  - "grounding_biomedclip_processed_scaled"

# # 'accuracy', 'grounding_biomedclip', 'grounding_biomedclip_processed',
# #        'grounding_llama32_11b', 'grounding_llama32_11b_processed',
# #        'grounding_llama32_70b', 'grounding_llama32_70b_processed',
# #        'grounding_qwen_vl', 'grounding_qwen_vl_processed',
# #        'max_grounding_biomedclip', 'predictive_entropy_min_max_scaled',
# #        'lexical_similarity_min_max_scaled', 'semantic_entropy_min_max_scaled',
# #        'num_clusters_min_max_scaled', 'grounding_biomedclip_processed_scaled',
# #        'grounding_biomedclip_processed_scaled_average',
# #        'grounding_llama32_11b_processed_score',
# #        'grounding_llama32_70b_processed_score',
# #        'grounding_qwen_vl_processed_score',
# #        'grounding_llama32_11b_llama32_70b_agreement',
# #        'grounding_llama32_11b_qwen_vl_agreement',
# #        'grounding_llama32_70b_qwen_vl_agreement',
# #        'grounding_models_entropy_llama11b_llama70b_qwenvl',
# #        'grounding_models_entropy_llama11b_llama70b',
# #        'grounding_models_entropy_llama11b_qwenvl',
# #        'grounding_models_entropy_llama70b_qwenvl'],

# # what are the combinations of features you want to use for calibration?
# # Baselines:
# # 1) predictive_entropy_min_max_scaled
# # 2) lexical_similarity_min_max_scaled
# # 3) semantic_entropy_min_max_scaled
# # 4) num_clusters_min_max_scaled

# # Now lets add grounding 
# # 6) grounding_biomedclip_processed_scaled_average
# # 7) 'grounding_llama32_11b_processed_score',
# # 8) 'grounding_llama32_70b_processed_score',
# # 9) 'grounding_qwen_vl_processed_score'

# # Now lets do baselines + grounding 
# # 10) predictive_entropy_min_max_scales + grounding_biomedclip_processed_scaled_average
# # 11) predictive_entropy_min_max_scales + grounding_biomedclip_processed_scaled_average + grounding_llama32_11b_processed_score 
# # 12) predictive_entropy_min_max_scales + grounding_biomedclip_processed_scaled_average + grounding_llama32_11b_processed_score + grounding_llama32_70b_processed_score
# # 13) predictive_entropy_min_max_scales + grounding_biomedclip_processed_scaled_average + grounding_llama32_11b_processed_score + grounding_llama32_70b_processed_score + grounding_qwen_vl_processed_score 
# # 14) lexical_similarity_min_max_scaled + grounding_biomedclip_processed_scaled_average
# # 15) lexical_similarity_min_max_scaled + grounding_biomedclip_processed_scaled_average + grounding_llama32_11b_processed_score
# # 16) lexical_similarity_min_max_scaled + grounding_biomedclip_processed_scaled_average + grounding_llama32_11b_processed_score + grounding_llama32_70b_processed_score
# # 17) lexical_similarity_min_max_scaled + grounding_biomedclip_processed_scaled_average + grounding_llama32_11b_processed_score + grounding_llama32_70b_processed_score + grounding_qwen_vl_processed_score
# # 18) semantic_entropy_min_max_scaled + grounding_biomedclip_processed_scaled_average
# # 19) semantic_entropy_min_max_scaled + grounding_biomedclip_processed_scaled_average + grounding_llama32_11b_processed_score
# # 20) semantic_entropy_min_max_scaled + grounding_biomedclip_processed_scaled_average + grounding_llama32_11b_processed_score + grounding_llama32_70b_processed_score
# # 21) semantic_entropy_min_max_scaled + grounding_biomedclip_processed_scaled_average + grounding_llama32_11b_processed_score + grounding_llama32_70b_processed_score + grounding_qwen_vl_processed_score
# # 22) num_clusters_min_max_scaled + grounding_biomedclip_processed_scaled_average
# # 23) num_clusters_min_max_scaled + grounding_biomedclip_processed_scaled_average + grounding_llama32_11b_processed_score
# # 24) num_clusters_min_max_scaled + grounding_biomedclip_processed_scaled_average + grounding_llama32_11b_processed_score + grounding_llama32_70b_processed_score
# # 25) num_clusters_min_max_scaled + grounding_biomedclip_processed_scaled_average + grounding_llama32_11b_processed_score + grounding_llama32_70b_processed_score + grounding_qwen_vl_processed_score

# # Now lets use the agreement scores
# #        'grounding_llama32_11b_llama32_70b_agreement',
# #        'grounding_llama32_11b_qwen_vl_agreement',
# #        'grounding_llama32_70b_qwen_vl_agreement',
# # 26) predictive_entropy_min_max_scales + grounding_llama32_11b_llama32_70b_agreement
# # 27) lexical_similarity_min_max_scaled + grounding_llama32_11b_llama32_70b_agreement
# # 28) semantic_entropy_min_max_scaled + grounding_llama32_11b_llama32_70b_agreement
# # 29) num_clusters_min_max_scaled + grounding_llama32_11b_llama32_70b_agreement

# # Now lets use the entropy scores
# #        'grounding_models_entropy_llama11b_llama70b_qwenvl',
# #        'grounding_models_entropy_llama11b_llama70b',
# #        'grounding_models_entropy_llama11b_qwenvl',
# #        'grounding_models_entropy_llama70b_qwenvl'

# # 30) predictive_entropy_min_max_scales + grounding_models_entropy_llama11b_llama70b_qwenvl
# # 31) predictive_entropy_min_max_scales + grounding_models_entropy_llama11b_llama70b
# # 32) lexical_similarity_min_max_scaled + grounding_models_entropy_llama11b_llama70b_qwenvl 
# # 33) lexical_similarity_min_max_scaled + grounding_models_entropy_llama11b_llama70b
# # 34) semantic_entropy_min_max_scaled + grounding_models_entropy_llama11b_llama70b_qwenvl
# # 35) semantic_entropy_min_max_scaled + grounding_models_entropy_llama11b_llama70b
# # 36) num_clusters_min_max_scaled + grounding_models_entropy_llama11b_llama70b_qwenvl
# # 37) num_clusters_min_max_scaled + grounding_models_entropy_llama11b_llama70b

# # now lets do for entropy of the grounding models alone 
# # 38) grounding_models_entropy_llama11b_llama70b_qwenvl
# # 39) grounding_models_entropy_llama11b_llama70b
# # 40) grounding_models_entropy_llama11b_qwenvl
# # 41) grounding_models_entropy_llama70b_qwenvl


# # # Example: use 2 features (predictive entropy + grounding)
# # model_2d = calibrate_and_plot_reliability(
# #     df=df,
# #     feature_cols=["predictive_entropy_min_max_scaled", "grounding_biomedclip_processed_scaled_average"],
# #     accuracy_col="accuracy",
# #     uncertainty_cols=["predictive_entropy_min_max_scaled"],
# #     val_percent=0.20,   # 20% used for calibration
# #     degree=2,
# #     alpha=3.0,
# #     save_path="2d_calibrated_reliability.png",
# #     confidence_label="2D Calibrated"
# # )

# # # Example: use 3 features (predictive entropy + lexical similarity + semantic entropy)
# # model_3d = calibrate_and_plot_reliability(
# #     df=df,
# #     feature_cols=["predictive_entropy_min_max_scaled", 
# #                   "lexical_similarity_min_max_scaled",
# #                   "semantic_entropy_min_max_scaled"],
# #     accuracy_col="accuracy",
# #     uncertainty_cols=["predictive_entropy_min_max_scaled", "semantic_entropy_min_max_scaled"],
# #     val_percent=0.20,
# #     degree=2,
# #     alpha=3.0,
# #     save_path="3d_calibrated_reliability.png",
# #     confidence_label="3D Calibrated"
# # )

# def run_all_41_calibrations(df, output_prefix="calib_run", degree=2, alpha=3.0, val_percent=0.20):
#     """
#     Runs 41 calibration plots using `calibrate_and_plot_reliability`
#     for each enumerated combination of features.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         DataFrame containing columns:
#           'accuracy', 'predictive_entropy_min_max_scaled', 
#           'lexical_similarity_min_max_scaled', 'semantic_entropy_min_max_scaled',
#           'num_clusters_min_max_scaled', 'grounding_biomedclip_processed_scaled_average',
#           'grounding_llama32_11b_processed_score', 'grounding_llama32_70b_processed_score',
#           'grounding_qwen_vl_processed_score',
#           'grounding_llama32_11b_llama32_70b_agreement', 
#           'grounding_llama32_11b_qwen_vl_agreement',
#           'grounding_llama32_70b_qwen_vl_agreement',
#           'grounding_models_entropy_llama11b_llama70b_qwenvl',
#           'grounding_models_entropy_llama11b_llama70b',
#           'grounding_models_entropy_llama11b_qwenvl',
#           'grounding_models_entropy_llama70b_qwenvl', etc.

#     output_prefix : str
#         Prefix for saved plot filenames (e.g., "calib_run_1_...png").
#     degree : int
#         PolynomialFeatures degree for calibration.
#     alpha : float
#         Ridge regularization.
#     val_percent : float
#         Fraction of data used for calibration (rest for testing).
#     """

#     # A small helper to auto-detect uncertainty columns (contains 'entropy' in name):
#     def detect_uncertainty_cols(features):
#         return [col for col in features if "entropy" in col.lower()]
    
#     # --- List out the 41 combos exactly as enumerated ---
#     combos = [
#         # (combo_id, [feature_cols...])
#         (1,  ["predictive_entropy_min_max_scaled"]),
#         (2,  ["lexical_similarity_min_max_scaled"]),
#         (3,  ["semantic_entropy_min_max_scaled"]),
#         (4,  ["num_clusters_min_max_scaled"]),

#         (6,  ["grounding_biomedclip_processed_scaled_average"]),
#         (7,  ["grounding_llama32_11b_processed_score"]),
#         (8,  ["grounding_llama32_70b_processed_score"]),
#         (9,  ["grounding_qwen_vl_processed_score"]),

#         (10, ["predictive_entropy_min_max_scaled", "grounding_biomedclip_processed_scaled_average"]),
#         (11, ["predictive_entropy_min_max_scaled", "grounding_biomedclip_processed_scaled_average",
#               "grounding_llama32_11b_processed_score"]),
#         (12, ["predictive_entropy_min_max_scaled", "grounding_biomedclip_processed_scaled_average",
#               "grounding_llama32_11b_processed_score", "grounding_llama32_70b_processed_score"]),
#         (13, ["predictive_entropy_min_max_scaled", "grounding_biomedclip_processed_scaled_average",
#               "grounding_llama32_11b_processed_score", "grounding_llama32_70b_processed_score",
#               "grounding_qwen_vl_processed_score"]),

#         (14, ["lexical_similarity_min_max_scaled", "grounding_biomedclip_processed_scaled_average"]),
#         (15, ["lexical_similarity_min_max_scaled", "grounding_biomedclip_processed_scaled_average",
#               "grounding_llama32_11b_processed_score"]),
#         (16, ["lexical_similarity_min_max_scaled", "grounding_biomedclip_processed_scaled_average",
#               "grounding_llama32_11b_processed_score", "grounding_llama32_70b_processed_score"]),
#         (17, ["lexical_similarity_min_max_scaled", "grounding_biomedclip_processed_scaled_average",
#               "grounding_llama32_11b_processed_score", "grounding_llama32_70b_processed_score",
#               "grounding_qwen_vl_processed_score"]),

#         (18, ["semantic_entropy_min_max_scaled", "grounding_biomedclip_processed_scaled_average"]),
#         (19, ["semantic_entropy_min_max_scaled", "grounding_biomedclip_processed_scaled_average",
#               "grounding_llama32_11b_processed_score"]),
#         (20, ["semantic_entropy_min_max_scaled", "grounding_biomedclip_processed_scaled_average",
#               "grounding_llama32_11b_processed_score", "grounding_llama32_70b_processed_score"]),
#         (21, ["semantic_entropy_min_max_scaled", "grounding_biomedclip_processed_scaled_average",
#               "grounding_llama32_11b_processed_score", "grounding_llama32_70b_processed_score",
#               "grounding_qwen_vl_processed_score"]),

#         (22, ["num_clusters_min_max_scaled", "grounding_biomedclip_processed_scaled_average"]),
#         (23, ["num_clusters_min_max_scaled", "grounding_biomedclip_processed_scaled_average",
#               "grounding_llama32_11b_processed_score"]),
#         (24, ["num_clusters_min_max_scaled", "grounding_biomedclip_processed_scaled_average",
#               "grounding_llama32_11b_processed_score", "grounding_llama32_70b_processed_score"]),
#         (25, ["num_clusters_min_max_scaled", "grounding_biomedclip_processed_scaled_average",
#               "grounding_llama32_11b_processed_score", "grounding_llama32_70b_processed_score",
#               "grounding_qwen_vl_processed_score"]),

#         (26, ["predictive_entropy_min_max_scaled", "grounding_llama32_11b_llama32_70b_agreement"]),
#         (27, ["lexical_similarity_min_max_scaled", "grounding_llama32_11b_llama32_70b_agreement"]),
#         (28, ["semantic_entropy_min_max_scaled", "grounding_llama32_11b_llama32_70b_agreement"]),
#         (29, ["num_clusters_min_max_scaled", "grounding_llama32_11b_llama32_70b_agreement"]),

#         (30, ["predictive_entropy_min_max_scaled", "grounding_models_entropy_llama11b_llama70b_qwenvl"]),
#         (31, ["predictive_entropy_min_max_scaled", "grounding_models_entropy_llama11b_llama70b"]),
#         (32, ["lexical_similarity_min_max_scaled", "grounding_models_entropy_llama11b_llama70b_qwenvl"]),
#         (33, ["lexical_similarity_min_max_scaled", "grounding_models_entropy_llama11b_llama70b"]),
#         (34, ["semantic_entropy_min_max_scaled", "grounding_models_entropy_llama11b_llama70b_qwenvl"]),
#         (35, ["semantic_entropy_min_max_scaled", "grounding_models_entropy_llama11b_llama70b"]),
#         (36, ["num_clusters_min_max_scaled", "grounding_models_entropy_llama11b_llama70b_qwenvl"]),
#         (37, ["num_clusters_min_max_scaled", "grounding_models_entropy_llama11b_llama70b"]),

#         (38, ["grounding_models_entropy_llama11b_llama70b_qwenvl"]),
#         (39, ["grounding_models_entropy_llama11b_llama70b"]),
#         (40, ["grounding_models_entropy_llama11b_qwenvl"]),
#         (41, ["grounding_models_entropy_llama70b_qwenvl"])
#     ]

#     # We assume 'accuracy' column is the target accuracy
#     accuracy_col = "accuracy"

#     # We'll import the calibrate_and_plot_reliability function from previous code
#     # (Ensure it's in the same file or imported properly)
#     # from your_module import calibrate_and_plot_reliability

#     # Now run each combo
#     for combo_id, feat_cols in combos:
#         # Detect which columns are "uncertainty" (contain 'entropy' in name)
#         uncty_cols = detect_uncertainty_cols(feat_cols)

#         # Build a safe filename
#         # e.g. "calib_run_1_predictive_entropy_min_max_scaled.png"
#         features_str = "_".join(feat_cols)
#         save_path = f"{output_prefix}_{combo_id}_{features_str}.png"
#         features_str_name = features_str.replace("min_max_scaled", " ")
#         features_str_name = features_str_name.replace("processed", " ")
#         features_str_name = features_str_name.replace("grounding", " ")
#         # features_str_name = features_str_name.replace("_", "")
#         label_str = f"{features_str_name}"

#         print(f"\n[INFO] Running combo #{combo_id} with features = {feat_cols}")
#         print(f"[INFO] uncertainty_cols = {uncty_cols}")
#         print(f"[INFO] Output plot => {save_path}")

#         # Call the calibration plot function
#         calibrate_and_plot_reliability(
#             df=df,
#             feature_cols=feat_cols,
#             accuracy_col=accuracy_col,
#             uncertainty_cols=uncty_cols,
#             val_percent=val_percent,
#             degree=degree,
#             alpha=alpha,
#             save_path=save_path,
#             confidence_label=label_str
#         )

#     print("\n[INFO] All 41 calibrations have been generated.")
    
    
# # Run all 41 calibrations
# # run_all_41_calibrations(df, output_prefix="calib_run", degree=2, alpha=3.0, val_percent=0.20)

# # now lets see what all we have to do
# # 1) predictive_entropy_min_max_scaled
# # semantic_entropy_min_max_scaled
# # num_clusters_min_max_scaled
# # num_clusters_min_max_scaled
# # 2) predictive_entropy_min_max_scaled + grounding_llama32_70b_processed_score + entropy_grounding_llama32_70b_processed
# # 3) semantic_entropy_min_max_scaled + grounding_llama32_70b_processed_score + entropy_grounding_llama32_70b_processed
# # 4) num_clusters_min_max_scaled + grounding_llama32_70b_processed_score + entropy_grounding_llama32_70b_processed
# # 5) lexical_similarity_min_max_scaled + grounding_llama32_70b_processed_score + entropy_grounding_llama32_70b_processed

# # 6) predictive_entropy_min_max_scaled + grounding_llama32_70b_processed_score + grounding_models_entropy_llama11b_llama70b_qwenvl
# # 7) semantic_entropy_min_max_scaled + grounding_llama32_70b_processed_score + grounding_models_entropy_llama11b_llama70b_qwenvl
# # 8) num_clusters_min_max_scaled + grounding_llama32_70b_processed_score + grounding_models_entropy_llama11b_llama70b_qwenvl
# # 9) lexical_similarity_min_max_scaled + grounding_llama32_70b_processed_score + grounding_models_entropy_llama11b_llama70b_q
# # 10) predictive_entropy_min_max_scaled + grounding_llama32_70b_processed_score + grounding_models_entropy_llama11b_llama70b


# # model_2d = calibrate_and_plot_reliability(
# #     df=df,
# #     feature_cols=["predictive_entropy_min_max_scaled", "grounding_biomedclip_processed_scaled_average"],
# #     accuracy_col="accuracy",
# #     uncertainty_cols=["predictive_entropy_min_max_scaled"],
# #     val_percent=0.20,   # 20% used for calibration
# #     degree=2,
# #     alpha=3.0,
# #     save_path="2d_calibrated_reliability.png",
# #     confidence_label="2D Calibrated"
# # )


# # #############################################
# # # Helper to detect which columns are 'uncertainty'
# # #############################################
# # def get_uncertainty_cols(features):
# #     """
# #     Return a list of features that contain 'entropy' in their name,
# #     ignoring case.
# #     """
# #     return [col for col in features if "entropy" in col.lower()]


# # #############################################
# # # 1) Single feature: predictive_entropy_min_max_scaled
# # #############################################
# # combo_1_features = ["predictive_entropy_min_max_scaled"]
# # combo_1_uncert = get_uncertainty_cols(combo_1_features)

# # model_1 = calibrate_and_plot_reliability(
# #     df=df,
# #     feature_cols=combo_1_features,
# #     accuracy_col="accuracy",
# #     uncertainty_cols=combo_1_uncert,
# #     val_percent=0.20,   
# #     degree=2,
# #     alpha=3.0,
# #     save_path="combo_1_predEntropy.png",
# #     confidence_label="Combo 1"
# # )

# def get_uncertainty_cols(features):
#     """
#     Return a list of features that contain 'entropy' in their name, ignoring case.
#     """
#     return [col for col in features if "entropy" in col.lower()]

# #############################################
# # 2) predictive_entropy_min_max_scaled + grounding_llama32V_with_external_entropy
# #############################################
# # baseline 

# features = [
#     "predictive_entropy_min_max_scaled",
#     # "grounding_llama32V_with_external_entropy",
# ]
# # combo_2_uncert = get_uncertainty_cols(combo_2_features)

# model_2 = calibrate_and_plot_reliability(
#     df=df,
#     feature_cols=features,
#     accuracy_col="accuracy",
#     uncertainty_cols= ["predictive_entropy_min_max_scaled"],
#     val_percent=0.20,
#     degree=2,
#     alpha=3.0,
#     save_path="14_feb_pred_entropy_baseline.png",
#     confidence_label=" pred entropy "
# )

# # pred entropy + grounding llama32V with external entropy
# features = ["predictive_entropy_min_max_scaled", "grounding_llama32V_with_external_entropy"]

# model_2 = calibrate_and_plot_reliability(
#     df=df,
#     feature_cols=features,
#     accuracy_col="accuracy",
#     uncertainty_cols=["predictive_entropy_min_max_scaled"],
#     val_percent=0.20,
#     degree=2,
#     alpha=3.0,
#     save_path="14_feb_pred_entropy_with_external_gr.png",
#     confidence_label=" pred entropy with external gr"
# )

# features = [
#     "semantic_entropy_min_max_scaled",
#     # "grounding_llama32V_with_external_entropy",
# ]
# # combo_2_uncert = get_uncertainty_cols(combo_2_features)

# model_2 = calibrate_and_plot_reliability(
#     df=df,
#     feature_cols=features,
#     accuracy_col="accuracy",
#     uncertainty_cols= ["semantic_entropy_min_max_scaled"],
#     val_percent=0.20,
#     degree=2,
#     alpha=3.0,
#     save_path="14_feb_pred_entropy_baseline.png",
#     confidence_label=" pred entropy "
# )

# #############################################
# # 3) semantic_entropy_min_max_scaled + grounding_llama32_70b_processed_score + entropy_grounding_llama32_70b_processed
# #############################################
# features = ["semantic_entropy_min_max_scaled", "grounding_llama32V_with_external_entropy"]


# model_3 = calibrate_and_plot_reliability(
#     df=df,
#     feature_cols=features,
#     accuracy_col="accuracy",
#     uncertainty_cols=['semantic_entropy_min_max_scaled'],
#     val_percent=0.20,
#     degree=2,
#     alpha=3.0,
#     save_path="14_feb_se_entropy_with_external_gr.png",
#     confidence_label=" semantic entropy + grounding llama32V with external entropy"
# )

# # lexical similarity
# features = [
#     "lexical_similarity_min_max_scaled" ] 

# model_3 = calibrate_and_plot_reliability(
#     df=df,
#     feature_cols=features,
#     accuracy_col="accuracy",
#     val_percent=0.20,
#     degree=2,
#     alpha=3.0,
#     save_path="14_feb_lexical_similarity.png",
#     confidence_label=" lexical similarity"
# )

# # lexical similarity + grounding llama32V with external entropy
# features = [
#     "lexical_similarity_min_max_scaled",
#     "grounding_llama32V_with_external_entropy"
# ]

# model_3 = calibrate_and_plot_reliability(
#     df=df,
#     feature_cols=features,
#     accuracy_col="accuracy",
#     # uncertainty_cols=['lexical_similarity_min_max_scaled'],
#     val_percent=0.20,
#     degree=2,
#     alpha=3.0,
#     save_path="14_feb_lexical_similarity_with_external_gr.png",
#     confidence_label=" lexical similarity + grounding llama32V with external entropy"
# )

# # num clusters
# features = [
#     "num_clusters_min_max_scaled" ]

# model_3 = calibrate_and_plot_reliability(
#     df=df,
#     feature_cols=features,
#     accuracy_col="accuracy",
#     val_percent=0.20,
#     degree=2,
#     alpha=3.0,
#     save_path="14_feb_num_clusters.png",
#     confidence_label=" num clusters"
# )

# # num clusters + grounding llama32V with external entropy
# features = [
#     "num_clusters_min_max_scaled",
#     "grounding_llama32V_with_external_entropy"
# ]

# model_3 = calibrate_and_plot_reliability(
#     df=df,
#     feature_cols=features,
#     accuracy_col="accuracy",
#     # uncertainty_cols=['num_clusters_min_max_scaled'],
#     val_percent=0.20,
#     degree=2,
#     alpha=3.0,
#     save_path="14_feb_num_clusters_with_external_gr.png",
#     confidence_label=" num clusters + grounding llama32V with external entropy"
# )




# # #############################################
# # # 4) num_clusters_min_max_scaled + grounding_llama32V_with_external_entropy 
# # #############################################
# # combo_4_features = [
# #     "num_clusters_min_max_scaled",
# #     "grounding_llama32V_with_external_entropy",
# # ]
# # combo_4_uncert = get_uncertainty_cols(combo_4_features)

# # model_4 = calibrate_and_plot_reliability(
# #     df=df,
# #     feature_cols=combo_4_features,
# #     accuracy_col="accuracy",
# #     uncertainty_cols=combo_4_uncert,
# #     val_percent=0.20,
# #     degree=2,
# #     alpha=3.0,
# #     save_path="combo_4_numClust_ground70b_entropy_external.png",
# #     confidence_label=" num clusters + grounding llama32V with external entropy"
# # )


# # #############################################
# # # 5) lexical_similarity_min_max_scaled + grounding_llama32V_with_external_entropy + entropy_grounding_llama32_70b_processed
# # #############################################
# # combo_5_features = [
# #     "lexical_similarity_min_max_scaled",
# #     "grounding_llama32V_with_external_entropy",
# #     # "entropy_grounding_llama32_70b_processed"
# # ]
# # combo_5_uncert = get_uncertainty_cols(combo_5_features)

# # model_5 = calibrate_and_plot_reliability(
# #     df=df,
# #     feature_cols=combo_5_features,
# #     accuracy_col="accuracy",
# #     uncertainty_cols=combo_5_uncert,
# #     val_percent=0.20,
# #     degree=2,
# #     alpha=3.0,
# #     save_path="combo_5_lexSim_ground70b_entropy_external.png",
# #     confidence_label=" lex sim + grounding llama32V with external entropy"
# # )


# # #############################################
# # # 6) predictive_entropy_min_max_scaled + grounding_llama32_70b_processed_score + grounding_models_entropy_llama11b_llama70b_qwenvl
# # #############################################
# # combo_6_features = [
# #     "predictive_entropy_min_max_scaled",
# #     "grounding_llama32_70b_processed_score",
# #     "grounding_models_entropy_llama11b_llama70b_qwenvl"
# # ]
# # combo_6_uncert = get_uncertainty_cols(combo_6_features)

# # model_6 = calibrate_and_plot_reliability(
# #     df=df,
# #     feature_cols=combo_6_features,
# #     accuracy_col="accuracy",
# #     uncertainty_cols=combo_6_uncert,
# #     val_percent=0.20,
# #     degree=2,
# #     alpha=3.0,
# #     save_path="combo_6_predEnt_ground70b_llamaEntQwn.png",
# #     confidence_label="Combo 6"
# # )


# # #############################################
# # # 7) semantic_entropy_min_max_scaled + grounding_llama32_70b_processed_score + grounding_models_entropy_llama11b_llama70b_qwenvl
# # #############################################
# # combo_7_features = [
# #     "semantic_entropy_min_max_scaled",
# #     "grounding_llama32_70b_processed_score",
# #     "grounding_models_entropy_llama11b_llama70b_qwenvl"
# # ]
# # combo_7_uncert = get_uncertainty_cols(combo_7_features)

# # model_7 = calibrate_and_plot_reliability(
# #     df=df,
# #     feature_cols=combo_7_features,
# #     accuracy_col="accuracy",
# #     uncertainty_cols=combo_7_uncert,
# #     val_percent=0.20,
# #     degree=2,
# #     alpha=3.0,
# #     save_path="combo_7_semEnt_ground70b_llamaEntQwn.png",
# #     confidence_label="Combo 7"
# # )


# # #############################################
# # # 8) num_clusters_min_max_scaled + grounding_llama32_70b_processed_score + grounding_models_entropy_llama11b_llama70b_qwenvl
# # #############################################
# # combo_8_features = [
# #     "num_clusters_min_max_scaled",
# #     "grounding_llama32_70b_processed_score",
# #     "grounding_models_entropy_llama11b_llama70b_qwenvl"
# # ]
# # combo_8_uncert = get_uncertainty_cols(combo_8_features)

# # model_8 = calibrate_and_plot_reliability(
# #     df=df,
# #     feature_cols=combo_8_features,
# #     accuracy_col="accuracy",
# #     uncertainty_cols=combo_8_uncert,
# #     val_percent=0.20,
# #     degree=2,
# #     alpha=3.0,
# #     save_path="combo_8_numClusters_ground70b_llamaEntQwn.png",
# #     confidence_label="Combo 8"
# # )


# # #############################################
# # # 9) lexical_similarity_min_max_scaled + grounding_llama32_70b_processed_score + grounding_models_entropy_llama11b_llama70b_q
# # #    (Assuming you meant "grounding_models_entropy_llama11b_llama70b_q" or "qwenvl"? 
# # #     Adjust column name as needed.)
# # #############################################
# # combo_9_features = [
# #     "lexical_similarity_min_max_scaled",
# #     "grounding_llama32_70b_processed_score",
# #     "grounding_models_entropy_llama11b_llama70b_qwenvl"   # or _qwenvl if correct column
# # ]
# # combo_9_uncert = get_uncertainty_cols(combo_9_features)

# # model_9 = calibrate_and_plot_reliability(
# #     df=df,
# #     feature_cols=combo_9_features,
# #     accuracy_col="accuracy",
# #     uncertainty_cols=combo_9_uncert,
# #     val_percent=0.20,
# #     degree=2,
# #     alpha=3.0,
# #     save_path="combo_9_lexSim_ground70b_llamaEntQ.png",
# #     confidence_label="Combo 9"
# # )


# # #############################################
# # # 10) predictive_entropy_min_max_scaled + grounding_llama32_70b_processed_score + grounding_models_entropy_llama11b_llama70b
# # #############################################
# # combo_10_features = [
# #     "predictive_entropy_min_max_scaled",
# #     "grounding_llama32_70b_processed_score",
# #     "grounding_models_entropy_llama11b_llama70b"
# # ]
# # combo_10_uncert = get_uncertainty_cols(combo_10_features)

# # model_10 = calibrate_and_plot_reliability(
# #     df=df,
# #     feature_cols=combo_10_features,
# #     accuracy_col="accuracy",
# #     uncertainty_cols=combo_10_uncert,
# #     val_percent=0.20,
# #     degree=2,
# #     alpha=3.0,
# #     save_path="combo_10_predEnt_ground70b_llamaEnt.png",
# #     confidence_label="Combo 10"
# # )




# #############################################
# # Single feature: semantic_entropy_min_max_scaled
# #############################################
# # combo_s_features = ["semantic_entropy_min_max_scaled"]
# # combo_s_uncert = get_uncertainty_cols(combo_s_features)

# # model_sem = calibrate_and_plot_reliability(
# #     df=df,
# #     feature_cols=combo_s_features,
# #     accuracy_col="accuracy",
# #     uncertainty_cols=combo_s_uncert,
# #     val_percent=0.20,
# #     degree=2,
# #     alpha=3.0,
# #     save_path="combo_semantic_entropy.png",
# #     confidence_label="Semantic Entropy (Single Feature)"
# # )


# # #############################################
# # # Single feature: lexical_similarity_min_max_scaled (Run #1)
# # #############################################
# # combo_n1_features = ["lexical_similarity_min_max_scaled"]
# # combo_n1_uncert = get_uncertainty_cols(combo_n1_features)

# # model_nc1 = calibrate_and_plot_reliability(
# #     df=df,
# #     feature_cols=combo_n1_features,
# #     accuracy_col="accuracy",
# #     uncertainty_cols=combo_n1_uncert,
# #     val_percent=0.20,
# #     degree=2,
# #     alpha=3.0,
# #     save_path="combo_lexical_similarity_min_max_scaled_1.png",
# #     confidence_label="Lexical Smilarity"
# # )



# # #############################################
# # # Single feature: num_clusters_min_max_scaled (Run #2) 
# # # (If truly intended to repeat)
# # #############################################
# # combo_n2_features = ["num_clusters_min_max_scaled"]
# # combo_n2_uncert = get_uncertainty_cols(combo_n2_features)

# # model_nc2 = calibrate_and_plot_reliability(
# #     df=df,
# #     feature_cols=combo_n2_features,
# #     accuracy_col="accuracy",
# #     uncertainty_cols=combo_n2_uncert,
# #     val_percent=0.20,
# #     degree=2,
# #     alpha=3.0,
# #     save_path="combo_num_clusters_2.png",
# #     confidence_label="Num Clusters (Run #2)"
# # )