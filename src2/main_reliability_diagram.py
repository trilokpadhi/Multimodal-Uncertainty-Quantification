import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import random

# df = pd.read_csv('/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/my_outputs/merged_with_grounding_processed.csv')

df = pd.read_csv('/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/my_outputs/vqa/merged_with_grounding_processed_with_variance_march7.csv')

# Below are the columns in the dataframe:
# Index(['question_id', 'accuracy', 'predictive_entropy', 'lexical_similarity',
#        'semantic_entropy', 'num_clusters', 'responses', 'resp_log_probs',
#        'question', 'grounding_biomedclip', 'grounding_biomedclip_processed',
#        'grounding_llama32_11b', 'grounding_llama32_11b_processed',
#        'grounding_llama32_70b', 'grounding_llama32_70b_processed',
#        'grounding_qwen_vl', 'grounding_qwen_vl_processed',
#        'max_grounding_biomedclip', 'predictive_entropy_min_max_scaled',
#        'lexical_similarity_min_max_scaled', 'semantic_entropy_min_max_scaled',
#        'num_clusters_min_max_scaled', 'grounding_biomedclip_processed_scaled',
#        'grounding_biomedclip_processed_scaled_average',
#        'grounding_llama32_11b_processed_score',
#        'grounding_llama32_70b_processed_score',
#        'grounding_qwen_vl_processed_score',
#        'grounding_llama32_11b_llama32_70b_agreement',
#        'grounding_llama32_11b_qwen_vl_agreement',
#        'grounding_llama32_70b_qwen_vl_agreement',
#        'grounding_models_entropy_llama11b_llama70b_qwenvl',
#        'grounding_models_entropy_llama11b_llama70b',
#        'grounding_models_entropy_llama11b_qwenvl',
#        'grounding_models_entropy_llama70b_qwenvl'],
#       dtype='object')

import numpy as np
import matplotlib.pyplot as plt

# def plot_reliability_diagram(confidences, accuracies, confidence_type, save_path):
#     """
#     Plots a reliability diagram (confidence vs. accuracy) with vertical error bars 
#     indicating the variance (or standard deviation) of accuracies in each bin.
#     Saves the figure to 'save_path'.
#     """

#     plt.style.use('ggplot')
#     fig, ax = plt.subplots(figsize=(8, 6))

#     # Binning
#     bins = np.linspace(0, 1, 11)
#     bin_indices = np.digitize(confidences, bins) - 1

#     avg_accuracies = []
#     bin_centers = []
#     # For error bars: choose std or var
#     # (If you truly want variance, set 'use_variance=True')
#     use_variance = False
#     error_values = []

#     for i in range(len(bins) - 1):
#         mask = bin_indices == i
#         if mask.any():
#             bin_accuracies = accuracies[mask]
#             avg_acc = np.mean(bin_accuracies)
#             if use_variance:
#                 err_val = np.var(bin_accuracies)  # variance
#             else:
#                 err_val = np.std(bin_accuracies)  # standard deviation

#             avg_accuracies.append(avg_acc)
#             error_values.append(err_val)

#             bin_center = (bins[i] + bins[i+1]) / 2
#             bin_centers.append(bin_center)

#     avg_accuracies = np.array(avg_accuracies)
#     bin_centers = np.array(bin_centers)
#     error_values = np.array(error_values)

#     # Plot average accuracy (with error bars for variance or std)
#     # fmt='o-' means circle markers with a solid line connecting them
#     # capsize adds little caps on error bars
#     if use_variance:
#         label_text = 'Avg. Accuracy ± Variance'
#     else:
#         label_text = 'Avg. Accuracy ± Std'

#     lower_bounds = avg_accuracies - error_values
#     upper_bounds = avg_accuracies + error_values

#     # Clip them to [0, 1]
#     lower_bounds = np.clip(lower_bounds, 0, 1)
#     upper_bounds = np.clip(upper_bounds, 0, 1)

# # Now errorbar expects `yerr` to be how far *above* and *below* the mean.
#     yerr = np.array([
#         avg_accuracies - lower_bounds,
#         upper_bounds - avg_accuracies
#     ])

#     ax.errorbar(
#         bin_centers,
#         avg_accuracies,
#         yerr=yerr,
#         fmt='o-',
#         ecolor='black',
#         capsize=5,
#         linewidth=0.5,
#         label=label_text
#     )
#     # ax.errorbar(
#     #     bin_centers, 
#     #     avg_accuracies, 
#     #     yerr=error_values, 
#     #     fmt='o-', 
#     #     ecolor='black', 
#     #     capsize=5, 
#     #     linewidth=0.5, 
#     #     label=label_text
#     # )

#     # Plot the perfect calibration line
#     ax.plot([0,1],[0,1], '--', color='gray', label='Perfect Calibration')
#     # Increase the font size of the legend
#     ax.legend(loc='upper left', fontsize=20)

#     # Optionally fill the region between the perfect calibration and the measured accuracies
#     ax.fill_between(bin_centers, bin_centers, avg_accuracies, alpha=0.1)

#     ax.set_xlim([0,1])
#     ax.set_ylim([0,1])
#     ax.set_xlabel("Confidence", fontsize=20)
#     ax.set_ylabel("Accuracy", fontsize=20)
#     # ax.set_title(f"({confidence_type})")
#     ax.legend(loc='upper left')

#     # Calculate ECE + MCE
#     if len(bin_centers) > 0:
#         ece = np.mean(np.abs(avg_accuracies - bin_centers))
#         mce = np.max(np.abs(avg_accuracies - bin_centers))
#     else:
#         ece, mce = 0, 0

#     # Example text annotation (uncomment if you want it displayed on the plot)
#     stats_text = f"ECE: {ece:.3f}\nMCE: {mce:.3f}"
#     ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12, 
#             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close(fig)
#     print(f"[INFO] Saved reliability diagram => {save_path}")
    
    

def plot_reliability_diagram(confidences, accuracies, confidence_type, save_path):
    """
    reliability diagram, saves the figure to 'save_path'.
    """
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(8, 6))

    # Binning
    bins = np.linspace(0, 1, 11)
    bin_indices = np.digitize(confidences, bins) - 1

    avg_accuracies = []
    bin_centers = []
    for i in range(len(bins) - 1):
        mask = bin_indices == i
        if mask.any():
            avg_acc = np.mean(accuracies[mask])
            avg_accuracies.append(avg_acc)
            bin_center = (bins[i] + bins[i+1]) / 2
            bin_centers.append(bin_center)

    avg_accuracies = np.array(avg_accuracies)
    bin_centers = np.array(bin_centers)

    ax.plot(bin_centers, avg_accuracies, marker='o', linestyle='-', linewidth=2, label=f'Accuracy')
    ax.plot([0,1],[0,1], '--', color='gray', label='Perfect Calibration')

    # Adjust legend positions to avoid overlap
    # ax.legend(loc='lower right')
    # ax.legend(loc='upper left')
    # ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=12, frameon=True)
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12, frameon=True)

    ax.fill_between(bin_centers, bin_centers, avg_accuracies, alpha=0.1)
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_xlabel("Confidence", fontsize=20)
    ax.set_ylabel("Accuracy", fontsize=20)

    # ECE + MCE
    if len(bin_centers) > 0:
        ece = np.mean(np.abs(avg_accuracies - bin_centers))
        mce = np.max(np.abs(avg_accuracies - bin_centers))
    else:
        ece, mce = 0, 0
    stats_text = f"ECE: {ece:.3f}\nMCE: {mce:.3f}"
    stats_text = f"ECE: {ece:.3f}"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Saved reliability diagram => {save_path}")
    
    
def min_max_scale(array):
    arr = np.array(array, dtype=float)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-12:
        # Avoid dividing by zero if all values are the same
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)



def calibrate_and_plot_reliability(
    df,
    feature_cols,         # list of columns to use as features
    accuracy_col,         # e.g., "accuracy"
    uncertainty_cols=None,# which of these feature_cols are 'uncertainty' (apply 1 - scaled_value)
    val_percent=0.20,     # fraction for calibration set
    degree=2,
    alpha=3.0,
    save_path="reliability_diagram.png",
    confidence_label="Calibrated Model"
):
    """
    Fits a multi-feature calibration model (PolynomialFeatures + Ridge)
    using `val_percent` of data as calibration set. Then plots a reliability 
    diagram with the predicted confidence vs. actual accuracy on the test set.
    
    Steps:
      1. For each feature_col:
         - min-max scale the column.
         - if it's in uncertainty_cols, do "confidence = 1 - scaled_value".
      2. Split data into calibration vs. test sets.
      3. Fit polynomial + Ridge on calibration set.
      4. Predict on test set, then min-max scale predictions.
      5. Plot reliability diagram of predicted (scaled) confidence vs. accuracy.
    """
    if uncertainty_cols is None:
        uncertainty_cols = []

    # 1) Construct the feature matrix X_all
    #    For each column, scale it and possibly invert if it's an uncertainty metric.
    X_list = []
    for col in feature_cols:
        raw_values = df[col].values
        scaled = min_max_scale(raw_values)    # scaled in [0..1]
        if col in uncertainty_cols:
            # treat col as uncertainty => confidence = 1 - scaled
            scaled = 1.0 - scaled
        X_list.append(scaled)
    # shape => (num_samples, num_features)
    X_all = np.column_stack(X_list)

    # 2) Accuracy vector
    y_all = df[accuracy_col].values  # shape (N, ), e.g. 0/1 or float in [0,1]

    # 3) Split: calibration vs test
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_all, y_all,
        test_size=(1 - val_percent),
        random_state= 123,
        shuffle=True
    )

    # 4) Polynomial + Ridge pipeline
    pipe = make_pipeline(
        PolynomialFeatures(degree=degree),
        Ridge(alpha=alpha)
    )
    pipe.fit(X_cal, y_cal)

    # 5) Predict on the test set => raw predictions
    y_pred_test = pipe.predict(X_test)

    # 6) Min-max scale these predicted values to interpret as confidence
    conf_test_scaled = min_max_scale(y_pred_test)

    # 7) Plot reliability diagram
    plot_reliability_diagram(
        confidences=conf_test_scaled,
        accuracies=y_test,
        confidence_type=confidence_label,
        save_path=save_path
    )

    return pipe

features = [
    "predictive_entropy_min_max_scaled",
]
model_2 = calibrate_and_plot_reliability(
    df=df,
    feature_cols=features,
    accuracy_col="accuracy",
    uncertainty_cols= None,
    val_percent=0.20,
    degree=2,
    alpha=3.0,
    save_path="7_mar_pred_entropy.png",
    confidence_label=" pred entropy "
)

features = [
    "predictive_entropy_times_variance_times_grounding",
    # "grounding_llama32V_with_external_entropy",
]
# combo_2_uncert = get_uncertainty_cols(combo_2_features)

model_2 = calibrate_and_plot_reliability(
    df=df,
    feature_cols=features,
    accuracy_col="accuracy",
    uncertainty_cols= None,
    val_percent=0.20,
    degree=2,
    alpha=3.0,
    save_path="7_march_pred_entropy_variance_pe_gr.png",
    confidence_label=" pred entropy with variance "
)

# pred entropy + grounding llama32V with external entropy
# features = ["predictive_entropy_min_max_scaled", "grounding_llama32V_with_external_entropy"]
features =  ["semantic_entropy_times_variance_times_grounding"]

model_2 = calibrate_and_plot_reliability(
    df=df,
    feature_cols=features,
    accuracy_col="accuracy",
    uncertainty_cols=None,
    val_percent=0.20,
    degree=2,
    alpha=3.0,
    save_path="7mar_se_entropy_variance_se_gr.png",
    confidence_label=" se entropy with variance "
)

features = [
    "semantic_entropy_min_max_scaled",
]

model_2 = calibrate_and_plot_reliability(
    df=df,
    feature_cols=features,
    accuracy_col="accuracy",
    uncertainty_cols= ["semantic_entropy_min_max_scaled"],
    val_percent=0.20,
    degree=2,
    alpha=3.0,
    save_path="7march_se_entropy_baseline.png",
    confidence_label=" se entropy "
)

#############################################
# 3) semantic_entropy_min_max_scaled + grounding_llama32_70b_processed_score + entropy_grounding_llama32_70b_processed
#############################################
# features = ["semantic_entropy_min_max_scaled", "grounding_llama32V_with_external_entropy"]
features = ["lexical_similarity_times_variance_times_grounding"]


model_3 = calibrate_and_plot_reliability(
    df=df,
    feature_cols=features,
    accuracy_col="accuracy",
    uncertainty_cols=None,
    val_percent=0.20,
    degree=2,
    alpha=3.0,
    save_path="7march_ls_variance.png",
    confidence_label=" lc + variance gr "
)

# lexical similarity
features = [
    "lexical_similarity_min_max_scaled" ] 

model_3 = calibrate_and_plot_reliability(
    df=df,
    feature_cols=features,
    accuracy_col="accuracy",
    uncertainty_cols=None,
    val_percent=0.20,
    degree=2,
    alpha=3.0,
    save_path="7_march_lexical_similarity.png",
    confidence_label=" lexical similarity"
)


# num clusters
features = [
    "num_clusters_min_max_scaled" ]

model_3 = calibrate_and_plot_reliability(
    df=df,
    feature_cols=features,
    accuracy_col="accuracy",
    uncertainty_cols=None,
    val_percent=0.20,
    degree=2,
    alpha=3.0,
    save_path="7march_num_clusters.png",
    confidence_label=" num clusters"
)

# num clusters + grounding llama32V with external entropy
features = [
    # "num_clusters_min_max_scaled",
    # "grounding_llama32V_with_external_entropy"
    "num_clusters_times_variance_times_grounding"
]

model_3 = calibrate_and_plot_reliability(
    df=df,
    feature_cols=features,
    accuracy_col="accuracy",
    # uncertainty_cols=['num_clusters_min_max_scaled'],
    uncertainty_cols=None,
    val_percent=0.20,
    degree=2,
    alpha=3.0,
    save_path="7march_num_clusters_with_variance_gr.png",
    confidence_label=" num clusters + variance"
)

# Let's say you have a DataFrame df with columns:
#  - "accuracy"         (0 or 1) or maybe a float in [0..1]
#  - "predictive_entropy_min_max_scaled"
#  - "semantic_entropy_min_max_scaled"
#  - "grounding_biomedclip_processed_scaled"

# 'accuracy', 'grounding_biomedclip', 'grounding_biomedclip_processed',
#        'grounding_llama32_11b', 'grounding_llama32_11b_processed',
#        'grounding_llama32_70b', 'grounding_llama32_70b_processed',
#        'grounding_qwen_vl', 'grounding_qwen_vl_processed',
#        'max_grounding_biomedclip', 'predictive_entropy_min_max_scaled',
#        'lexical_similarity_min_max_scaled', 'semantic_entropy_min_max_scaled',
#        'num_clusters_min_max_scaled', 'grounding_biomedclip_processed_scaled',
#        'grounding_biomedclip_processed_scaled_average',
#        'grounding_llama32_11b_processed_score',
#        'grounding_llama32_70b_processed_score',
#        'grounding_qwen_vl_processed_score',
#        'grounding_llama32_11b_llama32_70b_agreement',
#        'grounding_llama32_11b_qwen_vl_agreement',
#        'grounding_llama32_70b_qwen_vl_agreement',
#        'grounding_models_entropy_llama11b_llama70b_qwenvl',
#        'grounding_models_entropy_llama11b_llama70b',
#        'grounding_models_entropy_llama11b_qwenvl',
#        'grounding_models_entropy_llama70b_qwenvl'],

# what are the combinations of features you want to use for calibration?
# Baselines:
# 1) predictive_entropy_min_max_scaled
# 2) lexical_similarity_min_max_scaled
# 3) semantic_entropy_min_max_scaled
# 4) num_clusters_min_max_scaled

# Now lets add grounding 
# 6) grounding_biomedclip_processed_scaled_average
# 7) 'grounding_llama32_11b_processed_score',
# 8) 'grounding_llama32_70b_processed_score',
# 9) 'grounding_qwen_vl_processed_score'

# Now lets do baselines + grounding 
# 10) predictive_entropy_min_max_scales + grounding_biomedclip_processed_scaled_average
# 11) predictive_entropy_min_max_scales + grounding_biomedclip_processed_scaled_average + grounding_llama32_11b_processed_score 
# 12) predictive_entropy_min_max_scales + grounding_biomedclip_processed_scaled_average + grounding_llama32_11b_processed_score + grounding_llama32_70b_processed_score
# 13) predictive_entropy_min_max_scales + grounding_biomedclip_processed_scaled_average + grounding_llama32_11b_processed_score + grounding_llama32_70b_processed_score + grounding_qwen_vl_processed_score 
# 14) lexical_similarity_min_max_scaled + grounding_biomedclip_processed_scaled_average
# 15) lexical_similarity_min_max_scaled + grounding_biomedclip_processed_scaled_average + grounding_llama32_11b_processed_score
# 16) lexical_similarity_min_max_scaled + grounding_biomedclip_processed_scaled_average + grounding_llama32_11b_processed_score + grounding_llama32_70b_processed_score
# 17) lexical_similarity_min_max_scaled + grounding_biomedclip_processed_scaled_average + grounding_llama32_11b_processed_score + grounding_llama32_70b_processed_score + grounding_qwen_vl_processed_score
# 18) semantic_entropy_min_max_scaled + grounding_biomedclip_processed_scaled_average
# 19) semantic_entropy_min_max_scaled + grounding_biomedclip_processed_scaled_average + grounding_llama32_11b_processed_score
# 20) semantic_entropy_min_max_scaled + grounding_biomedclip_processed_scaled_average + grounding_llama32_11b_processed_score + grounding_llama32_70b_processed_score
# 21) semantic_entropy_min_max_scaled + grounding_biomedclip_processed_scaled_average + grounding_llama32_11b_processed_score + grounding_llama32_70b_processed_score + grounding_qwen_vl_processed_score
# 22) num_clusters_min_max_scaled + grounding_biomedclip_processed_scaled_average
# 23) num_clusters_min_max_scaled + grounding_biomedclip_processed_scaled_average + grounding_llama32_11b_processed_score
# 24) num_clusters_min_max_scaled + grounding_biomedclip_processed_scaled_average + grounding_llama32_11b_processed_score + grounding_llama32_70b_processed_score
# 25) num_clusters_min_max_scaled + grounding_biomedclip_processed_scaled_average + grounding_llama32_11b_processed_score + grounding_llama32_70b_processed_score + grounding_qwen_vl_processed_score

# Now lets use the agreement scores
#        'grounding_llama32_11b_llama32_70b_agreement',
#        'grounding_llama32_11b_qwen_vl_agreement',
#        'grounding_llama32_70b_qwen_vl_agreement',
# 26) predictive_entropy_min_max_scales + grounding_llama32_11b_llama32_70b_agreement
# 27) lexical_similarity_min_max_scaled + grounding_llama32_11b_llama32_70b_agreement
# 28) semantic_entropy_min_max_scaled + grounding_llama32_11b_llama32_70b_agreement
# 29) num_clusters_min_max_scaled + grounding_llama32_11b_llama32_70b_agreement

# Now lets use the entropy scores
#        'grounding_models_entropy_llama11b_llama70b_qwenvl',
#        'grounding_models_entropy_llama11b_llama70b',
#        'grounding_models_entropy_llama11b_qwenvl',
#        'grounding_models_entropy_llama70b_qwenvl'

# 30) predictive_entropy_min_max_scales + grounding_models_entropy_llama11b_llama70b_qwenvl
# 31) predictive_entropy_min_max_scales + grounding_models_entropy_llama11b_llama70b
# 32) lexical_similarity_min_max_scaled + grounding_models_entropy_llama11b_llama70b_qwenvl 
# 33) lexical_similarity_min_max_scaled + grounding_models_entropy_llama11b_llama70b
# 34) semantic_entropy_min_max_scaled + grounding_models_entropy_llama11b_llama70b_qwenvl
# 35) semantic_entropy_min_max_scaled + grounding_models_entropy_llama11b_llama70b
# 36) num_clusters_min_max_scaled + grounding_models_entropy_llama11b_llama70b_qwenvl
# 37) num_clusters_min_max_scaled + grounding_models_entropy_llama11b_llama70b

# now lets do for entropy of the grounding models alone 
# 38) grounding_models_entropy_llama11b_llama70b_qwenvl
# 39) grounding_models_entropy_llama11b_llama70b
# 40) grounding_models_entropy_llama11b_qwenvl
# 41) grounding_models_entropy_llama70b_qwenvl





# # Example: use 2 features (predictive entropy + grounding)
# model_2d = calibrate_and_plot_reliability(
#     df=df,
#     feature_cols=["predictive_entropy_min_max_scaled", "grounding_biomedclip_processed_scaled_average"],
#     accuracy_col="accuracy",
#     uncertainty_cols=["predictive_entropy_min_max_scaled"],
#     val_percent=0.20,   # 20% used for calibration
#     degree=2,
#     alpha=3.0,
#     save_path="2d_calibrated_reliability.png",
#     confidence_label="2D Calibrated"
# )

# # Example: use 3 features (predictive entropy + lexical similarity + semantic entropy)
# model_3d = calibrate_and_plot_reliability(
#     df=df,
#     feature_cols=["predictive_entropy_min_max_scaled", 
#                   "lexical_similarity_min_max_scaled",
#                   "semantic_entropy_min_max_scaled"],
#     accuracy_col="accuracy",
#     uncertainty_cols=["predictive_entropy_min_max_scaled", "semantic_entropy_min_max_scaled"],
#     val_percent=0.20,
#     degree=2,
#     alpha=3.0,
#     save_path="3d_calibrated_reliability.png",
#     confidence_label="3D Calibrated"
# )

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
# combo_2_features = [
#     "predictive_entropy_min_max_scaled",
#     "grounding_llama32V_with_external_entropy",
# ]
# combo_2_uncert = get_uncertainty_cols(combo_2_features)

# model_2 = calibrate_and_plot_reliability(
#     df=df,
#     feature_cols=combo_2_features,
#     accuracy_col="accuracy",
#     uncertainty_cols=combo_2_uncert,
#     val_percent=0.20,
#     degree=2,
#     alpha=3.0,
#     save_path="combo_2_predEnt_ground70b_entropy_external.png",
#     confidence_label=" pred entropy + grounding llama32V with external entropy"
# )


# #############################################
# # 3) semantic_entropy_min_max_scaled + grounding_llama32_70b_processed_score + entropy_grounding_llama32_70b_processed
# #############################################
# combo_3_features = [
#     "semantic_entropy_min_max_scaled",
#     "grounding_llama32V_with_external_entropy"
# ]
# combo_3_uncert = get_uncertainty_cols(combo_3_features)

# model_3 = calibrate_and_plot_reliability(
#     df=df,
#     feature_cols=combo_3_features,
#     accuracy_col="accuracy",
#     uncertainty_cols=combo_3_uncert,
#     val_percent=0.20,
#     degree=2,
#     alpha=3.0,
#     save_path="combo_3_semEnt_ground70b_entropy_external.png",
#     confidence_label=" semantic entropy + grounding llama32V with external entropy"
# )


# #############################################
# # 4) num_clusters_min_max_scaled + grounding_llama32V_with_external_entropy 
# #############################################
# combo_4_features = [
#     "num_clusters_min_max_scaled",
#     "grounding_llama32V_with_external_entropy",
# ]
# combo_4_uncert = get_uncertainty_cols(combo_4_features)

# model_4 = calibrate_and_plot_reliability(
#     df=df,
#     feature_cols=combo_4_features,
#     accuracy_col="accuracy",
#     uncertainty_cols=combo_4_uncert,
#     val_percent=0.20,
#     degree=2,
#     alpha=3.0,
#     save_path="combo_4_numClust_ground70b_entropy_external.png",
#     confidence_label=" num clusters + grounding llama32V with external entropy"
# )


# #############################################
# # 5) lexical_similarity_min_max_scaled + grounding_llama32V_with_external_entropy + entropy_grounding_llama32_70b_processed
# #############################################
# combo_5_features = [
#     "lexical_similarity_min_max_scaled",
#     "grounding_llama32V_with_external_entropy",
#     # "entropy_grounding_llama32_70b_processed"
# ]
# combo_5_uncert = get_uncertainty_cols(combo_5_features)

# model_5 = calibrate_and_plot_reliability(
#     df=df,
#     feature_cols=combo_5_features,
#     accuracy_col="accuracy",
#     uncertainty_cols=combo_5_uncert,
#     val_percent=0.20,
#     degree=2,
#     alpha=3.0,
#     save_path="combo_5_lexSim_ground70b_entropy_external.png",
#     confidence_label=" lex sim + grounding llama32V with external entropy"
# )


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