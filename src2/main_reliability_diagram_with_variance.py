import pandas as pd
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import re  # for filename sanitization
import random
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

##########################################################################
# 0) LOAD YOUR DATA
##########################################################################
# df = pd.read_csv('/path/to/your.csv')  # <-- adjust as needed
df = pd.read_csv('/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/my_outputs/merged_with_grounding_processed_internal_g_entropy_feb14.csv')  # Adjust path as needed

# df = pd.read_csv('/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/my_outputs/merged_with_grounding_processed_external_g_entropy_feb_14.csv')  # Adjust path as needed 


##########################################################################
# 1) DEFINE UTILITIES
##########################################################################

def min_max_scale(array):
    arr = np.array(array, dtype=float)
    mn, mx = arr.min(), arr.max()
    if abs(mx - mn) < 1e-12:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def get_uncertainty_cols(feature_list):
    """Example: treat columns containing 'entropy' or 'clusters' as uncertainty."""
    unc = []
    for col in feature_list:
        if 'entropy' in col.lower() or 'cluster' in col.lower():
            unc.append(col)
    return unc

class ECELoss(nn.Module):
    def __init__(self, n_bins=15):
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, confidences, accuracies):
        ece = torch.zeros(1)
        acc_means, acc_stds = [], []
        conf_means, conf_stds = [], []
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = (confidences > bin_lower.item()) & (confidences <= bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                avg_acc = accuracies[in_bin].mean()
                std_acc = accuracies[in_bin].std()
                avg_conf = confidences[in_bin].mean()
                std_conf = confidences[in_bin].std()

                acc_means.append(avg_acc.item())
                acc_stds.append(std_acc.item())
                conf_means.append(avg_conf.item())
                conf_stds.append(std_conf.item())

                ece += torch.abs(avg_conf - avg_acc) * prop_in_bin
            else:
                # fill with zeros so arrays line up
                acc_means.append(0.0)
                acc_stds.append(0.0)
                conf_means.append((bin_lower.item() + bin_upper.item())/2)
                conf_stds.append(0.0)

        return ece.item(), acc_means, conf_means, acc_stds, conf_stds

def build_polynomial_calibrator(degree=2, alpha=3.0):
    return make_pipeline(
        PolynomialFeatures(degree=degree),
        Ridge(alpha=alpha)
    )

def run_calibration_5_seeds(
    df,
    feature_cols,
    accuracy_col="accuracy",
    uncertainty_cols=None,
    val_percent=0.20,
    degree=2,
    alpha=3.0,
    seeds=[0,1,2,3,4]
    # seeds = [42]
):
    """
    For each of 5 seeds:
      - 80/20 split => calibrate on 20%, test on 80%
      - Fit polynomial calibrator, predict on test => min-max scale => confidences
    Collect all test confidences & accuracies from the 5 seeds together.
    """
    # uncertainty_cols = get_uncertainty_cols(feature_cols)

    X_list = []
    for col in feature_cols:
        raw_vals = df[col].values
        scaled = min_max_scale(raw_vals)
        if uncertainty_cols is not None:
            if col in uncertainty_cols:
                scaled = 1.0 - scaled
        X_list.append(scaled)
    X_all = np.column_stack(X_list)

    y_all = df[accuracy_col].values

    all_confs = []
    all_accs  = []

    for seed in seeds:
        X_cal, X_test, y_cal, y_test = train_test_split(
            X_all, y_all,
            test_size=(1 - val_percent),
            random_state=seed,
            shuffle=True
        )
        pipe = build_polynomial_calibrator(degree=degree, alpha=alpha)
        pipe.fit(X_cal, y_cal)

        y_pred = pipe.predict(X_test)
        conf = min_max_scale(y_pred)

        all_confs.append(conf)
        all_accs.append(y_test)

    return np.concatenate(all_confs, axis=0), np.concatenate(all_accs, axis=0)

##########################################################################
# 2) PLOTTING FUNCTION
##########################################################################
def sanitize_filename(s):
    """Replace non-alphanumeric with underscores, to keep filenames safe."""
    return re.sub(r"[^\w\-]+", "_", s)

def plot_reliability_diagram(confidences, accuracies, features, degree, n_bins=15, save_path=None):
    """
    Bins (confidences, accuracies), plots:
      - Blue bars = mean accuracy
      - Red dot & vertical line = mean ± std
      - Orange line = x=y
      - ECE annotation
    """
    ece_loss = ECELoss(n_bins=n_bins)

    conf_tensor = torch.tensor(confidences, dtype=torch.float32)
    acc_tensor  = torch.tensor(accuracies,  dtype=torch.float32)

    ece_value, acc_bin_means, conf_bin_means, acc_bin_stds, conf_bin_stds = ece_loss(conf_tensor, acc_tensor)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1].numpy()
    bin_uppers = bin_boundaries[1:].numpy()
    bin_centers = 0.5*(bin_lowers + bin_uppers)
    width = bin_uppers[0] - bin_lowers[0]

    # Create a nice, explicit title
    feature_str = " + ".join(features)
    title = f"{feature_str}".replace("min_max_scaled", "")
    
    # remove _min_max_scaled from the feature names
    # title.replace("_min_max_scaled", "")
    
    # remove 

    plt.figure(figsize=(7, 5))
    plt.bar(
        x=bin_centers,
        height=acc_bin_means,
        width=width,
        color='blue',
        # alpha=0.6,
        align='center',
        label='Mean Accuracy per bin'
    )
    plt.errorbar(
        x=bin_centers,
        y=acc_bin_means,
        yerr=acc_bin_stds,
        fmt='o',
        color='red',
        ecolor='red',
        capsize=3,
        label='Accuracy ± std dev'
    )
    plt.plot([0,1],[0,1], color='orange', label='x=y')
    plt.text(0.05, 0.95, f"ECE = {ece_value:.4f}", transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.ylim([0,1])
    plt.xlim([0,1])
    plt.legend(loc='lower right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


##########################################################################
# 3) CREATE AND SAVE MULTIPLE PLOTS
##########################################################################
if __name__ == "__main__":
    # Baseline single-feature sets
    baselines = [
        ["predictive_entropy_min_max_scaled"],
        ["lexical_similarity_min_max_scaled"],
        ["semantic_entropy_min_max_scaled"],
        ["num_clusters_min_max_scaled"]
    ]

    # Improved: each baseline + "grounding_llama32V_with_cluster_wise_entropy"
    improved = [
        ["predictive_entropy_min_max_scaled", "grounding_llama32V_with_external_entropy"],
        ["lexical_similarity_min_max_scaled", "grounding_llama32V_with_external_entropy"],
        ["semantic_entropy_min_max_scaled", "grounding_llama32V_with_external_entropy"],
        ["num_clusters_min_max_scaled", "grounding_llama32V_with_external_entropy"]
    ]

    # Combine them all into one list, labeling them for clarity if you like:
    feature_sets = baselines + improved

    # We can produce multiple polynomial degrees:
    # degrees = [2, 3]  # for example
    degrees = [2]

    for feat_cols in feature_sets:
        for deg in degrees:
            # 1) Run calibration over 5 seeds => gather test confidences & accuracies
            if "predictive_entropy_min_max_scaled" in feat_cols:
                uncertainty_cols = ["predictive_entropy_min_max_scaled"]
            elif "semantic_entropy_min_max_scaled" in feat_cols:
                uncertainty_cols = ["semantic_entropy_min_max_scaled"]
                
            confs, accs = run_calibration_5_seeds(
                df=df,
                feature_cols=feat_cols,
                accuracy_col="accuracy",
                uncertainty_cols=None,
                val_percent=0.20,
                degree=deg,
                alpha=3.0,
                # seeds=[0,1,2,3,4]
                seeds = [random.randint(0, 100)]
            )

            # 2) Plot reliability
            feature_str = " + ".join(feat_cols)
            # create a safe filename
            # feature_str_for_file = sanitize_filename(feature_str)
            # out_name = f"reliability_{feature_str_for_file}_deg{deg}.png"
            out_name = f"feb_14_external_reliability_{feature_str}_deg{deg}.png"

            plot_reliability_diagram(
                confidences=confs,
                accuracies=accs,
                features=feat_cols,
                degree=deg,
                n_bins=10,
                save_path=out_name
            )

            print(f"[INFO] Saved => {out_name}")

# import pandas as pd
# import numpy as np
# import torch
# from torch import nn
# import matplotlib.pyplot as plt

# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import Ridge
# from sklearn.pipeline import make_pipeline
# from sklearn.model_selection import train_test_split

# #####################################################################
# # 0) LOAD DATA
# #####################################################################
# df = pd.read_csv('/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/my_outputs/merged_with_grounding_processed_internal_g_entropy_feb14.csv')  # Adjust path as needed


# #####################################################################
# # 1) DEFINE UTILITY FUNCTIONS
# #####################################################################

# def min_max_scale(array):
#     """
#     Min-max scale a 1D array into [0,1].
#     If all values are the same, returns all zeros (avoids div by zero).
#     """
#     arr = np.array(array, dtype=float)
#     mn, mx = arr.min(), arr.max()
#     if abs(mx - mn) < 1e-12:
#         return np.zeros_like(arr)
#     return (arr - mn) / (mx - mn)


# def get_uncertainty_cols(feature_list):
#     """
#     Example: treat any feature name containing 'entropy' as an uncertainty metric,
#     so we transform it via (1 - scaled_value).
#     You can tweak this logic as appropriate for your columns.
#     """
#     unc = []
#     for col in feature_list:
#         # e.g., if 'entropy' or 'num_clusters' are considered uncertainty:
#         if 'entropy' in col.lower() or 'clusters' in col.lower():
#             unc.append(col)
#     return unc


# class ECELoss(nn.Module):
#     """
#     Bins the confidences, then computes ECE, plus returns bin-level
#     means/stdev for both confidence & accuracy.
#     """
#     def __init__(self, n_bins=15):
#         super(ECELoss, self).__init__()
#         bin_boundaries = torch.linspace(0, 1, n_bins + 1)
#         self.bin_lowers = bin_boundaries[:-1]
#         self.bin_uppers = bin_boundaries[1:]

#     def forward(self, confidences, accuracies):
#         """
#         confidences: 1D tensor of predicted probabilities in [0,1]
#         accuracies:  1D tensor of correctness in {0,1} or [0,1]
#         """
#         ece = torch.zeros(1)
#         acc_bin_means = []
#         acc_bin_stds  = []
#         conf_bin_means = []
#         conf_bin_stds  = []

#         for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
#             in_bin = (confidences > bin_lower.item()) & (confidences <= bin_upper.item())
#             prop_in_bin = in_bin.float().mean()
#             if prop_in_bin.item() > 0:
#                 avg_acc  = accuracies[in_bin].mean()
#                 std_acc  = accuracies[in_bin].std()
#                 avg_conf = confidences[in_bin].mean()
#                 std_conf = confidences[in_bin].std()

#                 acc_bin_means.append(avg_acc.item())
#                 acc_bin_stds.append(std_acc.item())
#                 conf_bin_means.append(avg_conf.item())
#                 conf_bin_stds.append(std_conf.item())

#                 ece += torch.abs(avg_conf - avg_acc) * prop_in_bin
#             else:
#                 # If no samples in this bin, just fill with something so arrays line up
#                 acc_bin_means.append(0.0)
#                 acc_bin_stds.append(0.0)
#                 conf_bin_means.append((bin_lower.item() + bin_upper.item())/2.0)
#                 conf_bin_stds.append(0.0)

#         return ece.item(), acc_bin_means, conf_bin_means, acc_bin_stds, conf_bin_stds


# def build_polynomial_calibrator(degree=2, alpha=3.0):
#     """
#     Returns a scikit pipeline: PolynomialFeatures + Ridge
#     """
#     pipe = make_pipeline(
#         PolynomialFeatures(degree=degree),
#         Ridge(alpha=alpha)
#     )
#     return pipe


# def run_calibration_5_seeds(
#     df,
#     feature_cols,
#     accuracy_col="accuracy",
#     val_percent=0.20,
#     degree=2,
#     alpha=3.0,
#     seeds=[0,1,2,3,4]
# ):
#     """
#     For each seed in `seeds`:
#       - 80/20 split (cal vs. test)
#       - Fit polynomial calibrator on 20%
#       - Predict on 80%, min-max scale => confidences
#     Then concatenate all test confidences/accuracies across seeds.
#     Returns all_confidences, all_accuracies as numpy arrays.
#     """
#     # Decide which features are "uncertainty"
#     uncertainty_cols = get_uncertainty_cols(feature_cols)

#     # Build X
#     X_list = []
#     for col in feature_cols:
#         raw = df[col].values
#         scaled = min_max_scale(raw)
#         if col in uncertainty_cols:
#             # treat as uncertainty => confidence = 1 - scaled
#             scaled = 1.0 - scaled
#         X_list.append(scaled)
#     X_all = np.column_stack(X_list)

#     y_all = df[accuracy_col].values  # shape (N,)

#     all_confidences = []
#     all_accuracies  = []

#     for seed in seeds:
#         # Split data
#         X_cal, X_test, y_cal, y_test = train_test_split(
#             X_all, y_all,
#             test_size=(1 - val_percent),
#             random_state=seed,
#             shuffle=True
#         )
#         # Fit polynomial calibrator
#         pipe = build_polynomial_calibrator(degree=degree, alpha=alpha)
#         pipe.fit(X_cal, y_cal)

#         # Predict on test
#         y_pred = pipe.predict(X_test)
#         # Scale predicted => confidence
#         conf = min_max_scale(y_pred)

#         all_confidences.append(conf)
#         all_accuracies.append(y_test)

#     # Concatenate test sets across seeds
#     all_confidences = np.concatenate(all_confidences, axis=0)
#     all_accuracies  = np.concatenate(all_accuracies,  axis=0)
#     return all_confidences, all_accuracies


# #####################################################################
# # 2) PLOTTING FUNCTION
# #####################################################################
# def plot_reliability_diagram(confidences, accuracies, n_bins=15, title="", save_path=None):
#     """
#     Bins (confidences, accuracies), plots:
#       - Blue bars = mean accuracy in each bin
#       - Red dot = same mean
#       - Red error bar = std dev of accuracy in that bin
#       - Orange line for x=y
#       - ECE annotation
#     """
#     ece_loss = ECELoss(n_bins=n_bins)

#     # Turn into torch Tensors
#     conf_tensor = torch.tensor(confidences, dtype=torch.float32)
#     acc_tensor  = torch.tensor(accuracies,  dtype=torch.float32)

#     ece_value, acc_bin_means, conf_bin_means, acc_bin_stds, conf_bin_stds = ece_loss(conf_tensor, acc_tensor)

#     # Prepare bin centers
#     bin_boundaries = torch.linspace(0, 1, n_bins + 1)
#     bin_lowers = bin_boundaries[:-1].numpy()
#     bin_uppers = bin_boundaries[1:].numpy()
#     bin_centers = 0.5*(bin_lowers + bin_uppers)
#     width = bin_uppers[0] - bin_lowers[0]

#     plt.figure(figsize=(7, 5))
#     # Plot bars for accuracy
#     plt.bar(
#         x=bin_centers,
#         height=acc_bin_means,
#         width=width,
#         color='blue',
#         # alpha=0.6,
#         align='center',
#         label='Mean Accuracy per bin'
#     )
#     # Red error bars for std
#     plt.errorbar(
#         x=bin_centers,
#         y=acc_bin_means,
#         yerr=acc_bin_stds,
#         fmt='o',
#         color='red',
#         ecolor='red',
#         capsize=3,
#         label='Accuracy ± std'
#     )

#     # Orange diagonal line
#     plt.plot([0,1],[0,1], color='orange', label='x=y')

#     # ECE text
#     plt.text(0.05, 0.95, f"ECE = {ece_value:.4f}", transform=plt.gca().transAxes,
#              fontsize=11, verticalalignment='top',
#              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

#     plt.xlabel("Confidence")
#     plt.ylabel("Accuracy")
#     plt.title(title)
#     plt.ylim([0,1])
#     plt.xlim([0,1])
#     plt.legend(loc='lower right')
#     plt.tight_layout()

#     if save_path:
#         plt.savefig(save_path, dpi=300)
#     plt.close()


# #####################################################################
# # 3) GENERATE MULTIPLE PLOTS
# #####################################################################
# if __name__ == "__main__":
#     # Baseline single-feature sets
#     baselines = [
#         ["predictive_entropy_min_max_scaled"],
#         ["lexical_similarity_min_max_scaled"],
#         ["semantic_entropy_min_max_scaled"],
#         ["num_clusters_min_max_scaled"]
#     ]

#     # Improved: each baseline feature + "grounding_llama32V_with_cluster_wise_entropy"
#     improved = [
#         ["predictive_entropy_min_max_scaled", "grounding_llama32V_with_cluster_wise_entropy"],
#         ["lexical_similarity_min_max_scaled", "grounding_llama32V_with_cluster_wise_entropy"],
#         ["semantic_entropy_min_max_scaled", "grounding_llama32V_with_cluster_wise_entropy"],
#         ["num_clusters_min_max_scaled", "grounding_llama32V_with_cluster_wise_entropy"]
#     ]

#     # For each of the 4 baselines + 4 improved => 8 total feature sets
#     # Possibly do 2 polynomial degrees => 8*2 = 16 total
#     feature_sets = [("baseline_"+str(i), fs) for i,fs in enumerate(baselines, start=1)] \
#                  + [("improved_"+str(i), fs) for i,fs in enumerate(improved, start=1)]

#     degrees = [2,3]  # or just [2] if you only want one

#     for (set_name, feat_cols) in feature_sets:
#         for deg in degrees:
#             # 1) get test confidences across 5 seeds
#             confs, accs = run_calibration_5_seeds(
#                 df,
#                 feature_cols=feat_cols,
#                 accuracy_col="accuracy",
#                 val_percent=0.20,
#                 degree=deg,
#                 alpha=3.0,  # or whichever alpha you prefer
#                 seeds=[0,1,2,3,4]
#             )

#             # 2) plot reliability
#             plot_title = f"Reliability Diagram\n{set_name} (degree={deg})"
#             out_name = f"{set_name}_deg{deg}.png"

#             plot_reliability_diagram(
#                 confidences=confs,
#                 accuracies=accs,
#                 n_bins=15,
#                 title=plot_title,
#                 save_path=out_name
#             )

#             print(f"[INFO] Saved => {out_name}")

# # import pandas as pd
# # import numpy as np
# # import torch
# # from torch import nn
# # import matplotlib.pyplot as plt

# # from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
# # from sklearn.linear_model import Ridge
# # from sklearn.pipeline import make_pipeline
# # from sklearn.model_selection import train_test_split

# # ##############################################################################
# # # 1) Read your data (adjust path/columns as necessary).
# # ##############################################################################
# # df = pd.read_csv('/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/my_outputs/merged_with_grounding_processed_external_g_entropy.csv')

# # ##############################################################################
# # # 2) Define helper functions:
# # #    - min_max_scale
# # #    - pick uncertainty columns
# # #    - polynomial calibration
# # #    - ECE & binning logic
# # ##############################################################################

# # def min_max_scale(array):
# #     arr = np.array(array, dtype=float)
# #     mn, mx = arr.min(), arr.max()
# #     if abs(mx - mn) < 1e-12:
# #         return np.zeros_like(arr)
# #     return (arr - mn) / (mx - mn)

# # def get_uncertainty_cols(feature_list):
# #     """
# #     Dummy function. You said you have something like:
# #         combo_2_uncert = get_uncertainty_cols(combo_2_features)
# #     Adjust as needed to return which features are 'uncertainty' columns.
# #     """
# #     # For example, if the feature name contains 'entropy' we treat it as uncertainty:
# #     return [f for f in feature_list if 'entropy' in f.lower()]

# # class ECELoss(nn.Module):
# #     """
# #     Same style as your second snippet: bins the confidences, then computes
# #     ECE and also returns bin-level means and std-dev for both confidence & accuracy.
# #     """
# #     def __init__(self, n_bins=15):
# #         super(ECELoss, self).__init__()
# #         bin_boundaries = torch.linspace(0, 1, n_bins + 1)
# #         self.bin_lowers = bin_boundaries[:-1]
# #         self.bin_uppers = bin_boundaries[1:]

# #     def forward(self, confidences, accuracies):
# #         """
# #         confidences: 1D tensor of predicted probabilities in [0,1]
# #         accuracies:  1D tensor of correct/incorrect in [0 or 1]
# #         """
# #         ece = torch.zeros(1)
# #         acc_bin_means = []
# #         acc_bin_stds  = []
# #         conf_bin_means = []
# #         conf_bin_stds  = []

# #         for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
# #             in_bin = (confidences > bin_lower.item()) & (confidences <= bin_upper.item())
# #             prop_in_bin = in_bin.float().mean()
# #             if prop_in_bin.item() > 0:
# #                 avg_acc  = accuracies[in_bin].float().mean()
# #                 std_acc  = accuracies[in_bin].float().std()
# #                 avg_conf = confidences[in_bin].mean()
# #                 std_conf = confidences[in_bin].std()

# #                 acc_bin_means.append(avg_acc.item())
# #                 acc_bin_stds.append(std_acc.item())
# #                 conf_bin_means.append(avg_conf.item())
# #                 conf_bin_stds.append(std_conf.item())

# #                 ece += torch.abs(avg_conf - avg_acc) * prop_in_bin
# #             else:
# #                 # If no samples in this bin, just fill with zeros so arrays line up
# #                 acc_bin_means.append(0.0)
# #                 acc_bin_stds.append(0.0)
# #                 conf_bin_means.append((bin_lower.item() + bin_upper.item())/2.0)
# #                 conf_bin_stds.append(0.0)

# #         return ece.item(), acc_bin_means, conf_bin_means, acc_bin_stds, conf_bin_stds


# # def build_polynomial_calibrator(degree=2, alpha=3.0):
# #     """
# #     Returns a scikit pipeline: PolynomialFeatures + Ridge
# #     """
# #     pipe = make_pipeline(
# #         PolynomialFeatures(degree=degree),
# #         Ridge(alpha=alpha)
# #     )
# #     return pipe


# # ##############################################################################
# # # 3) Main calibration loop over 5 seeds
# # ##############################################################################
# # def run_calibration_5_seeds(
# #     df,
# #     feature_cols,
# #     accuracy_col="accuracy",
# #     uncertainty_cols=None,
# #     val_percent=0.20,
# #     degree=2,
# #     alpha=3.0,
# #     seeds=[0,1,2,3,4]
# # ):
# #     """
# #     1) For each seed in `seeds`, split data (using that seed) into calibration(20%) vs test(80%).
# #     2) Fit polynomial calibrator on calibration set, predict on test set, min-max scale predictions => confidence.
# #     3) Collect (confidence, accuracy) from each test sample across all seeds.
# #     4) Return two arrays: all_confidences, all_accuracies (concatenated from all seeds).
# #     """
# #     if uncertainty_cols is None:
# #         uncertainty_cols = []

# #     # Construct X from the features
# #     X_list = []
# #     for col in feature_cols:
# #         raw_values = df[col].values
# #         scaled = min_max_scale(raw_values)
# #         if col in uncertainty_cols:
# #             # treat as uncertainty => confidence = 1 - scaled
# #             scaled = 1.0 - scaled
# #         X_list.append(scaled)
# #     X_all = np.column_stack(X_list)

# #     y_all = df[accuracy_col].values  # shape (N, )

# #     all_confidences = []
# #     all_accuracies  = []

# #     for seed in seeds:
# #         # Split calibration vs test
# #         X_cal, X_test, y_cal, y_test = train_test_split(
# #             X_all, y_all,
# #             test_size=(1 - val_percent),
# #             random_state=seed,
# #             shuffle=True
# #         )
# #         # Fit polynomial calibrator
# #         pipe = build_polynomial_calibrator(degree=degree, alpha=alpha)
# #         pipe.fit(X_cal, y_cal)

# #         # Predict on test set
# #         y_pred = pipe.predict(X_test)
# #         # Min-max scale predicted values => conf in [0,1]
# #         conf = min_max_scale(y_pred)

# #         # Accumulate
# #         all_confidences.append(conf)
# #         all_accuracies.append(y_test)

# #     # Concatenate across seeds
# #     all_confidences = np.concatenate(all_confidences, axis=0)
# #     all_accuracies  = np.concatenate(all_accuracies,  axis=0)

# #     return all_confidences, all_accuracies


# # ##############################################################################
# # # 4) Actually run and then plot in the style of your second snippet
# # ##############################################################################

# # if __name__ == "__main__":
# #     # Example usage
# #     combo_2_features = [
# #         "predictive_entropy_min_max_scaled",
# #         "grounding_llama32V_with_external_entropy",
# #     ]
# #     # Decide which are uncertainty
# #     combo_2_uncert = get_uncertainty_cols(combo_2_features)

# #     # 4A) Get confidences & accuracies from 5 seeds
# #     all_confidences, all_accuracies = run_calibration_5_seeds(
# #         df,
# #         feature_cols=combo_2_features,
# #         accuracy_col="accuracy",
# #         uncertainty_cols=combo_2_uncert,
# #         val_percent=0.20,
# #         degree=2,
# #         alpha=3.0,
# #         seeds=[0,1,2,3,4]
# #     )

# #     # 4B) Compute bins + ECE
# #     ece_loss = ECELoss(n_bins=15)
# #     with torch.no_grad():
# #         # Convert to torch Tensors
# #         conf_tensor = torch.tensor(all_confidences, dtype=torch.float32)
# #         acc_tensor  = torch.tensor(all_accuracies,  dtype=torch.float32)

# #         ece_value, acc_bin_means, conf_bin_means, acc_bin_stds, conf_bin_stds = ece_loss(conf_tensor, acc_tensor)

# #     # 4C) Plot: bar chart with error bars
# #     #     - bar height = acc_bin_means
# #     #     - red dot at mean
# #     #     - red vertical line for std of accuracy
# #     bin_boundaries = torch.linspace(0, 1, 15 + 1)
# #     bin_lowers = bin_boundaries[:-1].numpy()
# #     bin_uppers = bin_boundaries[1:].numpy()
# #     bin_centers = 0.5*(bin_lowers + bin_uppers)
# #     width = bin_uppers[0] - bin_lowers[0]  # uniform bin width

# #     plt.figure(figsize=(8,6))
# #     # Blue bars for mean accuracy
# #     plt.bar(x=bin_centers, height=acc_bin_means, width=width, color='blue', align='center')

# #     # Red error bars: x=bin_centers, y=acc_bin_means, vertical error=acc_bin_stds
# #     plt.errorbar(
# #         x=bin_centers,
# #         y=acc_bin_means,
# #         yerr=acc_bin_stds,
# #         fmt='o',
# #         color='red',
# #         ecolor='red',
# #         capsize=3,
# #         label='Mean & Std of Accuracy'
# #     )

# #     # Also plot diagonal x=y
# #     plt.plot([0,1],[0,1], color='orange', label='x=y')

# #     # Show ECE in the corner
# #     plt.text(0.05, 0.95, f"ECE={ece_value:.4f}", transform=plt.gca().transAxes,
# #              fontsize=12, verticalalignment='top',
# #              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# #     plt.xlabel("Confidence (calibrated)")
# #     plt.ylabel("Accuracy")
# #     plt.title("Reliability Diagram (5 seeds merged)")
# #     plt.ylim([0,1])
# #     plt.xlim([0,1])
# #     plt.legend(loc='lower right')
# #     plt.tight_layout()
# #     plt.savefig("reliability_diagram_5_seeds.png", dpi=300)
# #     plt.show()

# # # import numpy as np
# # # import pandas as pd
# # # import math
# # # import torch
# # # import matplotlib.pyplot as plt
# # # from torch import nn
# # # from sklearn.model_selection import train_test_split
# # # from sklearn.preprocessing import PolynomialFeatures
# # # from sklearn.linear_model import Ridge
# # # from sklearn.pipeline import make_pipeline

# # # DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# # # def min_max_scale(array):
# # #     arr = np.array(array, dtype=float)
# # #     mn, mx = arr.min(), arr.max()
# # #     if mx - mn < 1e-12:
# # #         return np.zeros_like(arr)
# # #     return (arr - mn) / (mx - mn)

# # # class _ECELoss(nn.Module):
# # #     """
# # #     Calculates Expected Calibration Error by binning 
# # #     (confidence, accuracy) pairs into n_bins.
# # #     """
# # #     def __init__(self, n_bins=10):
# # #         super(_ECELoss, self).__init__()
# # #         bin_boundaries = torch.linspace(0, 1, n_bins + 1)
# # #         self.bin_lowers = bin_boundaries[:-1]
# # #         self.bin_uppers = bin_boundaries[1:]

# # #     def forward(self, confidences, accuracies):
# # #         """
# # #         confidences: Tensor (N,) in [0,1]
# # #         accuracies:  Tensor (N,) in [0,1] or {0,1}
# # #         returns ece, acc_bin, conf_bin, std_acc_bin, std_conf_bin
# # #         """
# # #         ece = torch.zeros(1, device=DEVICE)
# # #         acc_bin = []
# # #         std_acc_bin = []
# # #         conf_bin = []
# # #         std_conf_bin = []

# # #         for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
# # #             in_bin = (confidences > bin_lower.item()) & (confidences <= bin_upper.item())
# # #             prop_in_bin = in_bin.float().mean()
# # #             if prop_in_bin.item() > 0:
# # #                 # Accuracy stats
# # #                 acc_vals = accuracies[in_bin]
# # #                 mean_acc = acc_vals.mean()
# # #                 std_acc  = acc_vals.std()

# # #                 # Confidence stats
# # #                 conf_vals = confidences[in_bin]
# # #                 mean_conf = conf_vals.mean()
# # #                 std_conf  = conf_vals.std()

# # #                 acc_bin.append(mean_acc.item())
# # #                 std_acc_bin.append(std_acc.item())
# # #                 conf_bin.append(mean_conf.item())
# # #                 std_conf_bin.append(std_conf.item())

# # #                 ece += torch.abs(mean_conf - mean_acc) * prop_in_bin
# # #             else:
# # #                 acc_bin.append(0.0)
# # #                 std_acc_bin.append(0.0)
# # #                 conf_bin.append(0.0)
# # #                 std_conf_bin.append(0.0)

# # #         return ece.item(), acc_bin, conf_bin, std_acc_bin, std_conf_bin


# # # def run_multi_seed_calibration_and_plot(
# # #     df,
# # #     feature_cols,
# # #     accuracy_col,
# # #     seeds=[0,1,2,3,4],
# # #     degree=2,
# # #     alpha=3.0,
# # #     val_percent=0.20,
# # #     n_bins=10,
# # #     output_plot="reliability_multi_seed.png"
# # # ):
# # #     """
# # #     1) For multiple seeds, split data -> calibration/test
# # #     2) Fit polynomial + Ridge on calibration
# # #     3) Predict on test, scale predictions to [0,1]
# # #     4) Bin confidence & accuracy -> ECE
# # #     5) Plot in the style:
# # #        - Purple bars for binned accuracy (± black error bar)
# # #        - Red dots with red error bars for confidence (std dev), no connecting line
# # #        - Orange diagonal x=y
# # #        - ECE text at top
# # #     """

# # #     # Build feature matrix
# # #     X_list = []
# # #     for col in feature_cols:
# # #         raw_vals = df[col].values
# # #         scaled   = min_max_scale(raw_vals)
# # #         X_list.append(scaled)
# # #     X_all = np.column_stack(X_list)
# # #     y_all = df[accuracy_col].values

# # #     # We store final stats from the last seed
# # #     final_ece = 0.0
# # #     acc_bin_final = []
# # #     std_acc_bin_final = []
# # #     conf_bin_final = []
# # #     std_conf_bin_final = []

# # #     for s in seeds:
# # #         X_cal, X_test, y_cal, y_test = train_test_split(
# # #             X_all, y_all,
# # #             test_size=(1 - val_percent),
# # #             random_state=s,
# # #             shuffle=True
# # #         )

# # #         # Fit polynomial + Ridge
# # #         pipe = make_pipeline(
# # #             PolynomialFeatures(degree=degree),
# # #             Ridge(alpha=alpha)
# # #         )
# # #         pipe.fit(X_cal, y_cal)

# # #         # Predict on test
# # #         y_pred = pipe.predict(X_test)
# # #         y_pred_scaled = min_max_scale(y_pred)

# # #         # Evaluate ECE
# # #         ece_loss = _ECELoss(n_bins=n_bins).to(DEVICE)
# # #         conf_torch = torch.tensor(y_pred_scaled, device=DEVICE, dtype=torch.float)
# # #         acc_torch  = torch.tensor(y_test,        device=DEVICE, dtype=torch.float)

# # #         ece_val, acc_bin, c_bin, std_acc_bin, std_conf_bin = ece_loss(conf_torch, acc_torch)
# # #         print(f"[Seed={s}] ECE={ece_val:.3f}")

# # #         # We'll keep track of only the last seed
# # #         final_ece = ece_val
# # #         acc_bin_final = acc_bin
# # #         std_acc_bin_final = std_acc_bin
# # #         conf_bin_final = c_bin
# # #         std_conf_bin_final = std_conf_bin

# # #     # Plot from the last seed's bins
# # #     bin_edges = np.linspace(0, 1, n_bins+1)
# # #     bin_centers= (bin_edges[:-1] + bin_edges[1:]) / 2
# # #     width = bin_edges[1] - bin_edges[0]

# # #     plt.figure(figsize=(8,6))

# # #     # 1) Purple bars for binned accuracy
# # #     plt.bar(
# # #         bin_centers,
# # #         acc_bin_final,
# # #         yerr=std_acc_bin_final,
# # #         width=width*0.8,
# # #         color='purple',
# # #         alpha=1.0,
# # #         ecolor='black',
# # #         capsize=3,
# # #         label="{'Confidence' = 1-Uncertainty}"
# # #     )

# # #     # 2) Red points for confidence mean, with red vertical lines for std dev
# # #     #    No connecting line -> we set `linestyle='none'` or `fmt='o'` in errorbar
# # #     plt.errorbar(
# # #         bin_centers,
# # #         conf_bin_final,
# # #         yerr=std_conf_bin_final,
# # #         fmt='o',
# # #         color='red',
# # #         ecolor='red',
# # #         capsize=4,
# # #         linewidth=0,         # no line connecting points
# # #         label="Confidence $\pm$ std"
# # #     )

# # #     # 3) Orange diagonal x=y
# # #     plt.plot([0,1],[0,1], color='orange', label='x=y', linewidth=2)

# # #     plt.xlabel("Confidence")
# # #     plt.ylabel("Accuracy")
# # #     plt.xlim([0,1])
# # #     plt.ylim([0,1])
# # #     plt.title("COQA")

# # #     # ECE text
# # #     plt.text(
# # #         0.05, 0.92,
# # #         s=f"ECE={final_ece}",
# # #         transform=plt.gca().transAxes,
# # #         fontsize=14
# # #     )

# # #     plt.legend(loc='upper left')
# # #     plt.tight_layout()
# # #     plt.savefig(output_plot, dpi=300)
# # #     plt.close()
# # #     print(f"[INFO] Saved => {output_plot}")


# # # if __name__ == "__main__":
# # #     # 1) Load your DataFrame
# # #     df = pd.read_csv("/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/my_outputs/merged_with_grounding_processed_external_g_entropy.csv")

# # #     # 2) Define features & accuracy
# # #     my_features = ["predictive_entropy_min_max_scaled", "grounding_llama32V_with_external_entropy"]
# # #     accuracy_col = "accuracy"

# # #     # 3) Run multi-seed calibration & final plot
# # #     run_multi_seed_calibration_and_plot(
# # #         df=df,
# # #         feature_cols=my_features,
# # #         accuracy_col=accuracy_col,
# # #         seeds=[0,1,2,3,4],
# # #         degree=2,
# # #         alpha=3.0,
# # #         val_percent=0.20,
# # #         n_bins=10,
# # #         output_plot="test_gao.png"
# # #     )

# # # # import numpy as np
# # # # import pandas as pd
# # # # import math
# # # # import torch
# # # # import matplotlib.pyplot as plt
# # # # from torch import nn
# # # # from sklearn.model_selection import train_test_split
# # # # from sklearn.preprocessing import PolynomialFeatures
# # # # from sklearn.linear_model import Ridge
# # # # from sklearn.pipeline import make_pipeline

# # # # # Use GPU if available
# # # # DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# # # # def min_max_scale(array):
# # # #     arr = np.array(array, dtype=float)
# # # #     mn, mx = arr.min(), arr.max()
# # # #     if mx - mn < 1e-12:
# # # #         return np.zeros_like(arr)
# # # #     return (arr - mn) / (mx - mn)

# # # # class _ECELoss(nn.Module):
# # # #     """
# # # #     Calculates the Expected Calibration Error for a set of 
# # # #     (confidence, accuracy) pairs, dividing them into N bins.
# # # #     """
# # # #     def __init__(self, n_bins=10):
# # # #         super(_ECELoss, self).__init__()
# # # #         bin_boundaries = torch.linspace(0, 1, n_bins + 1)
# # # #         self.bin_lowers = bin_boundaries[:-1]
# # # #         self.bin_uppers = bin_boundaries[1:]

# # # #     def forward(self, confidences, accuracies):
# # # #         """
# # # #         confidences: Tensor of shape (N,) in [0,1]
# # # #         accuracies:  Tensor of shape (N,) in [0,1] or {0,1}

# # # #         Returns: (ece, acc_bin, conf_bin, std_acc_bin, std_conf_bin)
# # # #         """
# # # #         ece = torch.zeros(1, device=DEVICE)
# # # #         acc_bin = []
# # # #         std_acc_bin = []
# # # #         conf_bin = []
# # # #         std_conf_bin = []

# # # #         for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
# # # #             in_bin = (confidences > bin_lower.item()) & (confidences <= bin_upper.item())
# # # #             prop_in_bin = in_bin.float().mean()
# # # #             if prop_in_bin.item() > 0:
# # # #                 # Average accuracy in this bin
# # # #                 acc_in_bin = accuracies[in_bin]
# # # #                 accuracy_in_bin = acc_in_bin.mean()
# # # #                 std_acc_in_bin  = acc_in_bin.std()

# # # #                 # Average confidence in this bin
# # # #                 conf_in_bin = confidences[in_bin]
# # # #                 avg_conf_in_bin = conf_in_bin.mean()
# # # #                 std_conf_in_bin = conf_in_bin.std()

# # # #                 acc_bin.append(accuracy_in_bin.item())
# # # #                 std_acc_bin.append(std_acc_in_bin.item())
# # # #                 conf_bin.append(avg_conf_in_bin.item())
# # # #                 std_conf_bin.append(std_conf_in_bin.item())

# # # #                 # Weighted difference
# # # #                 ece += torch.abs(avg_conf_in_bin - accuracy_in_bin) * prop_in_bin
# # # #             else:
# # # #                 acc_bin.append(0.0)
# # # #                 std_acc_bin.append(0.0)
# # # #                 conf_bin.append(0.0)
# # # #                 std_conf_bin.append(0.0)

# # # #         return ece.item(), acc_bin, conf_bin, std_acc_bin, std_conf_bin


# # # # def run_multi_seed_calibration_and_plot(
# # # #     df,
# # # #     feature_cols,   
# # # #     accuracy_col,   
# # # #     seeds=[0,1,2,3,4],
# # # #     degree=2,
# # # #     alpha=3.0,
# # # #     val_percent=0.20,
# # # #     n_bins=10,
# # # #     output_plot="reliability_multi_seed.png"
# # # # ):
# # # #     """
# # # #     1) For each seed, do train_test_split => calibration/test
# # # #     2) Fit PolynomialFeatures+Ridge on (X_cal, y_cal)
# # # #     3) Predict on test => min-max scale => produce confidences in [0,1]
# # # #     4) Compute ECE in N bins
# # # #     5) Plot the last seed's results in a style 
# # # #        close to the sample reliability diagram:
# # # #          - Blue bars for binned accuracy (with black error bars)
# # # #          - Red points/line for binned confidence (with optional error bars)
# # # #          - Orange diagonal x=y
# # # #     """
# # # #     # Build feature matrix
# # # #     X_list = []
# # # #     for col in feature_cols:
# # # #         raw_vals = df[col].values
# # # #         scaled   = min_max_scale(raw_vals)
# # # #         X_list.append(scaled)
# # # #     X_all = np.column_stack(X_list)
# # # #     y_all = df[accuracy_col].values

# # # #     all_acc_bins   = []
# # # #     all_std_acc_bins = []
# # # #     all_conf_bins  = []
# # # #     all_std_conf_bins = []

# # # #     final_ece = 0.0

# # # #     # Loop over seeds
# # # #     for s in seeds:
# # # #         # Split calibration vs test
# # # #         X_cal, X_test, y_cal, y_test = train_test_split(
# # # #             X_all, y_all,
# # # #             test_size=(1 - val_percent),
# # # #             random_state=s,
# # # #             shuffle=True
# # # #         )

# # # #         # Fit polynomial + ridge
# # # #         pipe = make_pipeline(
# # # #             PolynomialFeatures(degree=degree),
# # # #             Ridge(alpha=alpha)
# # # #         )
# # # #         pipe.fit(X_cal, y_cal)

# # # #         # Predict on test
# # # #         y_pred = pipe.predict(X_test)

# # # #         # Scale to [0..1]
# # # #         y_pred_scaled = min_max_scale(y_pred)

# # # #         # ECE
# # # #         ece_loss = _ECELoss(n_bins=n_bins).to(DEVICE)
# # # #         conf_torch = torch.tensor(y_pred_scaled, device=DEVICE, dtype=torch.float)
# # # #         acc_torch  = torch.tensor(y_test,        device=DEVICE, dtype=torch.float)

# # # #         ece_val, acc_bin, conf_bin, std_acc_bin, std_conf_bin = ece_loss(conf_torch, acc_torch)
# # # #         print(f"[Seed={s}] ECE={ece_val:.3f}")

# # # #         # Store
# # # #         all_acc_bins.append(acc_bin)
# # # #         all_std_acc_bins.append(std_acc_bin)
# # # #         all_conf_bins.append(conf_bin)
# # # #         all_std_conf_bins.append(std_conf_bin)

# # # #         final_ece = ece_val  # Overwrite with the last seed's ECE

# # # #     # Use the last seed's bins for the final plot
# # # #     acc_bin_final    = all_acc_bins[-1]
# # # #     std_acc_bin_final= all_std_acc_bins[-1]
# # # #     conf_bin_final   = all_conf_bins[-1]
# # # #     std_conf_bin_final= all_std_conf_bins[-1]

# # # #     bin_edges   = np.linspace(0, 1, n_bins+1)
# # # #     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
# # # #     width       = bin_edges[1] - bin_edges[0]

# # # #     plt.figure(figsize=(8,6))

# # # #     # --- Blue bars for binned accuracy
# # # #     plt.bar(
# # # #         bin_centers,
# # # #         acc_bin_final,
# # # #         yerr=std_acc_bin_final,       # black error bars
# # # #         width=width*0.8,
# # # #         color='blue',
# # # #         alpha=0.5,
# # # #         ecolor='black',              # color for error bars
# # # #         capsize=3,                   # error bar cap size
# # # #         label="{'Confidence' = 1-Uncertainty}"
# # # #     )

# # # #     # --- The x=y diagonal line
# # # #     plt.plot([0,1],[0,1], color='orange', label='x=y', linewidth=2)

# # # #     # --- Red points/line for binned confidence
# # # #     plt.errorbar(
# # # #         bin_centers, conf_bin_final,
# # # #         yerr=std_conf_bin_final,
# # # #         fmt='o-r',
# # # #         ecolor='red',
# # # #         capsize=3,
# # # #         label='Binned Confidence'
# # # #     )

# # # #     plt.ylim([0,1])
# # # #     plt.xlim([0,1])
# # # #     plt.xlabel("Confidence")
# # # #     plt.ylabel("Accuracy")
# # # #     plt.title("COQA")  # or any dataset name you like

# # # #     # Show ECE for the last seed
# # # #     plt.text(
# # # #         0.05, 0.92,
# # # #         s=f"ECE={final_ece}",
# # # #         transform=plt.gca().transAxes,
# # # #         fontsize=12
# # # #     )
# # # #     plt.legend(loc='upper left')

# # # #     plt.tight_layout()
# # # #     plt.savefig(output_plot, dpi=300)
# # # #     plt.close()
# # # #     print(f"[INFO] Saved => {output_plot}")

# # # # if __name__ == "__main__":
# # # #     # 1) Load your DataFrame 
# # # #     df = pd.read_csv("/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/my_outputs/merged_with_grounding_processed_external_g_entropy.csv")

# # # #     # 2) Pick your features & accuracy
# # # #     my_features = ["predictive_entropy_min_max_scaled", "grounding_llama32V_with_external_entropy"]
# # # #     accuracy_col = "accuracy"

# # # #     # 3) Run the multi-seed calibration and plot
# # # #     run_multi_seed_calibration_and_plot(
# # # #         df=df,
# # # #         feature_cols=my_features,
# # # #         accuracy_col=accuracy_col,
# # # #         seeds=[0,1,2,3,4],
# # # #         degree=2,
# # # #         alpha=3.0,
# # # #         val_percent=0.20,
# # # #         n_bins=10,
# # # #         output_plot="test_gao.png"
# # # #     )

# # # # # # import numpy as np
# # # # # # import pandas as pd
# # # # # # import math
# # # # # # import torch
# # # # # # import re
# # # # # # import matplotlib.pyplot as plt
# # # # # # from torch import nn
# # # # # # from sklearn.model_selection import train_test_split
# # # # # # from sklearn.preprocessing import PolynomialFeatures
# # # # # # from sklearn.linear_model import Ridge
# # # # # # from sklearn.pipeline import make_pipeline

# # # # # # DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# # # # # # def min_max_scale(array):
# # # # # #     arr = np.array(array, dtype=float)
# # # # # #     mn, mx = arr.min(), arr.max()
# # # # # #     if mx - mn < 1e-12:
# # # # # #         # Avoid dividing by zero if all values are the same
# # # # # #         return np.zeros_like(arr)
# # # # # #     return (arr - mn) / (mx - mn)

# # # # # # class _ECELoss(nn.Module):
# # # # # #     """
# # # # # #     Adapted from https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
# # # # # #     This class calculates the Expected Calibration Error for a set of 
# # # # # #     (confidence, accuracy) pairs, dividing them into N bins.
# # # # # #     """
# # # # # #     def __init__(self, n_bins=10):
# # # # # #         super(_ECELoss, self).__init__()
# # # # # #         bin_boundaries = torch.linspace(0, 1, n_bins + 1)
# # # # # #         self.bin_lowers = bin_boundaries[:-1]
# # # # # #         self.bin_uppers = bin_boundaries[1:]

# # # # # #     def forward(self, confidences, accuracies):
# # # # # #         """
# # # # # #         confidences: Tensor of shape (N,) in [0,1]
# # # # # #         accuracies:  Tensor of shape (N,) in {0,1} or continuous [0,1]
# # # # # #         Returns: ece, acc_bin, conf_bin, std_acc_bin, std_conf_bin
# # # # # #         """
# # # # # #         ece = torch.zeros(1, device=DEVICE)
# # # # # #         acc_bin = []
# # # # # #         std_acc_bin = []
# # # # # #         conf_bin = []
# # # # # #         std_conf_bin = []

# # # # # #         for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
# # # # # #             in_bin = (confidences > bin_lower.item()) & (confidences <= bin_upper.item())
# # # # # #             prop_in_bin = in_bin.float().mean()
# # # # # #             if prop_in_bin.item() > 0:
# # # # # #                 accuracy_in_bin = accuracies[in_bin].float().mean()
# # # # # #                 std_acc_in_bin  = accuracies[in_bin].float().std()
# # # # # #                 avg_confidence_in_bin = confidences[in_bin].mean()
# # # # # #                 std_conf_in_bin       = confidences[in_bin].std()

# # # # # #                 acc_bin.append(accuracy_in_bin.item())
# # # # # #                 std_acc_bin.append(std_acc_in_bin.item())
# # # # # #                 conf_bin.append(avg_confidence_in_bin.item())
# # # # # #                 std_conf_bin.append(std_conf_in_bin.item())

# # # # # #                 ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
# # # # # #             else:
# # # # # #                 # No samples in this bin -> append zeros to keep dimension
# # # # # #                 acc_bin.append(0.0)
# # # # # #                 std_acc_bin.append(0.0)
# # # # # #                 conf_bin.append(0.0)
# # # # # #                 std_conf_bin.append(0.0)

# # # # # #         return ece.item(), acc_bin, conf_bin, std_acc_bin, std_conf_bin
    
# # # # # # def run_multi_seed_calibration_and_plot(
# # # # # #     df,
# # # # # #     feature_cols,  # Which columns to combine for calibration
# # # # # #     accuracy_col,  # e.g. "accuracy"
# # # # # #     seeds=[0,1,2,3,4],
# # # # # #     degree=2,
# # # # # #     alpha=3.0,
# # # # # #     val_percent=0.20,
# # # # # #     n_bins=10,
# # # # # #     output_plot="reliability_multi_seed.png"
# # # # # # ):
# # # # # #     """
# # # # # #     For each seed in `seeds`, do:
# # # # # #       - train_test_split with that seed
# # # # # #       - fit polynomial + Ridge on calibr set
# # # # # #       - predict on test set
# # # # # #       - min-max scale predictions to [0..1]
# # # # # #       - compute ECE with _ECELoss
# # # # # #       - store bin info
# # # # # #     Then make one plot summarizing them or just the last one, depending on your preference.
# # # # # #     """

# # # # # #     # 1) Build feature matrix + accuracy
# # # # # #     X_list = []
# # # # # #     for col in feature_cols:
# # # # # #         raw_vals = df[col].values
# # # # # #         scaled   = min_max_scale(raw_vals)
# # # # # #         X_list.append(scaled)
# # # # # #     X_all = np.column_stack(X_list)
# # # # # #     y_all = df[accuracy_col].values  # shape (N,)

# # # # # #     # We'll store all seeds' bin stats for final plotting
# # # # # #     all_acc_bins   = []
# # # # # #     all_conf_bins  = []

# # # # # #     for s in seeds:
# # # # # #         # 2) Split for calibration + test
# # # # # #         X_cal, X_test, y_cal, y_test = train_test_split(
# # # # # #             X_all, y_all,
# # # # # #             test_size=(1 - val_percent),
# # # # # #             random_state=s,
# # # # # #             shuffle=True
# # # # # #         )

# # # # # #         # 3) Fit polynomial + ridge
# # # # # #         pipe = make_pipeline(
# # # # # #             PolynomialFeatures(degree=degree),
# # # # # #             Ridge(alpha=alpha)
# # # # # #         )
# # # # # #         pipe.fit(X_cal, y_cal)

# # # # # #         # 4) Predict on test
# # # # # #         y_pred = pipe.predict(X_test)

# # # # # #         # 5) Min-max scale
# # # # # #         y_pred_scaled = min_max_scale(y_pred)

# # # # # #         # 6) Compute ECE bins with _ECELoss
# # # # # #         ece_loss = _ECELoss(n_bins=n_bins).to(DEVICE)
# # # # # #         # convert data to torch
# # # # # #         conf_torch = torch.tensor(y_pred_scaled, device=DEVICE, dtype=torch.float)
# # # # # #         acc_torch  = torch.tensor(y_test,        device=DEVICE, dtype=torch.float)

# # # # # #         ece_value, acc_bin, conf_bin, std_acc_bin, std_conf_bin = ece_loss.forward(conf_torch, acc_torch)
# # # # # #         print(f"[Seed={s}] ECE={ece_value:.3f}")

# # # # # #         # We can store the bin stats for plotting a reliability diagram
# # # # # #         all_acc_bins.append(acc_bin)
# # # # # #         all_conf_bins.append(conf_bin)

# # # # # #     # --- Step 7) Plot the reliability diagram(s) ---
# # # # # #     # We'll just show the last seed's bins for a single plot,
# # # # # #     # or you could average them, etc.
# # # # # #     # The bin centers ~ for n_bins=10 => 0-0.1, 0.1-0.2, ..., 0.9-1.0
# # # # # #     # We'll do a bar or line chart like your manager's snippet.

# # # # # #     acc_bin_final = all_acc_bins[-1]   # from last seed
# # # # # #     conf_bin_final= all_conf_bins[-1]

# # # # # #     # We'll also define bin boundaries
# # # # # #     bin_edges = np.linspace(0, 1, n_bins+1)
# # # # # #     bin_centers= (bin_edges[:-1] + bin_edges[1:]) / 2  # midpoints

# # # # # #     plt.figure(figsize=(8,6))
# # # # # #     width = bin_edges[1] - bin_edges[0]  # e.g. 0.1 if n_bins=10

# # # # # #     # Plot as bars + error bars, or a line
# # # # # #     plt.bar(bin_centers, acc_bin_final, width=width*0.8, color='blue', alpha=0.2, label='Binned Accuracy')
# # # # # #     # Perfect line
# # # # # #     plt.plot([0,1],[0,1], '--', color='orange', label='Perfect')

# # # # # #     # If you want to highlight the confidence line:
# # # # # #     plt.plot(bin_centers, conf_bin_final, marker='o', color='red', label='Binned Confidence')

# # # # # #     plt.xlabel("Confidence (binned center)")
# # # # # #     plt.ylabel("Accuracy in bin")
# # # # # #     plt.title(f"Multi-Seed Reliability (Last Seed={seeds[-1]})")
# # # # # #     plt.legend(loc='upper left')
# # # # # #     plt.ylim([0,1])
# # # # # #     plt.xlim([0,1])

# # # # # #     plt.text(0.05, 0.90, f"ECE (last seed)={ece_value:.3f}", transform=plt.gca().transAxes)

# # # # # #     plt.tight_layout()
# # # # # #     plt.savefig(output_plot, dpi=300)
# # # # # #     plt.close()
# # # # # #     print(f"[INFO] Saved => {output_plot}")
    
# # # # # # if __name__ == "__main__":
# # # # # #     # Suppose you load your big DF
# # # # # #     df = pd.read_csv("/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/my_outputs/merged_with_grounding_processed_external_g_entropy.csv")

# # # # # #     # Choose your features + accuracy
# # # # # #     my_features = ["predictive_entropy_min_max_scaled", "grounding_llama32V_with_external_entropy"]
# # # # # #     accuracy_col= "accuracy"

# # # # # #     run_multi_seed_calibration_and_plot(
# # # # # #         df=df,
# # # # # #         feature_cols=my_features,
# # # # # #         accuracy_col=accuracy_col,
# # # # # #         seeds=[0,1,2,3,4],
# # # # # #         degree=2,
# # # # # #         alpha=3.0,
# # # # # #         val_percent=0.20,
# # # # # #         n_bins=10,
# # # # # #         output_plot="test_gao.png"
# # # # # #     )

# # # # # import numpy as np
# # # # # import pandas as pd
# # # # # import math
# # # # # import torch
# # # # # import re
# # # # # import matplotlib.pyplot as plt
# # # # # from torch import nn
# # # # # from sklearn.model_selection import train_test_split
# # # # # from sklearn.preprocessing import PolynomialFeatures
# # # # # from sklearn.linear_model import Ridge
# # # # # from sklearn.pipeline import make_pipeline

# # # # # # Use GPU if available
# # # # # DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# # # # # def min_max_scale(array):
# # # # #     arr = np.array(array, dtype=float)
# # # # #     mn, mx = arr.min(), arr.max()
# # # # #     if mx - mn < 1e-12:
# # # # #         # Avoid dividing by zero if all values are the same
# # # # #         return np.zeros_like(arr)
# # # # #     return (arr - mn) / (mx - mn)

# # # # # class _ECELoss(nn.Module):
# # # # #     """
# # # # #     Adapted from: https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
# # # # #     This class calculates the Expected Calibration Error for a set of 
# # # # #     (confidence, accuracy) pairs, dividing them into N bins.
# # # # #     """
# # # # #     def __init__(self, n_bins=10):
# # # # #         super(_ECELoss, self).__init__()
# # # # #         bin_boundaries = torch.linspace(0, 1, n_bins + 1)
# # # # #         self.bin_lowers = bin_boundaries[:-1]
# # # # #         self.bin_uppers = bin_boundaries[1:]

# # # # #     def forward(self, confidences, accuracies):
# # # # #         """
# # # # #         confidences: Tensor of shape (N,) in [0,1]
# # # # #         accuracies:  Tensor of shape (N,) in [0,1] or {0,1}
        
# # # # #         Returns:
# # # # #           ece: scalar float
# # # # #           acc_bin: list of length n_bins, each bin's average accuracy
# # # # #           conf_bin: list of length n_bins, each bin's average confidence
# # # # #           std_acc_bin: list of length n_bins, stdev of accuracy in that bin
# # # # #           std_conf_bin: list of length n_bins, stdev of confidence in that bin
# # # # #         """
# # # # #         ece = torch.zeros(1, device=DEVICE)
# # # # #         acc_bin = []
# # # # #         std_acc_bin = []
# # # # #         conf_bin = []
# # # # #         std_conf_bin = []

# # # # #         for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
# # # # #             in_bin = (confidences > bin_lower.item()) & (confidences <= bin_upper.item())
# # # # #             prop_in_bin = in_bin.float().mean()
# # # # #             if prop_in_bin.item() > 0:
# # # # #                 # Average accuracy in bin
# # # # #                 accuracy_in_bin = accuracies[in_bin].float().mean()
# # # # #                 # Std dev of accuracy in bin
# # # # #                 std_acc_in_bin  = accuracies[in_bin].float().std()
# # # # #                 # Average confidence in bin
# # # # #                 avg_conf_in_bin = confidences[in_bin].mean()
# # # # #                 # Std dev of confidence in bin
# # # # #                 std_conf_in_bin = confidences[in_bin].std()

# # # # #                 acc_bin.append(accuracy_in_bin.item())
# # # # #                 std_acc_bin.append(std_acc_in_bin.item())
# # # # #                 conf_bin.append(avg_conf_in_bin.item())
# # # # #                 std_conf_bin.append(std_conf_in_bin.item())

# # # # #                 # Weighted absolute difference
# # # # #                 ece += torch.abs(avg_conf_in_bin - accuracy_in_bin) * prop_in_bin
# # # # #             else:
# # # # #                 # No samples => fill with 0 for placeholder
# # # # #                 acc_bin.append(0.0)
# # # # #                 std_acc_bin.append(0.0)
# # # # #                 conf_bin.append(0.0)
# # # # #                 std_conf_bin.append(0.0)

# # # # #         return ece.item(), acc_bin, conf_bin, std_acc_bin, std_conf_bin
    
# # # # # def run_multi_seed_calibration_and_plot(
# # # # #     df,
# # # # #     feature_cols,   # columns to combine for calibration
# # # # #     accuracy_col,   # "accuracy"
# # # # #     seeds=[0,1,2,3,4],
# # # # #     degree=2,
# # # # #     alpha=3.0,
# # # # #     val_percent=0.20,
# # # # #     n_bins=10,
# # # # #     output_plot="reliability_multi_seed.png"
# # # # # ):
# # # # #     """
# # # # #     For each seed in `seeds`, do:
# # # # #       - train_test_split with that seed
# # # # #       - fit polynomial + Ridge on calibration set
# # # # #       - predict on test set
# # # # #       - min-max scale predictions to [0..1]
# # # # #       - compute ECE with _ECELoss
# # # # #     We then plot the binned accuracy for the last seed, with error bars, 
# # # # #     plus a diagonal x=y for reference, and a text label of the last seed's ECE.
# # # # #     """
# # # # #     # 1) Build feature matrix + accuracy vector
# # # # #     X_list = []
# # # # #     for col in feature_cols:
# # # # #         raw_vals = df[col].values
# # # # #         scaled   = min_max_scale(raw_vals)
# # # # #         X_list.append(scaled)
# # # # #     X_all = np.column_stack(X_list)
# # # # #     y_all = df[accuracy_col].values  # shape (N,)

# # # # #     all_acc_bins   = []
# # # # #     all_std_acc_bins = []
# # # # #     all_conf_bins  = []

# # # # #     final_ece = 0.0

# # # # #     for s in seeds:
# # # # #         # 2) Split calibration vs test
# # # # #         X_cal, X_test, y_cal, y_test = train_test_split(
# # # # #             X_all, y_all,
# # # # #             test_size=(1 - val_percent),
# # # # #             random_state=s,
# # # # #             shuffle=True
# # # # #         )

# # # # #         # 3) Fit polynomial + ridge
# # # # #         pipe = make_pipeline(
# # # # #             PolynomialFeatures(degree=degree),
# # # # #             Ridge(alpha=alpha)
# # # # #         )
# # # # #         pipe.fit(X_cal, y_cal)

# # # # #         # 4) Predict on test
# # # # #         y_pred = pipe.predict(X_test)

# # # # #         # 5) Min-max scale predicted confidence
# # # # #         y_pred_scaled = min_max_scale(y_pred)

# # # # #         # 6) Compute ECE bins
# # # # #         ece_loss = _ECELoss(n_bins=n_bins).to(DEVICE)
# # # # #         conf_torch = torch.tensor(y_pred_scaled, device=DEVICE, dtype=torch.float)
# # # # #         acc_torch  = torch.tensor(y_test,        device=DEVICE, dtype=torch.float)

# # # # #         ece_val, acc_bin, conf_bin, std_acc_bin, std_conf_bin = ece_loss.forward(conf_torch, acc_torch)
# # # # #         print(f"[Seed={s}] ECE={ece_val:.3f}")

# # # # #         all_acc_bins.append(acc_bin)
# # # # #         all_std_acc_bins.append(std_acc_bin)
# # # # #         all_conf_bins.append(conf_bin)

# # # # #         final_ece = ece_val  # store the last seed's ECE

# # # # #     # ----- Plot the last seed's bins -----
# # # # #     acc_bin_final    = all_acc_bins[-1]
# # # # #     std_acc_bin_final= all_std_acc_bins[-1]
# # # # #     conf_bin_final   = all_conf_bins[-1]

# # # # #     bin_edges   = np.linspace(0, 1, n_bins+1)
# # # # #     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
# # # # #     width       = bin_edges[1] - bin_edges[0]

# # # # #     plt.figure(figsize=(8,6))

# # # # #     # Plot bars for binned accuracy
# # # # #     plt.bar(
# # # # #         bin_centers, 
# # # # #         acc_bin_final, 
# # # # #         yerr=std_acc_bin_final, 
# # # # #         width=width*0.8, 
# # # # #         color='blue', 
# # # # #         alpha=0.5,
# # # # #         label="{'Confidence' = 1-Uncertainty}"
# # # # #     )
# # # # #     # x=y diagonal
# # # # #     plt.plot([0,1],[0,1], color='orange', label='x=y')

# # # # #     # Optionally plot the mean confidence in each bin as red dots 
# # # # #     # (like your manager's snippet). We'll add vertical lines 
# # # # #     # if you want the standard dev for confidence too, but here we just do points
# # # # #     plt.plot(bin_centers, conf_bin_final, 'r-o', label='Binned Confidence')

# # # # #     plt.ylim([0,1])
# # # # #     plt.xlim([0,1])
# # # # #     plt.xlabel("Confidence")
# # # # #     plt.ylabel("Accuracy")
# # # # #     plt.title("COQA")  # Or whatever dataset name you want

# # # # #     # Show ECE
# # # # #     plt.text(
# # # # #         0.05, 0.92, 
# # # # #         s=f"ECE={final_ece}",
# # # # #         transform=plt.gca().transAxes,
# # # # #         fontsize=12
# # # # #     )
# # # # #     plt.legend(loc='upper left')

# # # # #     plt.tight_layout()
# # # # #     plt.savefig(output_plot, dpi=300)
# # # # #     plt.close()
# # # # #     print(f"[INFO] Saved => {output_plot}")

# # # # # if __name__ == "__main__":
# # # # #     # 1) Load your DataFrame (path unchanged)
# # # # #     df = pd.read_csv("/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/my_outputs/merged_with_grounding_processed_external_g_entropy.csv")

# # # # #     # 2) Define features & accuracy
# # # # #     my_features = ["predictive_entropy_min_max_scaled", "grounding_llama32V_with_external_entropy"]
# # # # #     accuracy_col = "accuracy"

# # # # #     # 3) Run multi-seed calibration & plot
# # # # #     run_multi_seed_calibration_and_plot(
# # # # #         df=df,
# # # # #         feature_cols=my_features,
# # # # #         accuracy_col=accuracy_col,
# # # # #         seeds=[0,1,2,3,4],
# # # # #         degree=2,
# # # # #         alpha=3.0,
# # # # #         val_percent=0.20,
# # # # #         n_bins=10,
# # # # #         output_plot="test_gao.png"
# # # # #     )
