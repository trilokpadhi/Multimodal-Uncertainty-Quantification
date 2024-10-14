import json
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import Ridge
import random
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import seaborn as sns
import math

# Function to apply min-max scaling to a list of values
def min_max_scale(values):
    min_val = min(values)
    max_val = max(values)
    if max_val - min_val == 0:
        return [1 for _ in values]  # If all values are the same, set them to 1
    return [(v - min_val) / (max_val - min_val) for v in values]

# Load uncertainty data from a pickle file
def load_uncertainty_data(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

# Filter out results that do not contain required keys
def filter_results(results):
    required_keys = {'predictive_entropy', 'lexical_similarity', 'semantic_entropy', 'num_clusters', 'accuracy'}
    return {key: value for key, value in results.items() if required_keys.issubset(value.keys())}

# Extract the necessary metrics
def extract_metrics(results):
    predictive_entropy = []
    lexical_similarity = []
    semantic_entropy = []
    semantic_clusters = []
    accuracies = []
    
    for key, value in results.items():
        predictive_entropy.append(value["predictive_entropy"])
        lexical_similarity.append(value["lexical_similarity"])
        semantic_entropy.append(value["semantic_entropy"])
        semantic_clusters.append(value["num_clusters"])  # Assuming 'num_clusters' represents semantic_clusters
        accuracies.append(value["accuracy"])
        
    return predictive_entropy, lexical_similarity, semantic_entropy, semantic_clusters, accuracies

# Fit polynomial regression and return coefficients
def fit_polynomial_regression(X_val, y_val, degree=6, alpha=3.0):
    model = make_pipeline(PolynomialFeatures(degree=degree), Ridge(alpha=alpha))
    model.fit(X_val, y_val)
    return model.named_steps['ridge'].coef_, model.named_steps['polynomialfeatures']

# Prepare the dataset for training and validation
def split_dataset(predictive_entropy, lexical_similarity, semantic_entropy, semantic_clusters, accuracies, val_percent=0.30, seed=42):
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    n_samples = len(predictive_entropy)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    n_val = int(val_percent * n_samples)
    val_indices = indices[:n_val]
    test_indices = indices[n_val:]
    
    # Calibration (validation) set
    calibration = {
        'predictive_entropy': np.array([predictive_entropy[i] for i in val_indices]),
        'lexical_similarity': np.array([lexical_similarity[i] for i in val_indices]),
        'semantic_entropy': np.array([semantic_entropy[i] for i in val_indices]),
        'semantic_clusters': np.array([semantic_clusters[i] for i in val_indices]),
        'accuracy': np.array([accuracies[i] for i in val_indices])
    }
    
    # Test set
    test_set = {
        'predictive_entropy': np.array([predictive_entropy[i] for i in test_indices]),
        'lexical_similarity': np.array([lexical_similarity[i] for i in test_indices]),
        'semantic_entropy': np.array([semantic_entropy[i] for i in test_indices]),
        'semantic_clusters': np.array([semantic_clusters[i] for i in test_indices]),
        'accuracy': np.array([accuracies[i] for i in test_indices])
    }
    
    return calibration, test_set

# Calculate the new confidence values
def calculate_confidence_values(X_test, alphas, poly_features):
    X_test_poly = poly_features.transform(X_test)
    new_confidence_values = np.dot(X_test_poly, alphas)
    return min_max_scale(new_confidence_values)

# Plot reliability diagram
def plot_reliability_diagram(confidences, accuracies, confidence_type, save_path):
    # Set the style to a built-in Matplotlib style
    plt.style.use('ggplot')

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Calculate bin statistics
    bins = np.linspace(0, 1, 11)
    bin_indices = np.digitize(confidences, bins) - 1

    avg_accuracies = []
    bin_centers = []
    for i in range(len(bins) - 1):
        bin_mask = bin_indices == i
        if bin_mask.any():
            avg_accuracy = np.mean(accuracies[bin_mask])
            avg_accuracies.append(avg_accuracy)
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
        else:
            print(f"No data found in bin {i}")

    avg_accuracies = np.array(avg_accuracies)
    bin_centers = np.array(bin_centers)

    # Plot the reliability curve
    ax.plot(bin_centers, avg_accuracies, marker='o', linestyle='-', linewidth=2, 
            markersize=8, label=f'Reliability ({confidence_type})')

    # Plot the perfect calibration line
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2, label='Perfect Calibration')

    # Fill the area between the curves
    ax.fill_between(bin_centers, bin_centers, avg_accuracies, alpha=0.2)

    # Set labels and title
    ax.set_xlabel('Confidence', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title(f'Reliability Diagram ({confidence_type})', fontsize=16, fontweight='bold')

    # Customize the grid
    ax.grid(True, linestyle=':', alpha=0.7)

    # Customize the legend
    ax.legend(fontsize=12, loc='lower right')

    # Set the aspect ratio to 'equal' for a square plot
    ax.set_aspect('equal')

    # Add textbox with statistics
    ece = np.mean(np.abs(avg_accuracies - bin_centers))
    mce = np.max(np.abs(avg_accuracies - bin_centers))
    stats_text = f'ECE: {ece:.3f}\nMCE: {mce:.3f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Customize ticks
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Tight layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

# Create a dictionary with scaled and calibrated results (Optional)
def create_scaled_results(predictive_entropy, lexical_similarity, semantic_entropy, semantic_clusters, new_confidences, accuracies):
    scaled_results = {}
    for i in range(len(predictive_entropy)):
        scaled_results[i] = {
            "predictive_entropy": predictive_entropy[i],
            "lexical_similarity": lexical_similarity[i],
            "semantic_entropy": semantic_entropy[i],
            "semantic_clusters": semantic_clusters[i],
            'confidence_predictive_entropy': new_confidences['predictive_entropy'][i],
            'confidence_lexical_similarity': new_confidences['lexical_similarity'][i],
            'confidence_semantic_entropy': new_confidences['semantic_entropy'][i],
            'confidence_semantic_clusters': new_confidences['semantic_clusters'][i],
            "accuracy": accuracies[i]
        }
    return scaled_results

# Main function to run the pipeline
def run_pipeline(filepath, save_path_prefix, seed=42):
    # Load data
    results = load_uncertainty_data(filepath)
    
    # Filter results to ensure all required keys are present
    filtered_results = filter_results(results)
    if not filtered_results:
        print("No valid data found after filtering. Please check your data file.")
        return
    
    # Extract metrics
    predictive_entropy, lexical_similarity, semantic_entropy, semantic_clusters, accuracies = extract_metrics(filtered_results)
    
    # Scale all the metrics except accuracies
    predictive_entropy = min_max_scale(predictive_entropy)
    lexical_similarity = min_max_scale(lexical_similarity)
    semantic_entropy = min_max_scale(semantic_entropy)
    semantic_clusters = min_max_scale(semantic_clusters)
    # accuracies are binary and should not be scaled
    
    # Ensure that all metrics have the same length
    assert len(predictive_entropy) == len(lexical_similarity) == len(semantic_entropy) == len(semantic_clusters) == len(accuracies), \
        "Mismatch in the number of samples among metrics."
    
    # Split dataset into calibration (30%) and test (70%)
    calibration, test_set = split_dataset(predictive_entropy, lexical_similarity, semantic_entropy, semantic_clusters, accuracies, val_percent=0.30, seed=seed)
    
    # Calibration for Predictive Entropy
    # Confidence is defined as 1 - uncertainty
    X_calib_pe = 1 - calibration['predictive_entropy'].reshape(-1, 1)
    y_calib_pe = calibration['accuracy']
    alphas_pe, poly_features_pe = fit_polynomial_regression(X_calib_pe, y_calib_pe)
    calibrated_conf_pe = calculate_confidence_values(X_calib_pe, alphas_pe, poly_features_pe)
    
    # Calibration for Lexical Similarity
    # Lexical similarity is already confidence
    X_calib_ls = calibration['lexical_similarity'].reshape(-1, 1)
    y_calib_ls = calibration['accuracy']
    alphas_ls, poly_features_ls = fit_polynomial_regression(X_calib_ls, y_calib_ls)
    calibrated_conf_ls = calculate_confidence_values(X_calib_ls, alphas_ls, poly_features_ls)
    
    # Calibration for Semantic Entropy
    # Confidence is defined as 1 - uncertainty
    X_calib_se = 1 - calibration['semantic_entropy'].reshape(-1, 1)
    y_calib_se = calibration['accuracy']
    alphas_se, poly_features_se = fit_polynomial_regression(X_calib_se, y_calib_se)
    calibrated_conf_se = calculate_confidence_values(X_calib_se, alphas_se, poly_features_se)
    
    # Calibration for Semantic Clusters
    # Confidence is defined as 1 - (scaled) semantic_clusters
    X_calib_sc = 1 - calibration['semantic_clusters'].reshape(-1, 1)
    y_calib_sc = calibration['accuracy']
    alphas_sc, poly_features_sc = fit_polynomial_regression(X_calib_sc, y_calib_sc)
    calibrated_conf_sc = calculate_confidence_values(X_calib_sc, alphas_sc, poly_features_sc)
    
    # Apply calibration to test set
    # Predictive Entropy
    X_test_pe = 1 - test_set['predictive_entropy'].reshape(-1, 1)
    X_test_pe_poly = poly_features_pe.transform(X_test_pe)
    calibrated_test_conf_pe = np.dot(X_test_pe_poly, alphas_pe)
    calibrated_test_conf_pe = min_max_scale(calibrated_test_conf_pe)
    
    # Lexical Similarity
    X_test_ls = test_set['lexical_similarity'].reshape(-1, 1)
    X_test_ls_poly = poly_features_ls.transform(X_test_ls)
    calibrated_test_conf_ls = np.dot(X_test_ls_poly, alphas_ls)
    calibrated_test_conf_ls = min_max_scale(calibrated_test_conf_ls)
    
    # Semantic Entropy
    X_test_se = 1 - test_set['semantic_entropy'].reshape(-1, 1)
    X_test_se_poly = poly_features_se.transform(X_test_se)
    calibrated_test_conf_se = np.dot(X_test_se_poly, alphas_se)
    calibrated_test_conf_se = min_max_scale(calibrated_test_conf_se)
    
    # Semantic Clusters
    X_test_sc = 1 - test_set['semantic_clusters'].reshape(-1, 1)
    X_test_sc_poly = poly_features_sc.transform(X_test_sc)
    calibrated_test_conf_sc = np.dot(X_test_sc_poly, alphas_sc)
    calibrated_test_conf_sc = min_max_scale(calibrated_test_conf_sc)
    
    # Uncalibrated confidences
    uncalib_conf_pe = X_test_pe.flatten()
    uncalib_conf_ls = X_test_ls.flatten()  # Lexical similarity is already confidence
    uncalib_conf_se = X_test_se.flatten()
    uncalib_conf_sc = X_test_sc.flatten()
    
    # Actual accuracies
    test_accuracies = test_set['accuracy']
    
    # Plot reliability diagrams
    # Predictive Entropy
    plot_reliability_diagram(uncalib_conf_pe, test_accuracies, "Predictive Entropy (Uncalibrated)", f"{save_path_prefix}_pe_uncalibrated.png")
    plot_reliability_diagram(calibrated_test_conf_pe, test_accuracies, "Predictive Entropy (Calibrated)", f"{save_path_prefix}_pe_calibrated.png")
    
    # Lexical Similarity
    plot_reliability_diagram(uncalib_conf_ls, test_accuracies, "Lexical Similarity (Uncalibrated)", f"{save_path_prefix}_ls_uncalibrated.png")
    plot_reliability_diagram(calibrated_test_conf_ls, test_accuracies, "Lexical Similarity (Calibrated)", f"{save_path_prefix}_ls_calibrated.png")
    
    # Semantic Entropy
    plot_reliability_diagram(uncalib_conf_se, test_accuracies, "Semantic Entropy (Uncalibrated)", f"{save_path_prefix}_se_uncalibrated.png")
    plot_reliability_diagram(calibrated_test_conf_se, test_accuracies, "Semantic Entropy (Calibrated)", f"{save_path_prefix}_se_calibrated.png")
    
    # Semantic Clusters
    plot_reliability_diagram(uncalib_conf_sc, test_accuracies, "Semantic Clusters (Uncalibrated)", f"{save_path_prefix}_sc_uncalibrated.png")
    plot_reliability_diagram(calibrated_test_conf_sc, test_accuracies, "Semantic Clusters (Calibrated)", f"{save_path_prefix}_sc_calibrated.png")
    
    print(f"Reliability diagrams saved with prefix '{save_path_prefix}'")

# Example usage
if __name__ == "__main__":
    # Replace the filepath with the path to your pickle file
    pickle_filepath = '/home/ubuntu/Multimodal-Uncertainty-Quantification/runs/llava_gqa_yes_gsam_grounding_random_100/uncertainty/uncertainty_scores_baseline.pkl'
    
    # Replace 'reliability_diagram' with your desired save path prefix
    run_pipeline(pickle_filepath, "reliability_diagram")