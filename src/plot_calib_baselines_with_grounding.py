import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
# import xgboost as xgb

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
    # Including 'question_id' as a required key
    required_keys = {'predictive_entropy', 'lexical_similarity', 'semantic_entropy', 'num_clusters', 'accuracy', 'question_id'}
    return {key: value for key, value in results.items() if required_keys.issubset(value.keys())}

# Extract the necessary metrics, including question_ids
def extract_metrics(results):
    predictive_entropy = []
    lexical_similarity = []
    semantic_entropy = []
    semantic_clusters = []
    accuracies = []
    question_ids = []
    
    for key, value in results.items():
        predictive_entropy.append(value["predictive_entropy"])
        lexical_similarity.append(value["lexical_similarity"])
        semantic_entropy.append(value["semantic_entropy"])
        semantic_clusters.append(value["num_clusters"])  # Assuming 'num_clusters' represents semantic_clusters
        accuracies.append(value["accuracy"])
        # Extract question_id; if not present, use the key as question_id
        # question_id = value.get("question_id", key)
        question_ids.append(key)
            
    return predictive_entropy, lexical_similarity, semantic_entropy, semantic_clusters, accuracies, question_ids

# Load grounding scores from grounding files and map them to question_ids
def load_grounding_scores(root_dir_grounding):
    grounding_files = os.listdir(root_dir_grounding)
    question_grounding_scores_dict = {}
    
    # for file in grounding_files:
    for file in tqdm(grounding_files, desc='Loading grounding scores'):
        if not file.endswith('.pkl'):
            continue  # Skip non-pickle files
        path = os.path.join(root_dir_grounding, file)
        grounding_score_list = []
        
        # Check if the file is empty
        if os.path.getsize(path) == 0:
            print(f"Skipping empty file: {file}")
            continue
        
        with open(path, 'rb') as f:
            grounding_data = pickle.load(f)
            question_id = grounding_data.get('question_id', [])
            for key in grounding_data.keys():
                if 'response' in key:
                    response_key = key
                    response = grounding_data.get(response_key, {})
                    ## check if decoded output is '' or tokens less than 3
                    if 'decoded_output' in response:
                        decoded_output = response['decoded_output']
                        if len(decoded_output.split()) < 3:
                            continue
                    if 'grounding_with_gd_score' in response:
                        # grounding_score = response['grounding_score']
                        grounding_score = response.get('grounding_with_gd_score', None)
                        grounding_score_list.append(grounding_score)
                    else:
                        # Assign a default grounding score if missing
                        # grounding_score = 0
                        continue  # Skip this response if grounding score is missing
        # question_grounding_scores_dict[question_id] = sum(grounding_score_list)/len(grounding_score_list) if len(grounding_score_list) > 0 else None
        if grounding_score_list:
            average_score = sum(grounding_score_list) / len(grounding_score_list)
            question_grounding_scores_dict[str(question_id)] = average_score # stringifying question_id to match the format of question_ids
        else:
            question_grounding_scores_dict[str(question_id)] = None  # Assign a default score
    return question_grounding_scores_dict

# Fit polynomial regression and return coefficients
def fit_polynomial_regression(X_val, y_val, degree=2, alpha=3.0):
    model = make_pipeline(PolynomialFeatures(degree=degree), Ridge(alpha=alpha))
    model.fit(X_val, y_val)
    return model.named_steps['ridge'].coef_, model.named_steps['polynomialfeatures']

# def fit_xgboost(X_val, y_val):
#     model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, objective='reg:squarederror')
#     model.fit(X_val, y_val)
#     return model

# Prepare the dataset for training and validation
def split_dataset(predictive_entropy, lexical_similarity, semantic_entropy, semantic_clusters, grounding_scores, accuracies, val_percent=0.30, seed=42):
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
        'grounding_score': np.array([grounding_scores[i] for i in val_indices]),
        'accuracy': np.array([accuracies[i] for i in val_indices])
    }
    
    # Test set
    test_set = {
        'predictive_entropy': np.array([predictive_entropy[i] for i in test_indices]),
        'lexical_similarity': np.array([lexical_similarity[i] for i in test_indices]),
        'semantic_entropy': np.array([semantic_entropy[i] for i in test_indices]),
        'semantic_clusters': np.array([semantic_clusters[i] for i in test_indices]),
        'grounding_score': np.array([grounding_scores[i] for i in test_indices]),
        'accuracy': np.array([accuracies[i] for i in test_indices])
    }
    
    return calibration, test_set

# Calculate the new confidence values
def calculate_confidence_values(X_test, alphas, poly_features):
    X_test_poly = poly_features.transform(X_test)
    new_confidence_values = np.dot(X_test_poly, alphas)
    return min_max_scale(new_confidence_values)
    # return new_confidence_values

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
def create_scaled_results(predictive_entropy, lexical_similarity, semantic_entropy, semantic_clusters, grounding_scores, new_confidences, accuracies):
    scaled_results = {}
    for i in range(len(predictive_entropy)):
        scaled_results[i] = {
            "predictive_entropy": predictive_entropy[i],
            "lexical_similarity": lexical_similarity[i],
            "semantic_entropy": semantic_entropy[i],
            "semantic_clusters": semantic_clusters[i],
            "grounding_score": grounding_scores[i],
            'confidence_predictive_entropy': new_confidences['predictive_entropy'][i],
            'confidence_lexical_similarity': new_confidences['lexical_similarity'][i],
            'confidence_semantic_entropy': new_confidences['semantic_entropy'][i],
            'confidence_semantic_clusters': new_confidences['semantic_clusters'][i],
            'confidence_grounding_score': new_confidences['grounding_score'][i],
            "accuracy": accuracies[i]
        }
    return scaled_results

# Main function to run the pipeline
def run_pipeline(uncertainty_filepath, grounding_root_dir, save_path_prefix, seed=42):
    # Load uncertainty data
    results = load_uncertainty_data(uncertainty_filepath)
    
    # # Filter results to ensure all required keys are present
    # filtered_results = filter_results(results)
    # if not filtered_results:
    #     print("No valid data found after filtering. Please check your data file.")
    #     return
    
    # def filter_results(results):
    #     # Exclude entries where any required metric is missing or None
    #     required_keys = {'predictive_entropy', 'lexical_similarity', 'semantic_entropy', 'num_clusters', 'accuracy'}
    #     return {qid: value for qid, value in results.items() if required_keys.issubset(value.keys()) and all(value[key] is not None for key in required_keys)}
    
    # results = filter_results(results)
    # Extract metrics
    predictive_entropy, lexical_similarity, semantic_entropy, semantic_clusters, accuracies, question_ids = extract_metrics(results)
    
    # Load grounding scores
    grounding_scores_mapping = load_grounding_scores(grounding_root_dir)
    print(f"Loaded {len(grounding_scores_mapping)} grounding scores.")
    # Map grounding scores to each sample based on question_id
    grounding_scores = [grounding_scores_mapping.get(qid, None) for qid in question_ids]
    
    # remove None values from grounding scores
    predictive_entropy = [p for p, g in zip(predictive_entropy, grounding_scores) if g is not None]
    
    lexical_similarity = [l for l, g in zip(lexical_similarity, grounding_scores) if g is not None]
    
    semantic_entropy = [s for s, g in zip(semantic_entropy, grounding_scores) if g is not None]
    
    semantic_clusters = [s for s, g in zip(semantic_clusters, grounding_scores) if g is not None]
    
    accuracies = [a for a, g in zip(accuracies, grounding_scores) if g is not None]
    
    grounding_scores = [g for g in grounding_scores if g is not None]
    
    assert len(predictive_entropy) == len(lexical_similarity) == len(semantic_entropy) == len(semantic_clusters) == len(grounding_scores) == len(accuracies), \
        "Mismatch in the number of samples among metrics."
    
    # Scale all the metrics except accuracies
    predictive_entropy_scaled = min_max_scale(predictive_entropy)
    lexical_similarity_scaled = min_max_scale(lexical_similarity)
    semantic_entropy_scaled = min_max_scale(semantic_entropy)
    semantic_clusters_scaled = min_max_scale(semantic_clusters)
    grounding_scores_scaled = min_max_scale(grounding_scores)
    # grounding_scores_scaled = grounding_scores
    grounding_scores_scaled = min_max_scale(grounding_scores)
    # accuracies = np.array(accuracies)
    # accuracies = min_max_scale(accuracies)
    # accuracies are binary and should not be scaled
    
    # map all the metrics to thee question_ids
    filtered_question_ids = [qid for qid, g in zip(question_ids, grounding_scores) if g is not None]
    
    # results = zip(predictive_entropy_scaled, lexical_similarity_scaled, semantic_entropy_scaled, semantic_clusters_scaled, grounding_scores_scaled, accuracies)
    
    results_dict = dict(zip(filtered_question_ids, zip(predictive_entropy_scaled, lexical_similarity_scaled, semantic_entropy_scaled, semantic_clusters_scaled, grounding_scores_scaled, accuracies)))
    
    
    # map all the metrics to thee question_ids, tot 
    # Ensure that all metrics have the same length
    assert len(predictive_entropy_scaled) == len(lexical_similarity_scaled) == len(semantic_entropy_scaled) == len(semantic_clusters_scaled) == len(grounding_scores_scaled) == len(accuracies), \
        "Mismatch in the number of samples among metrics."
    
    # Split dataset into calibration (30%) and test (70%)
    calibration, test_set = split_dataset(
        predictive_entropy_scaled, 
        lexical_similarity_scaled, 
        semantic_entropy_scaled, 
        semantic_clusters_scaled, 
        grounding_scores_scaled, 
        accuracies, 
        val_percent=0.25, 
        seed=seed
    )
    
    # get the question_ids for the calibration and test set
    calibration_question_ids = [qid for qid, g in zip(filtered_question_ids, grounding_scores_scaled) if g in calibration['grounding_score']]
    test_question_ids = [qid for qid, g in zip(filtered_question_ids, grounding_scores_scaled) if g in test_set['grounding_score']]
    
    # map all the metrics to the test and calibration question_ids
    calibration_results = {qid: results_dict[qid] for qid in calibration_question_ids}
    test_results = {qid: results_dict[qid] for qid in test_question_ids}
    
    
    # List of original metrics and their corresponding calibration transformations
    metrics = {
        'predictive_entropy': {
            'X_calib': 1 - calibration['predictive_entropy'].reshape(-1, 1),
            'X_test': 1 - test_set['predictive_entropy'].reshape(-1, 1)
        },
        'lexical_similarity': {
            'X_calib': calibration['lexical_similarity'].reshape(-1, 1),
            'X_test': test_set['lexical_similarity'].reshape(-1, 1)
        },
        'semantic_entropy': {
            'X_calib': 1 - calibration['semantic_entropy'].reshape(-1, 1),
            'X_test': 1 - test_set['semantic_entropy'].reshape(-1, 1)
        },
        'semantic_clusters': {
            'X_calib': 1 - calibration['semantic_clusters'].reshape(-1, 1),
            'X_test': 1 - test_set['semantic_clusters'].reshape(-1, 1)
        },
        'grounding_score': {
            'X_calib': calibration['grounding_score'].reshape(-1, 1),
            'X_test': test_set['grounding_score'].reshape(-1, 1)
        }
    }
    
    
    # Dictionaries to store calibration models and calibrated confidences
    calibration_models = {}
    calibrated_confidences = {}
    
    # Calibrate each metric
    for metric_name, data in metrics.items():
        X_calib = data['X_calib']
        y_calib = calibration['accuracy']
        alphas, poly_features = fit_polynomial_regression(X_calib, y_calib)
        # xgb_model = fit_xgboost(X_calib, y_calib)
        # Store the trained model for each metric
        # calibration_models[metric_name] = xgb_model
        calibration_models[metric_name] = (alphas, poly_features)
        calibrated_conf = calculate_confidence_values(X_calib, alphas, poly_features)
        # Predict the calibrated confidence values
        # calibrated_conf = xgb_model.predict(X_calib)
        calibrated_confidences[metric_name] = calibrated_conf
    
    # Apply calibration to test set and store calibrated confidences
    calibrated_test_conf = {}
    for metric_name, data in metrics.items():
        alphas, poly_features = calibration_models[metric_name]
        X_test = data['X_test']
        calibrated_conf = calculate_confidence_values(X_test, alphas, poly_features)
        calibrated_test_conf[metric_name] = calibrated_conf
    
    # map the calibrated confidences to the question_ids for each metric
    calibrated_test_results = {qid: [calibrated_test_conf[metric_name][i] for i in range(len(calibrated_test_conf[metric_name]))] for qid in test_question_ids for metric_name in metrics.keys()}
    
    # Uncalibrated confidences
    uncalib_conf = {}
    for metric_name, data in metrics.items():
        X_test = data['X_test']
        uncalib_conf[metric_name] = X_test.flatten()
    
    # Define combined metrics by adding grounding score to each original metric
    combined_metrics = {}
    for base_metric in ['predictive_entropy', 'lexical_similarity', 'semantic_entropy', 'semantic_clusters']:
        combined_metric_name = f"{base_metric}_with_grounding"
        combined_metrics[combined_metric_name] = {
            'combined_confidences': np.array(uncalib_conf[base_metric]) + np.array(uncalib_conf['grounding_score'])
        }
    
    # Scale combined metrics
    for combined_metric_name in combined_metrics:
        combined_metrics[combined_metric_name]['combined_confidences'] = min_max_scale(combined_metrics[combined_metric_name]['combined_confidences'])
    
    # Calibration for combined metrics
    combined_calibration_models = {}
    calibrated_combined_test_conf = {}
    combined_uncalib_conf = {}
    
    for combined_metric_name, data in combined_metrics.items():
        # Combine calibration confidences
        combined_X_calib = calibration['predictive_entropy'] + calibration['grounding_score'] if 'predictive_entropy_with_grounding' in combined_metric_name else \
                           calibration['lexical_similarity'] + calibration['grounding_score'] if 'lexical_similarity_with_grounding' in combined_metric_name else \
                           calibration['semantic_entropy'] + calibration['grounding_score'] if 'semantic_entropy_with_grounding' in combined_metric_name else \
                           calibration['semantic_clusters'] + calibration['grounding_score']
        
        combined_X_calib = combined_X_calib.reshape(-1, 1)
        y_calib = calibration['accuracy']
        alphas, poly_features = fit_polynomial_regression(combined_X_calib, y_calib)
        combined_calibration_models[combined_metric_name] = (alphas, poly_features)
        calibrated_conf = calculate_confidence_values(combined_X_calib, alphas, poly_features)
        combined_metrics[combined_metric_name]['calibrated_conf'] = calibrated_conf
        
        # Apply calibration to test set
        if 'predictive_entropy_with_grounding' in combined_metric_name:
            X_test_combined = test_set['predictive_entropy'] + test_set['grounding_score']
        elif 'lexical_similarity_with_grounding' in combined_metric_name:
            X_test_combined = test_set['lexical_similarity'] + test_set['grounding_score']
        elif 'semantic_entropy_with_grounding' in combined_metric_name:
            X_test_combined = test_set['semantic_entropy'] + test_set['grounding_score']
        elif 'semantic_clusters_with_grounding' in combined_metric_name:
            X_test_combined = test_set['semantic_clusters'] + test_set['grounding_score']
        else:
            X_test_combined = 0  # Default case, should not occur
        
        X_test_combined = X_test_combined.reshape(-1, 1)
        alphas, poly_features = combined_calibration_models[combined_metric_name]
        calibrated_conf = calculate_confidence_values(X_test_combined, alphas, poly_features)
        calibrated_combined_test_conf[combined_metric_name] = calibrated_conf
        
        # Uncalibrated combined confidences
        combined_uncalib_conf[combined_metric_name] = X_test_combined.flatten()
    
    # Plot reliability diagrams for original metrics
    for metric_name in metrics.keys():
        # Uncalibrated
        plot_reliability_diagram(
            uncalib_conf[metric_name], 
            # test_accuracies, 
            test_set['accuracy'],
            f"{metric_name.replace('_', ' ').title()} (Uncalibrated)", 
            f"{save_path_prefix}_{metric_name}_uncalibrated.png"
        )
        # Calibrated
        plot_reliability_diagram(
            calibrated_test_conf[metric_name], 
            # test_accuracies, 
            test_set['accuracy'],
            f"{metric_name.replace('_', ' ').title()} (Calibrated)", 
            f"{save_path_prefix}_{metric_name}_calibrated.png"
        )
    
    # Plot reliability diagrams for combined metrics
    for combined_metric_name in combined_metrics.keys():
        # Uncalibrated
        plot_reliability_diagram(
            combined_uncalib_conf[combined_metric_name], 
            # test_accuracies, 
            test_set['accuracy'],
            f"{combined_metric_name.replace('_', ' ').title()} (Uncalibrated)", 
            f"{save_path_prefix}_{combined_metric_name}_uncalibrated.png"
        )
        # Calibrated
        plot_reliability_diagram(
            calibrated_combined_test_conf[combined_metric_name], 
            # test_accuracies, 
            test_set['accuracy'],
            f"{combined_metric_name.replace('_', ' ').title()} (Calibrated)", 
            f"{save_path_prefix}_{combined_metric_name}_calibrated.png"
        )
    
    print(f"Reliability diagrams saved with prefix '{save_path_prefix}'")

# Example usage
if __name__ == "__main__":
    # Replace the following paths with your actual file paths
    # uncertainty_filepath = '/home/ubuntu/Multimodal-Uncertainty-Quantification/runs/llava_gqa_yes_gsam_grounding_random_100/uncertainty/uncertainty_scores_baseline.pkl'
    # uncertainty_filepath = '/home/ec2-user/Multimodal-Uncertainty-Quantification/runs/uncertainty/uncertainty_scores_baseline.pkl'
    # grounding_root_dir = '/mnt/myebsvolume/home/ubuntu/Multimodal-Uncertainty-Quantification/runs/llava_gqa_yes_gsam_grounding_random_10000/grounding/'
    
    ### *** CHANGE THE FOLLOWING PATHS *** ### -- below is for gqa dataset
    # uncertainty_filepath = '/mnt/data/home/ubuntu/Multimodal-Uncertainty-Quantification/runs/llava_gqa_yes_gsam_grounding_random_1000_test_17nov/uncertainty/uncertainty_scores_baseline.pkl'
    # grounding_root_dir = '/mnt/data/home/ubuntu/Multimodal-Uncertainty-Quantification/runs/llava_gqa_yes_gsam_grounding_random_1000_test_17nov/grounding_test/'
    
    ### for vqa dataset
    uncertainty_filepath = '/home/ubuntu/trilok/Multimodal-Uncertainty-Quantification/runs_vqa/llava_vqa_yes_gsam_grounding_random_1000/uncertainty/uncertainty_scores_baseline.pkl'
    grounding_root_dir = '/home/ubuntu/trilok/Multimodal-Uncertainty-Quantification/runs_vqa/llava_vqa_yes_gsam_grounding_random_1000/grounding/'
    save_path_prefix = "reliability_diagram_VQA_gsam_grounding_random_1000"
    seed = random.randint(1, 1000)
    print(f"Using random seed: {seed}")
    run_pipeline(uncertainty_filepath, grounding_root_dir, save_path_prefix, seed)