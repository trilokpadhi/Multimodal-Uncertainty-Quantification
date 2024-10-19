import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from collections import Counter
import xgboost as xgb

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
        question_ids.append(key)
            
    return predictive_entropy, lexical_similarity, semantic_entropy, semantic_clusters, accuracies, question_ids

# Load grounding scores from grounding files and map them to question_ids
def load_grounding_scores(root_dir_grounding):
    grounding_files = os.listdir(root_dir_grounding)
    question_grounding_scores_dict = {}
    
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
            question_ids = grounding_data.get('question_ids', [])
            if not question_ids:
                continue  # Skip if no question_ids
            question_id = question_ids[0]  # Assuming one question per file
            for key in grounding_data.keys():
                if 'response' in key:
                    response_key = key
                    response = grounding_data.get(response_key, {})
                    if 'grounding_score' in response:
                        grounding_score = response['grounding_score']
                        grounding_score_list.append(grounding_score)
                    else:
                        continue  # Skip this response if grounding score is missing
        # Average grounding scores if multiple responses
        if grounding_score_list:
            question_grounding_scores_dict[question_id] = sum(grounding_score_list) / len(grounding_score_list)
        else:
            question_grounding_scores_dict[question_id] = 0  # Default value if no grounding scores found
                
    return question_grounding_scores_dict

# Compute ECE and MCE
def compute_ece_mce(confidences, accuracies, num_bins=10):
    """
    Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    
    Args:
        confidences (np.array): Array of confidence scores.
        accuracies (np.array): Array of accuracy labels (0 or 1).
        num_bins (int): Number of bins to use for calibration.
        
    Returns:
        tuple: (ECE, MCE)
    """
    bins = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(confidences, bins) - 1  # Bin indices start at 0

    ece = 0.0
    mce = 0.0
    N = len(confidences)

    for i in range(num_bins):
        bin_mask = bin_indices == i
        if bin_mask.any():
            bin_conf = np.mean(confidences[bin_mask])
            bin_acc = np.mean(accuracies[bin_mask])
            bin_size = len(confidences[bin_mask])

            ece += np.abs(bin_acc - bin_conf) * (bin_size / N)
            mce = max(mce, np.abs(bin_acc - bin_conf))

    return ece, mce

# Perform grid search to find the best calibration models
def grid_search_calibration(X_calib, y_calib, param_grid, num_bins=10):
    """
    Perform grid search over specified hyperparameters to find the best calibration model.
    
    Args:
        X_calib (dict): Dictionary containing calibration inputs for each metric.
        y_calib (np.array): Calibration set targets.
        param_grid (dict): Dictionary specifying the hyperparameters to search.
        num_bins (int): Number of bins for ECE calculation.
        
    Returns:
        dict: Best parameters and corresponding calibration model for each metric.
    """
    best_params = {}
    best_models = {}
    best_ece = {}
    
    for metric in X_calib.keys():
        best_ece[metric] = float('inf')
        best_params[metric] = {}
        best_models[metric] = None
        
        print(f"\nPerforming grid search for {metric}...")
        
        for params in tqdm(generate_param_combinations(param_grid), desc=f"Grid Search for {metric}"):
            # Initialize the XGBRegressor with current hyperparameters
            model = xgb.XGBRegressor(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                objective='reg:squarederror',
                verbosity=0,
                use_label_encoder=False,
                random_state=42
            )
            
            # Fit the model
            model.fit(X_calib[metric], y_calib)
            
            # Predict calibrated confidences on calibration set
            calibrated_conf = model.predict(X_calib[metric])
            calibrated_conf = min_max_scale(calibrated_conf)
            
            # Compute ECE
            ece, _ = compute_ece_mce(np.array(calibrated_conf), y_calib, num_bins=num_bins)
            
            # Update best parameters if ECE is lower
            if ece < best_ece[metric]:
                best_ece[metric] = ece
                best_params[metric] = params
                best_models[metric] = model
        
        print(f"Best ECE for {metric}: {best_ece[metric]:.4f} with parameters: {best_params[metric]}")
    
    return best_params, best_models, best_ece

# Helper function to generate all combinations of hyperparameters
def generate_param_combinations(param_grid):
    """
    Generate all combinations of hyperparameters from the param_grid.
    
    Args:
        param_grid (dict): Dictionary specifying the hyperparameters to search.
        
    Returns:
        generator: Yields dictionaries of hyperparameter combinations.
    """
    from itertools import product
    keys = param_grid.keys()
    values = param_grid.values()
    for v in product(*values):
        yield dict(zip(keys, v))

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

# Calculate the new confidence values using the trained model
def calculate_confidence_values(model, X_test):
    calibrated_conf = model.predict(X_test)
    calibrated_conf = min_max_scale(calibrated_conf)
    return np.array(calibrated_conf)

# Plot reliability diagram
def plot_reliability_diagram(confidences, accuracies, confidence_type, save_path, num_bins=10):
    # Set the style to a built-in Matplotlib style
    plt.style.use('ggplot')

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Calculate bin statistics
    bins = np.linspace(0, 1, num_bins + 1)
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
    ece, mce = compute_ece_mce(np.array(confidences), np.array(accuracies), num_bins=num_bins)
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

# Main function to run the pipeline with optimization
def run_pipeline(
    uncertainty_filepath, 
    grounding_root_dir, 
    save_path_prefix, 
    param_grid={
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }, 
    num_iterations=5, 
    num_bins=10
):
    """
    Run the calibration pipeline with grid search and multiple iterations to find the best calibration models.

    Args:
        uncertainty_filepath (str): Path to the uncertainty scores pickle file.
        grounding_root_dir (str): Root directory containing grounding score pickle files.
        save_path_prefix (str): Prefix for saving the reliability diagrams.
        param_grid (dict): Dictionary specifying the hyperparameters to search for XGBoost.
        num_iterations (int): Number of different random splits to perform.
        num_bins (int): Number of bins to use for ECE and MCE calculations.
    """
    # Load uncertainty data
    results = load_uncertainty_data(uncertainty_filepath)

    # # Filter results to ensure all required keys are present
    # filtered_results = filter_results(results)
    # if not filtered_results:
    #     print("No valid data found after filtering. Please check your data file.")
    #     return

    # Extract metrics
    predictive_entropy, lexical_similarity, semantic_entropy, semantic_clusters, accuracies, question_ids = extract_metrics(results)

    # Load grounding scores
    grounding_scores_mapping = load_grounding_scores(grounding_root_dir)
    print(f"Loaded {len(grounding_scores_mapping)} grounding scores.")

    # Map grounding scores to each sample based on question_id
    grounding_scores = [grounding_scores_mapping.get(qid, 0) for qid in question_ids]

    # Scale all the metrics except accuracies
    predictive_entropy_scaled = min_max_scale(predictive_entropy)
    lexical_similarity_scaled = min_max_scale(lexical_similarity)
    semantic_entropy_scaled = min_max_scale(semantic_entropy)
    semantic_clusters_scaled = min_max_scale(semantic_clusters)
    grounding_scores_scaled = min_max_scale(grounding_scores)
    # accuracies are binary and should not be scaled

    # Ensure that all metrics have the same length
    assert len(predictive_entropy_scaled) == len(lexical_similarity_scaled) == len(semantic_entropy_scaled) == len(semantic_clusters_scaled) == len(grounding_scores_scaled) == len(accuracies), \
        "Mismatch in the number of samples among metrics."

    # Initialize dictionaries to store best calibration models and their ECE across iterations
    all_best_ece = {metric: [] for metric in ['predictive_entropy', 'lexical_similarity', 'semantic_entropy', 'semantic_clusters', 'grounding_score']}
    all_best_models = {metric: [] for metric in ['predictive_entropy', 'lexical_similarity', 'semantic_entropy', 'semantic_clusters', 'grounding_score']}
    all_best_params = {metric: [] for metric in ['predictive_entropy', 'lexical_similarity', 'semantic_entropy', 'semantic_clusters', 'grounding_score']}

    # Iterate over multiple splits to find the best calibration models
    for iteration in range(num_iterations):
        print(f"\n=== Iteration {iteration + 1} ===")
        # Use a different seed for each iteration to get different splits
        seed = random.randint(1, 10000)
        print(f"Using random seed: {seed}")

        # Split dataset into calibration (30%) and test (70%)
        calibration, test_set = split_dataset(
            predictive_entropy_scaled, 
            lexical_similarity_scaled, 
            semantic_entropy_scaled, 
            semantic_clusters_scaled, 
            grounding_scores_scaled, 
            accuracies, 
            val_percent=0.30, 
            seed=seed
        )

        # Define calibration metrics
        metrics = {
            'predictive_entropy': {
                'X_calib': np.array(1 - calibration['predictive_entropy']).reshape(-1, 1),
                'X_test': np.array(1 - test_set['predictive_entropy']).reshape(-1, 1)
            },
            'lexical_similarity': {
                'X_calib': np.array(calibration['lexical_similarity']).reshape(-1, 1),
                'X_test': np.array(test_set['lexical_similarity']).reshape(-1, 1)
            },
            'semantic_entropy': {
                'X_calib': np.array(1 - calibration['semantic_entropy']).reshape(-1, 1),
                'X_test': np.array(1 - test_set['semantic_entropy']).reshape(-1, 1)
            },
            'semantic_clusters': {
                'X_calib': np.array(1 - calibration['semantic_clusters']).reshape(-1, 1),
                'X_test': np.array(1 - test_set['semantic_clusters']).reshape(-1, 1)
            },
            'grounding_score': {
                'X_calib': np.array(calibration['grounding_score']).reshape(-1, 1),
                'X_test': np.array(test_set['grounding_score']).reshape(-1, 1)
            }
        }

        # Extract only X_calib for grid search
        X_calib_dict = {metric: data['X_calib'] for metric, data in metrics.items()}

        # Perform grid search to find the best calibration models for this iteration
        best_params, best_models, best_ece = grid_search_calibration(X_calib_dict, calibration['accuracy'], param_grid, num_bins=num_bins)

        # Store the best ECE and models
        for metric in best_ece.keys():
            all_best_ece[metric].append(best_ece[metric])
            all_best_models[metric].append(best_models[metric])
            all_best_params[metric].append(best_params[metric])

    # After all iterations, select the models with the lowest ECE for each metric
    final_best_models = {}
    for metric in all_best_ece.keys():
        # Find the iteration with the lowest ECE
        min_ece_index = np.argmin(all_best_ece[metric])
        final_best_models[metric] = all_best_models[metric][min_ece_index]
        print(f"\nSelected best model for {metric} with ECE={all_best_ece[metric][min_ece_index]:.4f} from iteration {min_ece_index + 1}")

    # Aggregate the best hyperparameters for reporting (optional)
    aggregated_best_params = {}
    for metric in all_best_ece.keys():
        # Find the most common best hyperparameters
        # n_estimators = [model.get_xgb_params()['n_estimators'] for model in all_best_models[metric]]
        n_estimators = [all_best_params[metric][i]['n_estimators'] for i in range(len(all_best_params[metric]))]
        max_depth = [model.get_xgb_params()['max_depth'] for model in all_best_models[metric]]
        learning_rate = [model.get_xgb_params()['learning_rate'] for model in all_best_models[metric]]
        # max_depth = [model.get_xgb_params()['max_depth'] for model in all_best_models[metric]]
        # learning_rate = [model.get_xgb_params()['learning_rate'] for model in all_best_models[metric]]
        
        n_estimators_counter = Counter(n_estimators)
        max_depth_counter = Counter(max_depth)
        learning_rate_counter = Counter(learning_rate)
        
        most_common_n_estimators = n_estimators_counter.most_common(1)[0][0] if n_estimators_counter else None
        most_common_max_depth = max_depth_counter.most_common(1)[0][0] if max_depth_counter else None
        most_common_learning_rate = learning_rate_counter.most_common(1)[0][0] if learning_rate_counter else None
        
        aggregated_best_params[metric] = {
            'n_estimators': most_common_n_estimators,
            'max_depth': most_common_max_depth,
            'learning_rate': most_common_learning_rate
        }

    print("\n=== Aggregated Best Hyperparameters ===")
    for metric, params in aggregated_best_params.items():
        print(f"{metric}: n_estimators={params['n_estimators']}, max_depth={params['max_depth']}, learning_rate={params['learning_rate']}")

    # Re-split the dataset with a fixed seed for final calibration
    final_seed = 42  # You can choose a seed that had the best overall performance
    print(f"\n=== Final Calibration with Seed {final_seed} ===")
    calibration, test_set = split_dataset(
        predictive_entropy_scaled, 
        lexical_similarity_scaled, 
        semantic_entropy_scaled, 
        semantic_clusters_scaled, 
        grounding_scores_scaled, 
        accuracies, 
        val_percent=0.30, 
        seed=final_seed
    )

    # Define calibration metrics for final calibration
    final_metrics = {
        'predictive_entropy': {
            'X_calib': np.array(1 - calibration['predictive_entropy']).reshape(-1, 1),
            'X_test': np.array(1 - test_set['predictive_entropy']).reshape(-1, 1)
        },
        'lexical_similarity': {
            'X_calib': np.array(calibration['lexical_similarity']).reshape(-1, 1),
            'X_test': np.array(test_set['lexical_similarity']).reshape(-1, 1)
        },
        'semantic_entropy': {
            'X_calib': np.array(1 - calibration['semantic_entropy']).reshape(-1, 1),
            'X_test': np.array(1 - test_set['semantic_entropy']).reshape(-1, 1)
        },
        'semantic_clusters': {
            'X_calib': np.array(1 - calibration['semantic_clusters']).reshape(-1, 1),
            'X_test': np.array(1 - test_set['semantic_clusters']).reshape(-1, 1)
        },
        'grounding_score': {
            'X_calib': np.array(calibration['grounding_score']).reshape(-1, 1),
            'X_test': np.array(test_set['grounding_score']).reshape(-1, 1)
        }
    }

    # Extract only X_calib for final calibration
    final_X_calib_dict = {metric: data['X_calib'] for metric, data in final_metrics.items()}

    # Train final calibration models using the best models selected earlier
    final_best_calibration_models = {}
    for metric, model in final_best_models.items():
        print(f"\nTraining final calibration model for {metric}...")
        # Retrieve the aggregated best hyperparameters
        params = aggregated_best_params[metric]
        # Initialize a new XGBRegressor with the best hyperparameters
        final_model = xgb.XGBRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            objective='reg:squarederror',
            verbosity=0,
            use_label_encoder=False,
            random_state=42
        )
        # Fit the model on the final calibration set
        final_model.fit(final_metrics[metric]['X_calib'], calibration['accuracy'])
        final_best_calibration_models[metric] = final_model

    # Apply final calibration models to the test set
    calibrated_test_conf = {}
    for metric, model in final_best_calibration_models.items():
        calibrated_conf = calculate_confidence_values(model, final_metrics[metric]['X_test'])
        calibrated_test_conf[metric] = calibrated_conf

    # Uncalibrated confidences
    uncalib_conf = {}
    for metric, data in final_metrics.items():
        X_test = data['X_test']
        uncalib_conf[metric] = X_test.flatten()

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

    # Calibration for combined metrics using the best hyperparameters of the base metric
    combined_calibration_models = {}
    calibrated_combined_test_conf = {}
    combined_uncalib_conf = {}
    for combined_metric_name, data in combined_metrics.items():
        # Determine the base metric
        base_metric = combined_metric_name.split('_with_grounding')[0]
        params = aggregated_best_params[base_metric]
        
        # Combine calibration confidences
        if base_metric == 'predictive_entropy':
            combined_X_calib = np.array(calibration['predictive_entropy'] + calibration['grounding_score']).reshape(-1, 1)
            combined_X_test = np.array(test_set['predictive_entropy'] + test_set['grounding_score']).reshape(-1, 1)
        elif base_metric == 'lexical_similarity':
            combined_X_calib = np.array(calibration['lexical_similarity'] + calibration['grounding_score']).reshape(-1, 1)
            combined_X_test = np.array(test_set['lexical_similarity'] + test_set['grounding_score']).reshape(-1, 1)
        elif base_metric == 'semantic_entropy':
            combined_X_calib = np.array(calibration['semantic_entropy'] + calibration['grounding_score']).reshape(-1, 1)
            combined_X_test = np.array(test_set['semantic_entropy'] + test_set['grounding_score']).reshape(-1, 1)
        elif base_metric == 'semantic_clusters':
            combined_X_calib = np.array(calibration['semantic_clusters'] + calibration['grounding_score']).reshape(-1, 1)
            combined_X_test = np.array(test_set['semantic_clusters'] + test_set['grounding_score']).reshape(-1, 1)
        else:
            combined_X_calib = np.zeros((len(calibration['grounding_score']), 1))  # Default case, should not occur
            combined_X_test = np.zeros((len(test_set['grounding_score']), 1))
    
        # Create and fit the calibration model
        print(f"\nTraining final calibration model for {combined_metric_name}...")
        combined_model = xgb.XGBRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            objective='reg:squarederror',
            verbosity=0,
            use_label_encoder=False,
            random_state=42
        )
        combined_model.fit(combined_X_calib, calibration['accuracy'])

        # Predict calibrated confidences on test set
        calibrated_conf = calculate_confidence_values(combined_model, combined_X_test)
        calibrated_combined_test_conf[combined_metric_name] = calibrated_conf

        # Uncalibrated combined confidences
        combined_uncalib_conf[combined_metric_name] = combined_X_test.flatten()

    # Plot reliability diagrams for original metrics
    for metric_name in final_metrics.keys():
        # Uncalibrated
        plot_reliability_diagram(
            uncalib_conf[metric_name], 
            test_set['accuracy'],
            f"{metric_name.replace('_', ' ').title()} (Uncalibrated)", 
            f"{save_path_prefix}_{metric_name}_uncalibrated.png",
            num_bins=num_bins
        )
        # Calibrated
        plot_reliability_diagram(
            calibrated_test_conf[metric_name], 
            test_set['accuracy'],
            f"{metric_name.replace('_', ' ').title()} (Calibrated)", 
            f"{save_path_prefix}_{metric_name}_calibrated.png",
            num_bins=num_bins
        )

    # Plot reliability diagrams for combined metrics
    for combined_metric_name in combined_metrics.keys():
        # Uncalibrated
        plot_reliability_diagram(
            combined_uncalib_conf[combined_metric_name], 
            test_set['accuracy'],
            f"{combined_metric_name.replace('_', ' ').title()} (Uncalibrated)", 
            f"{save_path_prefix}_{combined_metric_name}_uncalibrated.png",
            num_bins=num_bins
        )
        # Calibrated
        plot_reliability_diagram(
            calibrated_combined_test_conf[combined_metric_name], 
            test_set['accuracy'],
            f"{combined_metric_name.replace('_', ' ').title()} (Calibrated)", 
            f"{save_path_prefix}_{combined_metric_name}_calibrated.png",
            num_bins=num_bins
        )

    print(f"\nReliability diagrams saved with prefix '{save_path_prefix}'")

# Example usage
if __name__ == "__main__":
    # Replace the following paths with your actual file paths
    # uncertainty_filepath = '/home/ubuntu/Multimodal-Uncertainty-Quantification/runs/llava_gqa_yes_gsam_grounding_random_100/uncertainty/uncertainty_scores_baseline.pkl'
    # grounding_root_dir = '/home/ubuntu/Multimodal-Uncertainty-Quantification/runs/llava_gqa_yes_gsam_grounding_random_100/grounding'
    
    uncertainty_filepath = '/home/ec2-user/Multimodal-Uncertainty-Quantification/runs/uncertainty/uncertainty_scores_baseline.pkl'
    grounding_root_dir = '/mnt/myebsvolume/home/ubuntu/Multimodal-Uncertainty-Quantification/runs/llava_gqa_yes_gsam_grounding_random_10000/grounding/'
    save_path_prefix = "reliability_diagram_optimized_7000"
    
    # Define XGBoost hyperparameter ranges for grid search
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    
    # Set the number of iterations (different random splits)
    num_iterations = 10 # You can increase this number for more robust results
    
    # Set the number of bins for ECE calculation
    num_bins = 10
    
    # Run the pipeline
    run_pipeline(
        uncertainty_filepath=uncertainty_filepath, 
        grounding_root_dir=grounding_root_dir, 
        save_path_prefix=save_path_prefix, 
        param_grid=param_grid, 
        num_iterations=num_iterations, 
        num_bins=num_bins
    )