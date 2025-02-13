# import os
# import pickle
# import random
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import Ridge
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.pipeline import make_pipeline
# import seaborn as sns
# from tqdm import tqdm
# from sklearn.metrics import mean_squared_error
# # import xgboost as xgb

# # Function to apply min-max scaling to a list of values
# def min_max_scale(values):
#     min_val = min(values)
#     max_val = max(values)
#     if max_val - min_val == 0:
#         return [1 for _ in values]  # If all values are the same, set them to 1
#     return [(v - min_val) / (max_val - min_val) for v in values]

# # Load uncertainty data from a pickle file
# def load_uncertainty_data(filepath):
#     with open(filepath, 'rb') as f:
#         return pickle.load(f)

# # Filter out results that do not contain required keys
# def filter_results(results):
#     # Including 'question_id' as a required key
#     required_keys = {'predictive_entropy', 'lexical_similarity', 'semantic_entropy', 'num_clusters', 'accuracy', 'question_id'}
#     return {key: value for key, value in results.items() if required_keys.issubset(value.keys())}

# # Extract the necessary metrics, including question_ids
# def extract_metrics(results):
#     predictive_entropy = []
#     lexical_similarity = []
#     semantic_entropy = []
#     semantic_clusters = []
#     accuracies = []
#     question_ids = []
    
#     for key, value in results.items():
#         # predictive_entropy.append(value["predictive_entropy"])
#         predictive_entropy.append(np.exp(value["predictive_entropy"]))
#         lexical_similarity.append(value["lexical_similarity"])
#         semantic_entropy.append(value["semantic_entropy"])
#         semantic_clusters.append(value["num_clusters"])  # Assuming 'num_clusters' represents semantic_clusters
#         accuracies.append(value["accuracy"])
#         # Extract question_id; if not present, use the key as question_id
#         # question_id = value.get("question_id", key)
#         question_ids.append(key)
            
#     return predictive_entropy, lexical_similarity, semantic_entropy, semantic_clusters, accuracies, question_ids

# def load_grounding_scores(root_dir_grounding):
#     grounding_files = os.listdir(root_dir_grounding)
#     question_grounding_scores_gdsam_dict = {}
#     question_grounding_scores_llama32_dict = {}
    
#     for file in grounding_files:
#         if not file.endswith('.pkl'):
#             continue
#         path = os.path.join(root_dir_grounding, file)
#         with open(path, 'rb') as f:
#             grounding_data = pickle.load(f)
#             question_id = grounding_data.get('question_id', [])
            
#             question_grounding_scores_gdsam_dict[str(question_id)] = grounding_data.get('grounding_gdsam_score', None)
#             question_grounding_scores_llama32_dict[str(question_id)] = grounding_data.get('grounding_llama32_score', None)
            
#     return question_grounding_scores_gdsam_dict, question_grounding_scores_llama32_dict
            

# # # Load grounding scores from grounding files and map them to question_ids
# # def load_grounding_scores(root_dir_grounding):
# #     grounding_files = os.listdir(root_dir_grounding)
# #     question_grounding_scores_dict = {}
    
# #     # for file in grounding_files:
# #     for file in tqdm(grounding_files, desc='Loading grounding scores'):
# #         if not file.endswith('.pkl'):
# #             continue  # Skip non-pickle files
# #         path = os.path.join(root_dir_grounding, file)
# #         grounding_score_list = []
        
# #         # Check if the file is empty
# #         if os.path.getsize(path) == 0:
# #             print(f"Skipping empty file: {file}")
# #             continue
        
# #         llama_32_scores = []
# #         with open(path, 'rb') as f:
# #             grounding_data = pickle.load(f)
# #             # question_id = grounding_data.get('question_id', [])[0]
# #             question_id = grounding_data.get('question_ids', [])[0] # for vqa
# #             # for slake dataset
# #             # question_id = grounding_data.get('qids', [])[0]
# #             for key in grounding_data.keys():
# #                 if 'response' in key:
# #                     response_key = key
# #                     response = grounding_data.get(response_key, {})
# #                     ## check if decoded output is '' or tokens less than 3
# #                     if 'decoded_output' in response:
# #                         decoded_output = response['decoded_output']
# #                         if len(decoded_output.split()) < 3:
# #                             continue
# #                     # if 'grounding_with_gd_score' in response:
# #                     #     # grounding_score = response['grounding_score']
# #                     #     grounding_score = response.get('grounding_with_gd_score', None)
# #                     #     grounding_score_list.append(grounding_score)
# #                     # else:
# #                     #     # Assign a default grounding score if missing
# #                     #     # grounding_score = 0
# #                     #     continue  # Skip this response if grounding score is missing
# #                     if 'llama_32_response' in response:
# #                         # grounding_score = response.get('llama_32_response', None)
# #                         # grounding_score_list.append(grounding_score)
# #                                     # Parse the textual verdict from Llama 3.2
# #                         llama_text = response.get('llama_32_response', None).lower().strip()
# #                         # The string might look like "yes.<|eot_id|>" or "no.<|eot_id|>"
# #                         # so we remove any special tokens and trailing punctuation
# #                         llama_text = llama_text.replace('.<|eot_id|>', '')
# #                         llama_text = llama_text.replace('<|eot_id|>', '')
# #                         llama_text = llama_text.strip()

# #                         # Convert to numeric scores
# #                         if llama_text.startswith("yes"):
# #                             llama_score = 1.0
# #                         # elif llama_text.startswith("no"):
# #                         #     llama_score = 0.0
# #                         # elif llama_text.startswith("not sure"):
# #                         #     # You can also treat “Not sure” as 0.0 if you prefer
# #                         #     llama_score = 0.5
# #                         else:
# #                             # If the model output is unexpected, treat as 0 or skip
# #                             llama_score = 0.0

# #                         llama_32_scores.append(llama_score)

# #         # Once we have processed all response_* keys, compute an average
# #         if llama_32_scores:
# #             average_score = sum(llama_32_scores) / len(llama_32_scores)
# #             question_grounding_scores_dict[str(question_id)] = average_score
# #         else:
# #             question_grounding_scores_dict[str(question_id)] = None  # Assign a default score
            
            
# #         # # question_grounding_scores_dict[question_id] = sum(grounding_score_list)/len(grounding_score_list) if len(grounding_score_list) > 0 else None
# #         # if grounding_score_list:
# #         #     average_score = sum(grounding_score_list) / len(grounding_score_list)
# #         #     question_grounding_scores_dict[str(question_id)] = average_score # stringifying question_id to match the format of question_ids
# #         # else:
# #         #     question_grounding_scores_dict[str(question_id)] = None  # Assign a default score
    
# #     return question_grounding_scores_dict

# #############################################################################
# # New helper function to calibrate a 2D combined metric using polynomial regression
# #############################################################################
# def calibrate_two_features(
#     X1_calib: np.ndarray,
#     X2_calib: np.ndarray,
#     y_calib: np.ndarray,
#     degree: int = 2,
#     alpha: float = 3.0
# ):
#     """
#     Learn a calibration mapping from two 1D features --> single confidence.
#     Returns a pipeline (PolynomialFeatures + Ridge).
#     """
#     # 1) Stack the two features horizontally: shape (N, 2)
#     X_calib_2d = np.column_stack([X1_calib, X2_calib])
    
#     # 2) Build the polynomial regression pipeline
#     model = make_pipeline(
#         PolynomialFeatures(degree=degree),
#         Ridge(alpha=alpha)
#     )
    
#     # 3) Fit to calibration set
#     model.fit(X_calib_2d, y_calib)
#     return model

# def apply_two_features_calibration(
#     model,
#     X1_test: np.ndarray,
#     X2_test: np.ndarray
# ) -> np.ndarray:
#     """
#     Apply the 2D calibration model to new data, producing calibrated 1D confidences.
#     """
#     X_test_2d = np.column_stack([X1_test, X2_test])
#     conf = model.predict(X_test_2d)
#     # Optionally min-max scale if you want final [0,1]
#     return min_max_scale(conf)


# # Fit polynomial regression and return coefficients
# def fit_polynomial_regression(X_val, y_val, degree=2, alpha=3.0):
#     model = make_pipeline(PolynomialFeatures(degree=degree), Ridge(alpha=alpha))
#     model.fit(X_val, y_val)
#     return model.named_steps['ridge'].coef_, model.named_steps['polynomialfeatures']

# # def fit_xgboost(X_val, y_val):
# #     model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, objective='reg:squarederror')
# #     model.fit(X_val, y_val)
# #     return model

# # Prepare the dataset for training and validation
# def split_dataset(predictive_entropy, lexical_similarity, semantic_entropy, semantic_clusters, grounding_scores, accuracies, val_percent=0.30, seed=42):
#     # Set random seed for reproducibility
#     random.seed(seed)
#     np.random.seed(seed)
    
#     n_samples = len(predictive_entropy)
#     indices = np.arange(n_samples)
#     np.random.shuffle(indices)
    
#     n_val = int(val_percent * n_samples)
#     val_indices = indices[:n_val]
#     test_indices = indices[n_val:]
    
#     # Calibration (validation) set
#     calibration = {
#         'predictive_entropy': np.array([predictive_entropy[i] for i in val_indices]),
#         'lexical_similarity': np.array([lexical_similarity[i] for i in val_indices]),
#         'semantic_entropy': np.array([semantic_entropy[i] for i in val_indices]),
#         'semantic_clusters': np.array([semantic_clusters[i] for i in val_indices]),
#         'grounding_score': np.array([grounding_scores[i] for i in val_indices]),
#         'accuracy': np.array([accuracies[i] for i in val_indices])
#     }
    
#     # Test set
#     test_set = {
#         'predictive_entropy': np.array([predictive_entropy[i] for i in test_indices]),
#         'lexical_similarity': np.array([lexical_similarity[i] for i in test_indices]),
#         'semantic_entropy': np.array([semantic_entropy[i] for i in test_indices]),
#         'semantic_clusters': np.array([semantic_clusters[i] for i in test_indices]),
#         'grounding_score': np.array([grounding_scores[i] for i in test_indices]),
#         'accuracy': np.array([accuracies[i] for i in test_indices])
#     }
    
#     return calibration, test_set

# # Calculate the new confidence values
# def calculate_confidence_values(X_test, alphas, poly_features):
#     X_test_poly = poly_features.transform(X_test)
#     new_confidence_values = np.dot(X_test_poly, alphas)
#     return min_max_scale(new_confidence_values)
#     # return new_confidence_values

# # Plot reliability diagram
# def plot_reliability_diagram(confidences, accuracies, confidence_type, save_path):
#     # Set the style to a built-in Matplotlib style
#     plt.style.use('ggplot')

#     # Create the figure and axis
#     fig, ax = plt.subplots(figsize=(10, 8))

#     # Calculate bin statistics
#     bins = np.linspace(0, 1, 11)
#     bin_indices = np.digitize(confidences, bins) - 1

#     avg_accuracies = []
#     bin_centers = []
#     for i in range(len(bins) - 1):
#         bin_mask = bin_indices == i
#         if bin_mask.any():
#             avg_accuracy = np.mean(accuracies[bin_mask])
#             avg_accuracies.append(avg_accuracy)
#             bin_centers.append((bins[i] + bins[i + 1]) / 2)
#         else:
#             print(f"No data found in bin {i}")

#     avg_accuracies = np.array(avg_accuracies)
#     bin_centers = np.array(bin_centers)

#     # Plot the reliability curve
#     ax.plot(bin_centers, avg_accuracies, marker='o', linestyle='-', linewidth=2, 
#             markersize=8, label=f'Reliability ({confidence_type})')

#     # Plot the perfect calibration line
#     ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2, label='Perfect Calibration')

#     # Fill the area between the curves
#     ax.fill_between(bin_centers, bin_centers, avg_accuracies, alpha=0.2)

#     # Set labels and title
#     ax.set_xlabel('Confidence', fontsize=14)
#     ax.set_ylabel('Accuracy', fontsize=14)
#     ax.set_title(f'Reliability Diagram ({confidence_type})', fontsize=16, fontweight='bold')

#     # Customize the grid
#     ax.grid(True, linestyle=':', alpha=0.7)

#     # Customize the legend
#     ax.legend(fontsize=12, loc='lower right')

#     # Set the aspect ratio to 'equal' for a square plot
#     ax.set_aspect('equal')

#     # Add textbox with statistics
#     ece = np.mean(np.abs(avg_accuracies - bin_centers))
#     mce = np.max(np.abs(avg_accuracies - bin_centers))
#     stats_text = f'ECE: {ece:.3f}\nMCE: {mce:.3f}'
#     ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
#             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

#     # Customize ticks
#     ax.tick_params(axis='both', which='major', labelsize=12)

#     # Tight layout and save
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close(fig)

# # Create a dictionary with scaled and calibrated results (Optional)
# def create_scaled_results(predictive_entropy, lexical_similarity, semantic_entropy, semantic_clusters, grounding_scores, new_confidences, accuracies):
#     scaled_results = {}
#     for i in range(len(predictive_entropy)):
#         scaled_results[i] = {
#             "predictive_entropy": predictive_entropy[i],
#             "lexical_similarity": lexical_similarity[i],
#             "semantic_entropy": semantic_entropy[i],
#             "semantic_clusters": semantic_clusters[i],
#             "grounding_score": grounding_scores[i],
#             'confidence_predictive_entropy': new_confidences['predictive_entropy'][i],
#             'confidence_lexical_similarity': new_confidences['lexical_similarity'][i],
#             'confidence_semantic_entropy': new_confidences['semantic_entropy'][i],
#             'confidence_semantic_clusters': new_confidences['semantic_clusters'][i],
#             'confidence_grounding_score': new_confidences['grounding_score'][i],
#             "accuracy": accuracies[i]
#         }
#     return scaled_results

# # # Main function to run the pipeline
# # def run_pipeline(uncertainty_filepath, grounding_root_dir, save_path_prefix, seed=42):
# #     # Load uncertainty data
# #     results = load_uncertainty_data(uncertainty_filepath)
    
# #     # # Filter results to ensure all required keys are present
# #     # filtered_results = filter_results(results)
# #     # if not filtered_results:
# #     #     print("No valid data found after filtering. Please check your data file.")
# #     #     return
    
# #     # def filter_results(results):
# #     #     # Exclude entries where any required metric is missing or None
# #     #     required_keys = {'predictive_entropy', 'lexical_similarity', 'semantic_entropy', 'num_clusters', 'accuracy'}
# #     #     return {qid: value for qid, value in results.items() if required_keys.issubset(value.keys()) and all(value[key] is not None for key in required_keys)}
    
# #     # results = filter_results(results)
# #     # Extract metrics
# #     predictive_entropy, lexical_similarity, semantic_entropy, semantic_clusters, accuracies, question_ids = extract_metrics(results)
    
# #     # Load grounding scores
# #     grounding_scores_mapping = load_grounding_scores(grounding_root_dir)
# #     print(f"Loaded {len(grounding_scores_mapping)} grounding scores.")
# #     # Map grounding scores to each sample based on question_id
# #     grounding_scores_gdsam = [grounding_scores_mapping[0].get(qid, None) for qid in question_ids]
# #     grounding_scores_llama32 = [grounding_scores_mapping[1].get(qid, None) for qid in question_ids]
    
# #     # remove None values from grounding scores
# #     predictive_entropy = [p for p, g in zip(predictive_entropy, grounding_scores_gdsam) if g is not None]
    
# #     lexical_similarity = [l for l, g in zip(lexical_similarity, grounding_scores_gdsam) if g is not None]
    
# #     semantic_entropy = [s for s, g in zip(semantic_entropy, grounding_scores_gdsam) if g is not None]
    
# #     semantic_clusters = [s for s, g in zip(semantic_clusters, grounding_scores_gdsam) if g is not None]
    
# #     accuracies = [a for a, g in zip(accuracies, grounding_scores_gdsam) if g is not None]
    
# #     grounding_scores_gdsam = [g for g in grounding_scores_gdsam if g is not None]
    
# #     assert len(predictive_entropy) == len(lexical_similarity) == len(semantic_entropy) == len(semantic_clusters) == len(grounding_scores_gdsam) == len(accuracies), \
# #         "Mismatch in the number of samples among metrics."
    
# #     # Scale all the metrics except accuracies
# #     predictive_entropy_scaled = min_max_scale(predictive_entropy)
# #     lexical_similarity_scaled = min_max_scale(lexical_similarity)
# #     semantic_entropy_scaled = min_max_scale(semantic_entropy)
# #     semantic_clusters_scaled = min_max_scale(semantic_clusters)
# #     # grounding_scores_scaled = min_max_scale(grounding_scores) # there are two now so we will scale them separately
# #     grounding_scores_gdsam_scaled = min_max_scale(grounding_scores_gdsam)
# #     grounding_scores_llama32_scaled = min_max_scale(grounding_scores_llama32)
    
# #     # grounding_scores_scaled = grounding_scores
# #     # grounding_scores_scaled = min_max_scale(grounding_scores)
# #     # accuracies = np.array(accuracies)
# #     # accuracies = min_max_scale(accuracies)
# #     # accuracies are binary and should not be scaled
    
# #     # map all the metrics to thee question_ids
# #     filtered_question_ids = [qid for qid, g in zip(question_ids, grounding_scores_gdsam) if g is not None]
    
# #     # results = zip(predictive_entropy_scaled, lexical_similarity_scaled, semantic_entropy_scaled, semantic_clusters_scaled, grounding_scores_scaled, accuracies)
    
# #     results_dict = dict(zip(filtered_question_ids, zip(predictive_entropy_scaled, lexical_similarity_scaled, semantic_entropy_scaled, semantic_clusters_scaled, grounding_scores_gdsam_scaled, grounding_scores_llama32_scaled, accuracies)))
    
    
# #     # map all the metrics to thee question_ids, tot 
# #     # Ensure that all metrics have the same length
# #     assert len(predictive_entropy_scaled) == len(lexical_similarity_scaled) == len(semantic_entropy_scaled) == len(semantic_clusters_scaled) == len(grounding_scores_gdsam_scaled) == len(accuracies), \
# #         "Mismatch in the number of samples among metrics."
    
# #     # Split dataset into calibration (30%) and test (70%)
# #     calibration, test_set = split_dataset(
# #         predictive_entropy_scaled, 
# #         lexical_similarity_scaled, 
# #         semantic_entropy_scaled, 
# #         semantic_clusters_scaled, 
# #         # grounding_scores_scaled, 
# #         grounding_scores_gdsam_scaled,
# #         grounding_scores_llama32_scaled,
# #         accuracies, 
# #         val_percent=0.25, 
# #         seed=seed
# #     )
    
# #     # get the question_ids for the calibration and test set
# #     calibration_question_ids = [qid for qid, g in zip(filtered_question_ids, grounding_scores_gdsam_scaled) if g in calibration['grounding_score_gdsam_scaled']]
# #     test_question_ids = [qid for qid, g in zip(filtered_question_ids, grounding_scores_gdsam_scaled) if g in test_set['grounding_score_gdsam_scaled']]
    
# #     # # map all the metrics to the test and calibration question_ids
# #     # calibration_results = {qid: results_dict[qid] for qid in calibration_question_ids}
# #     # test_results = {qid: results_dict[qid] for qid in test_question_ids}
    
    
# #     # List of original metrics and their corresponding calibration transformations
# #     metrics = {
# #         'predictive_entropy': {
# #             'X_calib': 1 - calibration['predictive_entropy'].reshape(-1, 1),
# #             'X_test': 1 - test_set['predictive_entropy'].reshape(-1, 1)
# #         },
# #         'lexical_similarity': {
# #             'X_calib': calibration['lexical_similarity'].reshape(-1, 1),
# #             'X_test': test_set['lexical_similarity'].reshape(-1, 1)
# #         },
# #         'semantic_entropy': {
# #             'X_calib': 1 - calibration['semantic_entropy'].reshape(-1, 1),
# #             'X_test': 1 - test_set['semantic_entropy'].reshape(-1, 1)
# #         },
# #         'semantic_clusters': {
# #             'X_calib': 1 - calibration['semantic_clusters'].reshape(-1, 1),
# #             'X_test': 1 - test_set['semantic_clusters'].reshape(-1, 1)
# #         },
# #         'grounding_score_gdsam': {
# #             'X_calib': calibration['grounding_score_gdsam'].reshape(-1, 1),
# #             'X_test': test_set['grounding_score_gdsam'].reshape(-1, 1)
# #         },
# #         'grounding_score_llama32': {
# #             'X_calib': calibration['grounding_score_llama32'].reshape(-1, 1),
# #             'X_test': test_set['grounding_score_llama32'].reshape(-1, 1)
# #         }
# #     }
    
    
# #     # Dictionaries to store calibration models and calibrated confidences
# #     calibration_models = {}
# #     calibrated_confidences = {}
    
# #     # Calibrate each metric
# #     for metric_name, data in metrics.items():
# #         X_calib = data['X_calib']
# #         y_calib = calibration['accuracy']
# #         alphas, poly_features = fit_polynomial_regression(X_calib, y_calib)
# #         # xgb_model = fit_xgboost(X_calib, y_calib)
# #         # Store the trained model for each metric
# #         # calibration_models[metric_name] = xgb_model
# #         calibration_models[metric_name] = (alphas, poly_features)
# #         calibrated_conf = calculate_confidence_values(X_calib, alphas, poly_features)
# #         # Predict the calibrated confidence values
# #         # calibrated_conf = xgb_model.predict(X_calib)
# #         calibrated_confidences[metric_name] = calibrated_conf
    
# #     # Apply calibration to test set and store calibrated confidences
# #     calibrated_test_conf = {}
# #     for metric_name, data in metrics.items():
# #         alphas, poly_features = calibration_models[metric_name]
# #         X_test = data['X_test']
# #         calibrated_conf = calculate_confidence_values(X_test, alphas, poly_features)
# #         calibrated_test_conf[metric_name] = calibrated_conf
    
# #     # map the calibrated confidences to the question_ids for each metric
# #     calibrated_test_results = {qid: [calibrated_test_conf[metric_name][i] for i in range(len(calibrated_test_conf[metric_name]))] for qid in test_question_ids for metric_name in metrics.keys()}
    
# #     # Uncalibrated confidences
# #     uncalib_conf = {}
# #     for metric_name, data in metrics.items():
# #         X_test = data['X_test']
# #         uncalib_conf[metric_name] = X_test.flatten()
    
# #     # Define combined metrics by adding grounding score to each original metric
# #     combined_metrics = {}
# #     for base_metric in ['predictive_entropy', 'lexical_similarity', 'semantic_entropy', 'semantic_clusters']:
# #         combined_metric_name = f"{base_metric}_with_grounding"
# #         combined_metrics[combined_metric_name] = {
# #             'combined_confidences': np.array(uncalib_conf[base_metric]) + np.array(uncalib_conf['grounding_score_gdsam']) 
# #         }
    
# #     # Scale combined metrics
# #     for combined_metric_name in combined_metrics:
# #         combined_metrics[combined_metric_name]['combined_confidences'] = min_max_scale(combined_metrics[combined_metric_name]['combined_confidences'])
    
# #     # Calibration for combined metrics
# #     combined_calibration_models = {}
# #     calibrated_combined_test_conf = {}
# #     combined_uncalib_conf = {}
    
# #     for combined_metric_name, data in combined_metrics.items():
# #         # Combine calibration confidences
# #         combined_X_calib = calibration['predictive_entropy'] + calibration['grounding_score'] if 'predictive_entropy_with_grounding' in combined_metric_name else \
# #                            calibration['lexical_similarity'] + calibration['grounding_score'] if 'lexical_similarity_with_grounding' in combined_metric_name else \
# #                            calibration['semantic_entropy'] + calibration['grounding_score'] if 'semantic_entropy_with_grounding' in combined_metric_name else \
# #                            calibration['semantic_clusters'] + calibration['grounding_score']
        
# #         combined_X_calib = combined_X_calib.reshape(-1, 1)
# #         y_calib = calibration['accuracy']
# #         alphas, poly_features = fit_polynomial_regression(combined_X_calib, y_calib)
# #         combined_calibration_models[combined_metric_name] = (alphas, poly_features)
# #         calibrated_conf = calculate_confidence_values(combined_X_calib, alphas, poly_features)
# #         combined_metrics[combined_metric_name]['calibrated_conf'] = calibrated_conf
        
# #         # Apply calibration to test set
# #         if 'predictive_entropy_with_grounding' in combined_metric_name:
# #             X_test_combined = test_set['predictive_entropy'] + test_set['grounding_score']
# #         elif 'lexical_similarity_with_grounding' in combined_metric_name:
# #             X_test_combined = test_set['lexical_similarity'] + test_set['grounding_score']
# #         elif 'semantic_entropy_with_grounding' in combined_metric_name:
# #             X_test_combined = test_set['semantic_entropy'] + test_set['grounding_score']
# #         elif 'semantic_clusters_with_grounding' in combined_metric_name:
# #             X_test_combined = test_set['semantic_clusters'] + test_set['grounding_score']
# #         else:
# #             X_test_combined = 0  # Default case, should not occur
        
# #         X_test_combined = X_test_combined.reshape(-1, 1)
# #         alphas, poly_features = combined_calibration_models[combined_metric_name]
# #         calibrated_conf = calculate_confidence_values(X_test_combined, alphas, poly_features)
# #         calibrated_combined_test_conf[combined_metric_name] = calibrated_conf
        
# #         # Uncalibrated combined confidences
# #         combined_uncalib_conf[combined_metric_name] = X_test_combined.flatten()
    
# #     # Plot reliability diagrams for original metrics
# #     for metric_name in metrics.keys():
# #         # Uncalibrated
# #         plot_reliability_diagram(
# #             uncalib_conf[metric_name], 
# #             # test_accuracies, 
# #             test_set['accuracy'],
# #             f"{metric_name.replace('_', ' ').title()} (Uncalibrated)", 
# #             f"{save_path_prefix}_{metric_name}_uncalibrated.png"
# #         )
# #         # Calibrated
# #         plot_reliability_diagram(
# #             calibrated_test_conf[metric_name], 
# #             # test_accuracies, 
# #             test_set['accuracy'],
# #             f"{metric_name.replace('_', ' ').title()} (Calibrated)", 
# #             f"{save_path_prefix}_{metric_name}_calibrated.png"
# #         )
    
# #     # Plot reliability diagrams for combined metrics
# #     for combined_metric_name in combined_metrics.keys():
# #         # Uncalibrated
# #         plot_reliability_diagram(
# #             combined_uncalib_conf[combined_metric_name], 
# #             # test_accuracies, 
# #             test_set['accuracy'],
# #             f"{combined_metric_name.replace('_', ' ').title()} (Uncalibrated)", 
# #             f"{save_path_prefix}_{combined_metric_name}_uncalibrated.png"
# #         )
# #         # Calibrated
# #         plot_reliability_diagram(
# #             calibrated_combined_test_conf[combined_metric_name], 
# #             # test_accuracies, 
# #             test_set['accuracy'],
# #             f"{combined_metric_name.replace('_', ' ').title()} (Calibrated)", 
# #             f"{save_path_prefix}_{combined_metric_name}_calibrated.png"
# #         )
    
# #     print(f"Reliability diagrams saved with prefix '{save_path_prefix}'")

# def calibrate_multi_features(
#     X_calib: np.ndarray,
#     y_calib: np.ndarray,
#     degree: int = 2,
#     alpha: float = 3.0
# ):
#     """
#     Fits a polynomial regression model (PolynomialFeatures + Ridge)
#     to map N-dimensional features -> accuracy in [0,1].
    
#     X_calib: shape (N, d), where d >= 2 (2D, 3D, etc.)
#     y_calib: shape (N,)
#     """
#     from sklearn.preprocessing import PolynomialFeatures
#     from sklearn.linear_model import Ridge
#     from sklearn.pipeline import make_pipeline

#     model = make_pipeline(
#         PolynomialFeatures(degree=degree),
#         Ridge(alpha=alpha)
#     )
#     model.fit(X_calib, y_calib)
#     return model

# def apply_multi_features_calibration(
#     model,
#     X_test: np.ndarray
# ) -> np.ndarray:
#     """
#     Applies the fitted polynomial pipeline to new data X_test (shape (M, d)),
#     returning a single 1D array of calibrated confidences in [0,1].
#     """
#     conf = model.predict(X_test)  # shape (M,)
#     return min_max_scale(conf)


# # def run_pipeline(uncertainty_filepath, grounding_root_dir, save_path_prefix, seed=42):
# #     # 1) Load your data
# #     results = load_uncertainty_data(uncertainty_filepath)
# #     # 2) Extract baseline metrics
# #     pred_ent, lex_sim, sem_ent, sem_clust, accuracies, question_ids = extract_metrics(results)

# #     # 3) Load your grounding scores
# #     grounding_scores_mapping = load_grounding_scores(grounding_root_dir)
# #     gdsam_scores   = [grounding_scores_mapping[0].get(qid, None) for qid in question_ids]
# #     llama32_scores = [grounding_scores_mapping[1].get(qid, None) for qid in question_ids]

# #     # 4) Filter out entries where GDSAM is None
# #     #    (You can do a more advanced filter if you want to keep partial data.)
# #     pred_ent = [p for p, g in zip(pred_ent, gdsam_scores) if g is not None]
# #     lex_sim  = [l for l, g in zip(lex_sim,  gdsam_scores) if g is not None]
# #     sem_ent  = [s for s, g in zip(sem_ent,  gdsam_scores) if g is not None]
# #     sem_clust= [s for s, g in zip(sem_clust,gdsam_scores) if g is not None]
# #     accuracies = [a for a, g in zip(accuracies, gdsam_scores) if g is not None]
# #     gdsam_scores   = [g for g in gdsam_scores   if g is not None]

# #     # NOTE: We haven't filtered on llama32 yet. 
# #     # If you need to skip samples where llama32 is None, do similarly.
# #     # For demonstration, we keep them as is (some might be None => treat as 0?)
# #     # But to keep it consistent, let's do a quick check:
# #     llama32_scores = [0.0 if (s is None) else s for s in llama32_scores]

# #     # 5) Scale
# #     pred_ent_scaled    = min_max_scale(pred_ent)
# #     lex_sim_scaled     = min_max_scale(lex_sim)
# #     sem_ent_scaled     = min_max_scale(sem_ent)
# #     sem_clust_scaled   = min_max_scale(sem_clust)
# #     gdsam_scaled       = min_max_scale(gdsam_scores)
# #     llama32_scaled     = min_max_scale(llama32_scores)

# #     # 6) Split into calibration/test
# #     calibration, test_set = split_dataset(
# #         pred_ent_scaled,
# #         lex_sim_scaled,
# #         sem_ent_scaled,
# #         sem_clust_scaled,
# #         gdsam_scaled,
# #         accuracies,
# #         val_percent=0.25,
# #         seed=seed
# #     )

# #     # 7) Single-metric dictionary 
# #     #    (We do 1 - pred_ent, etc. as you had before)
# #     metrics = {
# #         "predictive_entropy": {
# #             "X_calib": 1 - calibration["predictive_entropy"].reshape(-1, 1),
# #             "X_test":  1 - test_set["predictive_entropy"].reshape(-1, 1)
# #         },
# #         "lexical_similarity": {
# #             "X_calib": calibration["lexical_similarity"].reshape(-1, 1),
# #             "X_test":  test_set["lexical_similarity"].reshape(-1, 1)
# #         },
# #         "semantic_entropy": {
# #             "X_calib": 1 - calibration["semantic_entropy"].reshape(-1, 1),
# #             "X_test":  1 - test_set["semantic_entropy"].reshape(-1, 1)
# #         },
# #         "semantic_clusters": {
# #             "X_calib": 1 - calibration["semantic_clusters"].reshape(-1, 1),
# #             "X_test":  1 - test_set["semantic_clusters"].reshape(-1, 1)
# #         },
# #         "grounding_score_gdsam": {
# #             "X_calib": calibration["grounding_score"].reshape(-1, 1),
# #             "X_test":  test_set["grounding_score"].reshape(-1, 1)
# #         }
# #         # If you want single-metric llama32, do it similarly
# #     }

# #     # 8) Single-metric calibration 
# #     calibration_models = {}
# #     calibrated_test_conf = {}
# #     for metric_name, data in metrics.items():
# #         X_calib = data["X_calib"]
# #         y_calib = calibration["accuracy"]
# #         alphas, poly_features = fit_polynomial_regression(X_calib, y_calib)
# #         calibration_models[metric_name] = (alphas, poly_features)

# #         # Apply to test
# #         X_test = data["X_test"]
# #         test_conf = calculate_confidence_values(X_test, alphas, poly_features)
# #         calibrated_test_conf[metric_name] = test_conf

# #     # 9) Uncalibrated dict (for reliability diagrams)
# #     uncalib_conf = {}
# #     for metric_name, data in metrics.items():
# #         X_test = data["X_test"]
# #         uncalib_conf[metric_name] = X_test.flatten()

# #     # 10) Plot reliability diagrams (single metrics)
# #     for metric_name in metrics.keys():
# #         # Uncalibrated
# #         plot_reliability_diagram(
# #             uncalib_conf[metric_name],
# #             test_set["accuracy"],
# #             f"{metric_name} (Uncalibrated)",
# #             f"{save_path_prefix}_{metric_name}_uncalibrated.png"
# #         )
# #         # Calibrated
# #         plot_reliability_diagram(
# #             calibrated_test_conf[metric_name],
# #             test_set["accuracy"],
# #             f"{metric_name} (Calibrated)",
# #             f"{save_path_prefix}_{metric_name}_calibrated.png"
# #         )

# #     #################################################################
# #     ### 2D COMBINATIONS
# #     #################################################################
# #     # A) predictive_entropy + grounding_score_gdsam
# #     print("\n--- 2D Calibration: Predictive Entropy & GDSAM ---")
# #     x1_calib = metrics["predictive_entropy"]["X_calib"].flatten()
# #     x2_calib = metrics["grounding_score_gdsam"]["X_calib"].flatten()
# #     y_calib  = calibration["accuracy"]
# #     model_2d_gdsam = calibrate_multi_features(
# #         np.column_stack([x1_calib, x2_calib]),
# #         y_calib,
# #         degree=2,
# #         alpha=3.0
# #     )
# #     # Apply to test
# #     x1_test = metrics["predictive_entropy"]["X_test"].flatten()
# #     x2_test = metrics["grounding_score_gdsam"]["X_test"].flatten()
# #     conf_2d_gdsam = apply_multi_features_calibration(
# #         model_2d_gdsam,
# #         np.column_stack([x1_test, x2_test])
# #     )
# #     plot_reliability_diagram(
# #         conf_2d_gdsam,
# #         test_set["accuracy"],
# #         "PredictiveEntropy + GDSAM (2D Calibrated)",
# #         f"{save_path_prefix}_predEnt_plus_gdsam_2dCal.png"
# #     )

# #     # B) predictive_entropy + llama32
# #     # (First, ensure we have llama32 in the same structure as metrics => or we do direct approach)
# #     print("\n--- 2D Calibration: Predictive Entropy & Llama32 ---")
# #     # We'll treat the calibration data for llama32 as: 
# #     # "X_calib_llama32" = (the same split as everything else)
# #     # so we'll need to do a *manual* split for llama32 or incorporate it into `split_dataset`.
# #     # For brevity, let's assume you already have arrays: 
# #     # llama32_calib, llama32_test
# #     # We'll do it quickly:

# #     # Just quickly create arrays from the *scaled* llama32:
# #     # (the length is the same as pred_ent, so we can do the same indexing)
# #     # We'll replicate how your `split_dataset` does the indexing:
# #     num_samples = len(pred_ent_scaled)
# #     indices = np.arange(num_samples)
# #     random.seed(seed); np.random.seed(seed)
# #     np.random.shuffle(indices)
# #     n_val = int(0.25 * num_samples)
# #     val_indices = indices[:n_val]
# #     test_indices = indices[n_val:]

# #     llama32_calib = np.array([llama32_scaled[i] for i in val_indices])
# #     llama32_test  = np.array([llama32_scaled[i] for i in test_indices])

# #     x1_calib_llama = metrics["predictive_entropy"]["X_calib"].flatten()
# #     x2_calib_llama = llama32_calib
# #     model_2d_llama = calibrate_multi_features(
# #         np.column_stack([x1_calib_llama, x2_calib_llama]),
# #         y_calib,  # same
# #         degree=2,
# #         alpha=3.0
# #     )
# #     x1_test_llama = metrics["predictive_entropy"]["X_test"].flatten()
# #     x2_test_llama = llama32_test
# #     conf_2d_llama = apply_multi_features_calibration(
# #         model_2d_llama,
# #         np.column_stack([x1_test_llama, x2_test_llama])
# #     )
# #     plot_reliability_diagram(
# #         conf_2d_llama,
# #         test_set["accuracy"],
# #         "PredictiveEntropy + Llama32 (2D Calibrated)",
# #         f"{save_path_prefix}_predEnt_plus_llama32_2dCal.png"
# #     )

# #     #################################################################
# #     ### 3D COMBINATION (predictive_entropy, gdsam, llama32)
# #     #################################################################
# #     print("\n--- 3D Calibration: Predictive Entropy & GDSAM & Llama32 ---")

# #     # 1) Build calibration features (N,3)
# #     #    x1 = predEntropy, x2 = GDSAM, x3 = Llama32
# #     x1_calib_3d = metrics["predictive_entropy"]["X_calib"].flatten()
# #     x2_calib_3d = metrics["grounding_score_gdsam"]["X_calib"].flatten()
# #     x3_calib_3d = llama32_calib
# #     X_calib_3d  = np.column_stack([x1_calib_3d, x2_calib_3d, x3_calib_3d])

# #     # 2) Fit
# #     model_3d = calibrate_multi_features(X_calib_3d, y_calib, degree=2, alpha=3.0)

# #     # 3) Apply to test
# #     x1_test_3d = metrics["predictive_entropy"]["X_test"].flatten()
# #     x2_test_3d = metrics["grounding_score_gdsam"]["X_test"].flatten()
# #     x3_test_3d = llama32_test
# #     X_test_3d  = np.column_stack([x1_test_3d, x2_test_3d, x3_test_3d])

# #     conf_3d = apply_multi_features_calibration(model_3d, X_test_3d)

# #     # 4) Plot
# #     plot_reliability_diagram(
# #         conf_3d,
# #         test_set["accuracy"],
# #         "PredictiveEntropy + GDSAM + Llama32 (3D Calibrated)",
# #         f"{save_path_prefix}_predEnt_plus_gdsam_plus_llama32_3dCal.png"
# #     )

# #     print(f"\nReliability diagrams saved with prefix '{save_path_prefix}'")

# def run_pipeline(uncertainty_filepath, grounding_root_dir, save_path_prefix, seed=42):
#     # 1) Load your baseline uncertainty data
#     results = load_uncertainty_data(uncertainty_filepath)
    
#     # 2) Extract metrics
#     #    => predictive_entropy, lexical_similarity, semantic_entropy, semantic_clusters, accuracies, question_ids
#     pred_ent, lex_sim, sem_ent, sem_clust, accuracies, question_ids = extract_metrics(results)

#     # 3) Load your grounding scores from the pickles
#     #    => (dict_for_gdsam, dict_for_llama32)
#     grounding_scores_mapping = load_grounding_scores(grounding_root_dir)
#     gdsam_scores   = [grounding_scores_mapping[0].get(qid, None) for qid in question_ids]
#     llama32_scores = [grounding_scores_mapping[1].get(qid, None) for qid in question_ids]
#     print(f"Loaded GDSAM & LLAMA32 grounding scores for {len(grounding_scores_mapping[0])} questions.")

#     # 4) Filter out entries where GDSAM is None (for a consistent set).
#     pred_ent_filtered     = []
#     lex_sim_filtered      = []
#     sem_ent_filtered      = []
#     sem_clust_filtered    = []
#     accuracies_filtered   = []
#     gdsam_filtered        = []
#     llama32_filtered      = []

#     for (pe, ls, se, sc, acc, gd, ll) in zip(
#         pred_ent, lex_sim, sem_ent, sem_clust, accuracies, gdsam_scores, llama32_scores
#     ):
#         if gd is not None:
#             pred_ent_filtered.append(pe)
#             lex_sim_filtered.append(ls)
#             sem_ent_filtered.append(se)
#             sem_clust_filtered.append(sc)
#             accuracies_filtered.append(acc)
#             gdsam_filtered.append(gd)
#             # For LLAMA32, if it's None, we can treat it as 0 or also skip
#             llama32_filtered.append(ll if ll is not None else 0.0)

#     # 5) Scale everything except accuracies (0/1)
#     pred_ent_scaled   = min_max_scale(pred_ent_filtered)
#     lex_sim_scaled    = min_max_scale(lex_sim_filtered)
#     sem_ent_scaled    = min_max_scale(sem_ent_filtered)
#     sem_clust_scaled  = min_max_scale(sem_clust_filtered)
#     gdsam_scaled      = min_max_scale(gdsam_filtered)
#     llama32_scaled    = min_max_scale(llama32_filtered)
#     accuracies_final  = accuracies_filtered  # Typically [0 or 1]

#     # 6) Use your split_dataset to get calibration & test
#     calibration, test_set = split_dataset(
#         pred_ent_scaled,
#         lex_sim_scaled,
#         sem_ent_scaled,
#         sem_clust_scaled,
#         gdsam_scaled,
#         accuracies_final,
#         val_percent=0.25,
#         seed=seed
#     )

#     # 7) Build your single metrics dictionary
#     metrics = {
#         "predictive_entropy": {
#             "X_calib": 1 - calibration["predictive_entropy"].reshape(-1, 1),
#             "X_test":  1 - test_set["predictive_entropy"].reshape(-1, 1)
#         },
#         "lexical_similarity": {
#             "X_calib": calibration["lexical_similarity"].reshape(-1, 1),
#             "X_test":  test_set["lexical_similarity"].reshape(-1, 1)
#         },
#         "semantic_entropy": {
#             "X_calib": 1 - calibration["semantic_entropy"].reshape(-1, 1),
#             "X_test":  1 - test_set["semantic_entropy"].reshape(-1, 1)
#         },
#         "semantic_clusters": {
#             "X_calib": 1 - calibration["semantic_clusters"].reshape(-1, 1),
#             "X_test":  1 - test_set["semantic_clusters"].reshape(-1, 1)
#         },
#         "grounding_score_gdsam": {
#             "X_calib": calibration["grounding_score"].reshape(-1, 1),
#             "X_test":  test_set["grounding_score"].reshape(-1, 1)
#         }
#         # If you want single-metric llama32, 
#         # you'd integrate it similarly by adding "grounding_score_llama32": ...
#         # ""
#     }

#     # 8) Single-metric calibration
#     calibration_models = {}
#     calibrated_test_conf = {}
#     for metric_name, data in metrics.items():
#         X_calib = data["X_calib"]
#         y_calib = calibration["accuracy"]
#         alphas, poly_features = fit_polynomial_regression(X_calib, y_calib)
#         calibration_models[metric_name] = (alphas, poly_features)

#         # Calibrated confidence on the test set
#         X_test = data["X_test"]
#         conf_test = calculate_confidence_values(X_test, alphas, poly_features)
#         calibrated_test_conf[metric_name] = conf_test

#     # 9) Build dict of uncalibrated test for reliability diagrams
#     uncalib_conf = {}
#     for metric_name, data in metrics.items():
#         uncalib_conf[metric_name] = data["X_test"].flatten()

#     # 10) Plot single-metric reliability
#     for metric_name in metrics.keys():
#         # Uncalibrated
#         plot_reliability_diagram(
#             uncalib_conf[metric_name],
#             test_set["accuracy"],
#             f"{metric_name} (Uncalibrated)",
#             f"{save_path_prefix}_{metric_name}_uncalibrated.png"
#         )
#         # Calibrated
#         plot_reliability_diagram(
#             calibrated_test_conf[metric_name],
#             test_set["accuracy"],
#             f"{metric_name} (Calibrated)",
#             f"{save_path_prefix}_{metric_name}_calibrated.png"
#         )

#     ##########################################################################
#     #                   2D CALIBRATIONS (4 baselines x 2 groundings)
#     ##########################################################################
#     # We'll define an easy helper to get the test/calib arrays for each baseline
#     def get_calib_test_arrays(metric_dict, metric_name):
#         return (metric_dict[metric_name]["X_calib"].flatten(),
#                 metric_dict[metric_name]["X_test"].flatten())

#     # Our four baseline metrics
#     baseline_list = ["predictive_entropy", "lexical_similarity", "semantic_entropy", "semantic_clusters"]

#     # We'll do 2D with GDSAM, 
#     # but we also do 2D with LLAMA32 => we must manually create arrays for LLAMA32 in calibration/test sets
#     # let's do that similarly to how we do GDSAM.

#     ### Build llama32 arrays for calibration & test (like in step 6).
#     # We'll replicate the indexing approach from split_dataset
#     num_samples = len(pred_ent_scaled)
#     indices = np.arange(num_samples)
#     random.seed(seed); np.random.seed(seed)
#     np.random.shuffle(indices)
#     n_val = int(0.25 * num_samples)
#     val_indices = indices[:n_val]
#     test_indices = indices[n_val:]
#     llama32_calib = np.array([llama32_scaled[i] for i in val_indices])
#     llama32_test  = np.array([llama32_scaled[i] for i in test_indices])
#     y_calib = calibration["accuracy"]
#     y_test  = test_set["accuracy"]

#     # A) 2D with GDSAM
#     for base_metric in baseline_list:
#         base_calib, base_test = get_calib_test_arrays(metrics, base_metric)
#         gdsam_calib, gdsam_test = get_calib_test_arrays(metrics, "grounding_score_gdsam")

#         combined_name = f"{base_metric}_2D_GDSAM"
#         print(f"\n2D calibration: {base_metric} + GDSAM")

#         # shape (N,2)
#         X_calib_2d = np.column_stack([base_calib, gdsam_calib])
#         model_2d = calibrate_multi_features(X_calib_2d, y_calib, degree=2, alpha=3.0)

#         # test
#         X_test_2d = np.column_stack([base_test, gdsam_test])
#         conf_2d = apply_multi_features_calibration(model_2d, X_test_2d)

#         # Plot
#         plot_reliability_diagram(
#             conf_2d,
#             y_test,
#             f"{combined_name} (2D Calibrated)",
#             f"{save_path_prefix}_{combined_name}_calibrated.png"
#         )

#     # B) 2D with Llama32
#     for base_metric in baseline_list:
#         base_calib, base_test = get_calib_test_arrays(metrics, base_metric)

#         combined_name = f"{base_metric}_2D_LLAMA32"
#         print(f"\n2D calibration: {base_metric} + Llama32")

#         # shape (N,2)
#         X_calib_2d = np.column_stack([base_calib, llama32_calib])
#         model_2d = calibrate_multi_features(X_calib_2d, y_calib, degree=2, alpha=3.0)

#         # test
#         X_test_2d = np.column_stack([base_test, llama32_test])
#         conf_2d = apply_multi_features_calibration(model_2d, X_test_2d)

#         # Plot
#         plot_reliability_diagram(
#             conf_2d,
#             y_test,
#             f"{combined_name} (2D Calibrated)",
#             f"{save_path_prefix}_{combined_name}_calibrated.png"
#         )

#     ##########################################################################
#     #             3D CALIBRATIONS (4 baselines x GDSAM + Llama32)
#     ##########################################################################
#     for base_metric in baseline_list:
#         base_calib, base_test = get_calib_test_arrays(metrics, base_metric)
#         gdsam_calib, gdsam_test = get_calib_test_arrays(metrics, "grounding_score_gdsam")

#         combined_name_3d = f"{base_metric}_3D_GDSAM_LLAMA32"
#         print(f"\n3D calibration: {base_metric} + GDSAM + Llama32")

#         X_calib_3d = np.column_stack([base_calib, gdsam_calib, llama32_calib])
#         model_3d = calibrate_multi_features(X_calib_3d, y_calib, degree=2, alpha=3.0)

#         # test
#         X_test_3d = np.column_stack([base_test, gdsam_test, llama32_test])
#         conf_3d = apply_multi_features_calibration(model_3d, X_test_3d)

#         plot_reliability_diagram(
#             conf_3d,
#             y_test,
#             f"{combined_name_3d} (3D Calibrated)",
#             f"{save_path_prefix}_{combined_name_3d}_calibrated.png"
#         )

#     print(f"\nAll reliability diagrams saved with prefix '{save_path_prefix}'.")

# # Example usage
# if __name__ == "__main__":
#     # Replace the following paths with your actual file paths
#     # uncertainty_filepath = '/home/ubuntu/Multimodal-Uncertainty-Quantification/runs/llava_gqa_yes_gsam_grounding_random_100/uncertainty/uncertainty_scores_baseline.pkl'
#     # uncertainty_filepath = '/home/ec2-user/Multimodal-Uncertainty-Quantification/runs/uncertainty/uncertainty_scores_baseline.pkl'
#     # grounding_root_dir = '/mnt/myebsvolume/home/ubuntu/Multimodal-Uncertainty-Quantification/runs/llava_gqa_yes_gsam_grounding_random_10000/grounding/'
    
#     ### *** CHANGE THE FOLLOWING PATHS *** ### -- below is for gqa dataset
#     # uncertainty_filepath = '/mnt/data/home/ubuntu/Multimodal-Uncertainty-Quantification/runs/llava_gqa_yes_gsam_grounding_random_1000_test_17nov/uncertainty/uncertainty_scores_baseline.pkl'
#     # grounding_root_dir = '/mnt/data/home/ubuntu/Multimodal-Uncertainty-Quantification/runs/llava_gqa_yes_gsam_grounding_random_1000_test_17nov/grounding_test/'
    
#     ### for slake dataset
    
#     # uncertainty_filepath = '/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/runs_slake/llava_med_slake/uncertainty/uncertainty_scores_baseline.pkl'
#     # grounding_root_dir_llama32 = '/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/runs_slake/llava_med_slake/grounding_with_llama32'
#     # ### for vqa dataset
#     uncertainty_filepath = '/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/runs_vqa6/llava_vqa_yes_gsam_grounding_random_1000_temp_05/uncertainty/uncertainty_scores_baseline.pkl'
#     grounding_root_dir = '/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/runs_vqa6/llava_vqa_yes_gsam_grounding_random_1000_temp_05/grounding'
#     grounding_root_dir_llama32 ='/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/runs_vqa6/llava_vqa_yes_gsam_grounding_random_1000_temp_05/grounding_with_llama32'
#     grounding_root_dir_gdsam_llama32 = '/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/runs_vqa6/llava_vqa_yes_gsam_grounding_random_1000_temp_05/grounding_with_gdsam_llama32'
#     save_path_prefix = "reliability_diagram_VQA"
#     seed = random.randint(1, 1000)
#     print(f"Using random seed: {seed}")
#     # run_pipeline(uncertainty_filepath, grounding_root_dir_llama32, save_path_prefix, seed)
#     run_pipeline(uncertainty_filepath, grounding_root_dir_gdsam_llama32, save_path_prefix, seed)

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

#############################################################################
# UTILITY FUNCTIONS
#############################################################################

def make_output_dir(out_dir: str):
    """
    Create an output directory if it doesn't exist.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

def min_max_scale(values):
    """
    Apply min-max scaling to a list of values, returning scaled values in [0,1].
    If the range is 0, returns all 1s.
    """
    min_val = min(values)
    max_val = max(values)
    if max_val - min_val == 0:
        return [1 for _ in values]
    return [(v - min_val) / (max_val - min_val) for v in values]

def load_uncertainty_data(filepath):
    """
    Loads a pickled dictionary of question -> {
        'predictive_entropy': float,
        'lexical_similarity': float,
        'semantic_entropy': float,
        'num_clusters': float,
        'accuracy': float,
        ...
    }
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def filter_results(results):
    """
    Optional: Filter out results that do not contain required keys.
    """
    required_keys = {
        'predictive_entropy', 'lexical_similarity',
        'semantic_entropy', 'num_clusters', 'accuracy'
    }
    filtered = {}
    for qid, val in results.items():
        if required_keys.issubset(val.keys()):
            filtered[qid] = val
    return filtered

def extract_metrics(results):
    """
    Return 6 lists: predictive_entropy, lexical_similarity, semantic_entropy, semantic_clusters, accuracies, question_ids
    """
    predictive_entropy = []
    lexical_similarity = []
    semantic_entropy = []
    semantic_clusters = []
    accuracies = []
    question_ids = []

    for qid, val in results.items():
        # Using np.exp(...) if your 'predictive_entropy' was stored in log form
        pe = np.exp(val["predictive_entropy"])  
        ls = val["lexical_similarity"]
        se = val["semantic_entropy"]
        sc = val["num_clusters"]
        ac = val["accuracy"]

        predictive_entropy.append(pe)
        lexical_similarity.append(ls)
        semantic_entropy.append(se)
        semantic_clusters.append(sc)
        accuracies.append(ac)
        question_ids.append(qid)

    return predictive_entropy, lexical_similarity, semantic_entropy, semantic_clusters, accuracies, question_ids

def load_grounding_scores(root_dir_grounding):
    """
    Returns two dicts:
      dict_gdsam[qid_str] -> float or None
      dict_llama32[qid_str] -> float or None

    Each .pkl presumably has:
      {
          "question_id": <string or int>,
          "grounding_gdsam_score": <float>,
          "grounding_llama32_score": <float>,
          ...
      }
    """
    files = sorted([f for f in os.listdir(root_dir_grounding) if f.endswith('.pkl')])
    dict_gdsam = {}
    dict_llama32 = {}
    for fname in files:
        path = os.path.join(root_dir_grounding, fname)
        # skip empty
        if os.path.getsize(path) == 0:
            continue
        with open(path, 'rb') as f:
            data = pickle.load(f)
        qid = str(data.get('question_id', ''))
        if not qid:
            continue  # skip if no question_id

        dict_gdsam[qid]   = data.get('grounding_clip_score', None)
        dict_llama32[qid] = data.get('grounding_llama32_score', None)

    return dict_gdsam, dict_llama32

def fit_polynomial_regression(X_val, y_val, degree=2, alpha=3.0):
    """
    Single-feature polynomial calibration
    """
    pipe = make_pipeline(
        PolynomialFeatures(degree=degree),
        Ridge(alpha=alpha)
    )
    pipe.fit(X_val, y_val)
    # We'll return the final steps so we can re-apply them
    # We'll store the "named_steps" to do the transform
    return pipe.named_steps['ridge'].coef_, pipe.named_steps['polynomialfeatures'], pipe

def calculate_confidence_values(X_test, alphas, poly_features):
    """
    For single-feature calibration: X_test shape (N,1).
    alphas shape (some_larger_dim, ), poly_features = fitted poly transform
    """
    X_test_poly = poly_features.transform(X_test)
    # Dot product
    y = np.dot(X_test_poly, alphas)
    return min_max_scale(y)

def split_dataset(pred_ent, lex_sim, sem_ent, sem_clust, grounding_scores, accuracies, val_percent=0.25, seed=42):
    """
    Returns calibration dict, test_set dict
    """
    random.seed(seed)
    np.random.seed(seed)
    n = len(pred_ent)
    idx = np.arange(n)
    np.random.shuffle(idx)

    n_val = int(val_percent * n)
    val_idx = idx[:n_val]
    tst_idx = idx[n_val:]

    calibration = {
        'predictive_entropy': np.array([pred_ent[i] for i in val_idx]),
        'lexical_similarity': np.array([lex_sim[i]  for i in val_idx]),
        'semantic_entropy':   np.array([sem_ent[i] for i in val_idx]),
        'semantic_clusters':  np.array([sem_clust[i] for i in val_idx]),
        'grounding_score':    np.array([grounding_scores[i] for i in val_idx]),
        'accuracy':           np.array([accuracies[i] for i in val_idx])
    }
    test_set = {
        'predictive_entropy': np.array([pred_ent[i] for i in tst_idx]),
        'lexical_similarity': np.array([lex_sim[i]  for i in tst_idx]),
        'semantic_entropy':   np.array([sem_ent[i] for i in tst_idx]),
        'semantic_clusters':  np.array([sem_clust[i] for i in tst_idx]),
        'grounding_score':    np.array([grounding_scores[i] for i in tst_idx]),
        'accuracy':           np.array([accuracies[i] for i in tst_idx])
    }
    return calibration, test_set

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

    ax.plot(bin_centers, avg_accuracies, marker='o', linestyle='-', linewidth=2, label=f'{confidence_type}')
    ax.plot([0,1],[0,1], '--', color='gray', label='Perfect')

    ax.fill_between(bin_centers, bin_centers, avg_accuracies, alpha=0.1)
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability Diagram ({confidence_type})")
    ax.legend(loc='lower right')

    # ECE + MCE
    if len(bin_centers) > 0:
        ece = np.mean(np.abs(avg_accuracies - bin_centers))
        mce = np.max(np.abs(avg_accuracies - bin_centers))
    else:
        ece, mce = 0,0
    stats_text = f"ECE: {ece:.3f}\nMCE: {mce:.3f}"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Saved reliability diagram => {save_path}")

#############################################################################
# MULTI-FEATURE CALIBRATION (2D, 3D, ETC.)
#############################################################################
def calibrate_multi_features(X_calib, y_calib, degree=2, alpha=3.0):
    """
    Fits polynomial regression to map N-dim features -> accuracy
    """
    pipe = make_pipeline(
        PolynomialFeatures(degree=degree),
        Ridge(alpha=alpha)
    )
    pipe.fit(X_calib, y_calib)
    return pipe

def apply_multi_features_calibration(model, X_test):
    """
    Returns a 1D confidence in [0,1]
    """
    raw = model.predict(X_test)
    return min_max_scale(raw)

#############################################################################
# MAIN PIPELINE
#############################################################################

def run_pipeline(uncertainty_filepath, grounding_root_dir, save_path_prefix, seed=42):
    # 1) Create an output directory for reliability diagrams
    out_dir = "reliability_plots"
    make_output_dir(out_dir)

    # 2) Load your baseline uncertainty data
    print(f"Loading uncertainty data from: {uncertainty_filepath}")
    results = load_uncertainty_data(uncertainty_filepath)
    print(f"Total entries in results: {len(results)}")

    # (Optional) filter if you want
    # results = filter_results(results)

    # 3) Extract baseline metrics
    pred_ent, lex_sim, sem_ent, sem_clust, accuracies, question_ids = extract_metrics(results)
    print("[INFO] Extracted: predictive_entropy, lexical_similarity, semantic_entropy, semantic_clusters, accuracies.")
    print(f"[INFO] Total {len(pred_ent)} samples after extraction.")

    # 4) Load GDSAM & LLAMA3.2
    print(f"[INFO] Loading grounding from: {grounding_root_dir}")
    dict_gdsam, dict_llama32 = load_grounding_scores(grounding_root_dir)
    print(f"[INFO] GDSAM dict size: {len(dict_gdsam)}, Llama32 dict size: {len(dict_llama32)}")

    # 5) Map each question_id to a GDSAM/LLAMA32
    gdsam_scores   = []
    llama32_scores = []
    for qid in question_ids:
        qid_str = str(qid)
        gdsam_scores.append(dict_gdsam.get(qid_str, None))
        llama32_scores.append(dict_llama32.get(qid_str, None))

    # 6) Filter out samples with None in gdsam
    f_pred_ent   = []
    f_lex_sim    = []
    f_sem_ent    = []
    f_sem_clust  = []
    f_accuracies = []
    f_gdsam      = []
    f_llama32    = []
    for (pe, ls, se, sc, acc, gd, ll) in zip(pred_ent, lex_sim, sem_ent, sem_clust, accuracies, gdsam_scores, llama32_scores):
        if gd is not None:
            f_pred_ent.append(pe)
            f_lex_sim.append(ls)
            f_sem_ent.append(se)
            f_sem_clust.append(sc)
            f_accuracies.append(acc)
            f_gdsam.append(gd)
            # If llama32 is None, treat as 0
            if ll is None:
                ll = 0.0
            f_llama32.append(ll)

    print(f"[INFO] Remaining samples after removing None gdsam: {len(f_pred_ent)}")

    # 7) Scale
    pe_scaled = min_max_scale(f_pred_ent)
    ls_scaled = min_max_scale(f_lex_sim)
    se_scaled = min_max_scale(f_sem_ent)
    sc_scaled = min_max_scale(f_sem_clust)
    gd_scaled = min_max_scale(f_gdsam)
    ll_scaled = min_max_scale(f_llama32)
    ac_final  = f_accuracies

    # 8) Split into calibration/test
    cal_dict, test_dict = split_dataset(
        pe_scaled, ls_scaled, se_scaled, sc_scaled,
        gd_scaled, ac_final,
        val_percent=0.25,
        seed=seed
    )
    print(f"[INFO] Calibration set size: {len(cal_dict['accuracy'])}, Test set size: {len(test_dict['accuracy'])}")

    # 9) Single metrics
    metrics_dict = {
        "predictive_entropy": {
            "X_calib": 1 - cal_dict["predictive_entropy"].reshape(-1,1),
            "X_test":  1 - test_dict["predictive_entropy"].reshape(-1,1)
        },
        "lexical_similarity": {
            "X_calib": cal_dict["lexical_similarity"].reshape(-1,1),
            "X_test":  test_dict["lexical_similarity"].reshape(-1,1)
        },
        "semantic_entropy": {
            "X_calib": 1 - cal_dict["semantic_entropy"].reshape(-1,1),
            "X_test":  1 - test_dict["semantic_entropy"].reshape(-1,1)
        },
        "semantic_clusters": {
            "X_calib": 1 - cal_dict["semantic_clusters"].reshape(-1,1),
            "X_test":  1 - test_dict["semantic_clusters"].reshape(-1,1)
        },
        "grounding_score_gdsam": {
            "X_calib": cal_dict["grounding_score"].reshape(-1,1),
            "X_test":  test_dict["grounding_score"].reshape(-1,1)
        }
    }

    calibration_models = {}
    calibrated_test_conf = {}

    # 10) Fit single-metric calibration
    for metric_name, data in metrics_dict.items():
        X_c = data["X_calib"]
        y_c = cal_dict["accuracy"]
        alphas, poly_features, pipe_single = fit_polynomial_regression(X_c, y_c, degree=2, alpha=3.0)

        # Apply to test
        X_t = data["X_test"]
        conf_test = calculate_confidence_values(X_t, alphas, poly_features)
        calibration_models[metric_name] = (alphas, poly_features)
        calibrated_test_conf[metric_name] = conf_test

    # 11) Uncalibrated for single metrics
    uncalib_single = {}
    for metric_name, data in metrics_dict.items():
        uncalib_single[metric_name] = data["X_test"].flatten()

    # 12) Plot single metrics
    for metric_name in metrics_dict.keys():
        # Uncalibrated
        path_uncal = os.path.join("reliability_plots", f"{save_path_prefix}_{metric_name}_uncalibrated.png")
        plot_reliability_diagram(
            uncalib_single[metric_name],
            test_dict["accuracy"],
            f"{metric_name} (Uncalibrated)",
            path_uncal
        )
        # Calibrated
        path_cal = os.path.join("reliability_plots", f"{save_path_prefix}_{metric_name}_calibrated.png")
        plot_reliability_diagram(
            calibrated_test_conf[metric_name],
            test_dict["accuracy"],
            f"{metric_name} (Calibrated)",
            path_cal
        )

    ##########################################################################
    ### 2D: (4 baselines) x (GDSAM, Llama32)
    ##########################################################################
    num_samples = len(pe_scaled)
    idx = np.arange(num_samples)
    random.seed(seed); np.random.seed(seed)
    np.random.shuffle(idx)
    n_val = int(0.25 * num_samples)
    val_idx = idx[:n_val]
    tst_idx = idx[n_val:]

    llama32_calib = np.array([ll_scaled[i] for i in val_idx])
    llama32_test  = np.array([ll_scaled[i] for i in tst_idx])
    y_cal = cal_dict["accuracy"]
    y_tst = test_dict["accuracy"]

    def get_calib_test_arrays(d, name):
        return (d[name]["X_calib"].flatten(), d[name]["X_test"].flatten())

    baseline_list = ["predictive_entropy", "lexical_similarity", "semantic_entropy", "semantic_clusters"]

    # 2D w/ GDSAM
    for base in baseline_list:
        base_c, base_t = get_calib_test_arrays(metrics_dict, base)
        gdsam_c, gdsam_t = get_calib_test_arrays(metrics_dict, "grounding_score_gdsam")

        combo_name = f"{base}_2D_GDSAM"
        print(f"[INFO] 2D calibration => {combo_name}")
        X_cal_2d = np.column_stack([base_c, gdsam_c])
        model_2d = calibrate_multi_features(X_cal_2d, y_cal, degree=2, alpha=3.0)

        X_tst_2d = np.column_stack([base_t, gdsam_t])
        conf_2d  = apply_multi_features_calibration(model_2d, X_tst_2d)

        out_path = os.path.join("reliability_plots", f"{save_path_prefix}_{combo_name}_2d_calibrated.png")
        plot_reliability_diagram(
            conf_2d,
            y_tst,
            f"{combo_name} (2D Calibrated)",
            out_path
        )

    # 2D w/ Llama32
    for base in baseline_list:
        base_c, base_t = get_calib_test_arrays(metrics_dict, base)

        combo_name = f"{base}_2D_LLAMA32"
        print(f"[INFO] 2D calibration => {combo_name}")
        X_cal_2d = np.column_stack([base_c, llama32_calib])
        model_2d = calibrate_multi_features(X_cal_2d, y_cal, degree=2, alpha=3.0)

        X_tst_2d = np.column_stack([base_t, llama32_test])
        conf_2d  = apply_multi_features_calibration(model_2d, X_tst_2d)

        out_path = os.path.join("reliability_plots", f"{save_path_prefix}_{combo_name}_2d_calibrated.png")
        plot_reliability_diagram(
            conf_2d,
            y_tst,
            f"{combo_name} (2D Calibrated)",
            out_path
        )

    ##########################################################################
    ### 3D: (4 baselines) x (GDSAM + Llama32)
    ##########################################################################
    for base in baseline_list:
        base_c, base_t = get_calib_test_arrays(metrics_dict, base)
        gdsam_c, gdsam_t = get_calib_test_arrays(metrics_dict, "grounding_score_gdsam")

        combo_name_3d = f"{base}_3D_GDSAM_LLAMA32"
        print(f"[INFO] 3D calibration => {combo_name_3d}")
        X_cal_3d = np.column_stack([base_c, gdsam_c, llama32_calib])
        model_3d = calibrate_multi_features(X_cal_3d, y_cal, degree=2, alpha=3.0)

        X_tst_3d = np.column_stack([base_t, gdsam_t, llama32_test])
        conf_3d  = apply_multi_features_calibration(model_3d, X_tst_3d)

        out_path = os.path.join("reliability_plots", f"{save_path_prefix}_{combo_name_3d}_3d_calibrated.png")
        plot_reliability_diagram(
            conf_3d,
            y_tst,
            f"{combo_name_3d} (3D Calibrated)",
            out_path
        )

    print(f"[INFO] All reliability diagrams saved with prefix '{save_path_prefix}' in folder '{out_dir}'.")

if __name__ == "__main__":
    # Example usage
    # for slake dataset
    uncertainty_filepath = "/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/runs_slake/llava_med_slake/uncertainty_with_gr_biomedclip/uncertainty_scores_baseline.pkl"
    grounding_root_dir   = "/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/runs_slake/llava_med_slake/grounding_with_clip_llama32"
    
    # for vqa dataset 
    # uncertainty_filepath = "/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/runs_slake/llava_med_slake/uncertainty_with_gr_biomedclip/uncertainty_scores_baseline.pkl"
    # grounding_root_dir_clip = "/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/runs_slake/llava_med_slake/grounding"
    # grounding_root_dir_llama32 = "/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/runs_slake/llava_med_slake/grounding_with_llama32"
    
    save_path_prefix     = "my_run_slake" 
    seed = random.randint(1, 1000)
    print(f"Using random seed: {seed}")

    run_pipeline(uncertainty_filepath, grounding_root_dir, save_path_prefix, seed)