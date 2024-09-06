import json
import numpy as np
import matplotlib.pyplot as plt

# Load results from JSON file
# with open('results.json', 'r') as f:
#     results = json.load(f)

# with open('/home/ubuntu/Multimodal-Uncertainty-Quantification/results_with_selected_tokens.json', 'r') as f:
#     results = json.load(f)

with open('/home/ubuntu/Multimodal-Uncertainty-Quantification/results_with_selected_tokens_new.json', 'r') as f:
    results = json.load(f)

# Step 1: Calculate confidence
# confidences = []
entropies = []
accuracies = []

for data in results.values():
    entropy = data['entropy']
    accuracy = data['accuracy']
    # confidence = 1 - entropy
    entropies.append(entropy)
    # confidences.append(confidence)
    accuracies.append(accuracy)

# Step 2: Bin confidence values
# confidences = np.array(confidences)
entropies = np.array(entropies)
entropies = (entropies - np.min(entropies))/(np.max(entropies) - np.min(entropies))
confidences = 1 - entropies
accuracies = np.array(accuracies)

bins = np.linspace(0, 1, 11)  # Create 10 bins between 0 and 1
bin_indices = np.digitize(confidences, bins) - 1  # Get bin indices for each confidence value

# Calculate average accuracy for each bin
avg_accuracies = []
for i in range(len(bins) - 1):
    bin_mask = bin_indices == i
    if bin_mask.any():
        avg_accuracy = np.mean(accuracies[bin_mask])
        avg_accuracies.append(avg_accuracy)
    else:
        avg_accuracies.append(np.nan)  # If no data in bin, append NaN

# Step 3: Plot the results
bin_centers = (bins[:-1] + bins[1:]) / 2  # Get bin centers for plotting

plt.figure(figsize=(8, 6))
plt.plot(bin_centers, avg_accuracies, marker='o')
plt.xlabel('Confidence')
plt.ylabel('Average Accuracy')
plt.title('Average Accuracy vs. Confidence')
plt.grid(True)
plt.xlim(0, 1)  # Ensure x-axis goes from 0 to 1

# Save the plot
# plt.savefig('average_accuracy_vs_confidence.png')
plt.savefig('average_accuracy_vs_confidence_results_with_selected_tokens_new.png')

# Optionally, display the plot
plt.show()