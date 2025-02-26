import os
import math
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import json

# ============================
# Metric Calculation Functions
# ============================

def get_text_embedding_similarity(rollout_path, save_plot=False):
    # Identify subdirectories for untransformed and transformed data
    for files in os.listdir(rollout_path):
        if files == "untransformed":
            untransformed_path = os.path.join(rollout_path, files)
        elif files == "transformed":
            transformed_path = os.path.join(rollout_path, files)
            
    text_embedding_untransformed_path = os.path.join(untransformed_path, "text_embeddings")
    text_embedding_transformed_path = os.path.join(transformed_path, "text_embeddings")

    embedding_list = []
    print("Calculating text embedding similarity")
    for text_embedding in tqdm.tqdm(os.listdir(text_embedding_untransformed_path)):
        short_text_embedding = text_embedding.split("--")[:2]
        short_text_embedding = short_text_embedding[0] + "--" + short_text_embedding[1] + "--"
        
        # Find the matching transformed file
        for transformed_text_embedding in os.listdir(text_embedding_transformed_path):
            if short_text_embedding in transformed_text_embedding:
                file_transformed = transformed_text_embedding
                break
        # Load the embeddings (each file is assumed to be a dictionary)
        text_embedding_transformed = np.load(os.path.join(text_embedding_transformed_path, file_transformed),
                                              allow_pickle=True).item()
        text_embedding_untransformed = np.load(os.path.join(text_embedding_untransformed_path, text_embedding),
                                                allow_pickle=True).item()
        key = list(text_embedding_untransformed.keys())[0]
        text_embedding_untransformed = np.concatenate(text_embedding_untransformed[key], axis=1)
        average_text_embedding_untransformed = np.mean(text_embedding_untransformed, axis=1)
        text_embedding_transformed = np.concatenate(text_embedding_transformed[key], axis=1)
        average_text_embedding_transformed = np.mean(text_embedding_transformed, axis=1)
        try:
            text_embedding_similarity = cosine_similarity(average_text_embedding_untransformed, average_text_embedding_transformed)
            one_similarity = np.mean(np.diag(text_embedding_similarity))
        except Exception as e:
            print("Error in calculating cosine similarity:", e)
            continue
        embedding_list.append(one_similarity)
        
    # (Optional individual plot if needed)
    if save_plot:
        _plot_histogram(embedding_list, rollout_path,
                        title="text_embedding_similarity",
                        xlabel="Cosine Similarity", ylabel="Frequency")
    return embedding_list


def get_action_similarity(rollout_path, save_plot=False):
    # Identify subdirectories for untransformed and transformed data
    for files in os.listdir(rollout_path):
        if files == "untransformed":
            untransformed_path = os.path.join(rollout_path, files)
        elif files == "transformed":
            transformed_path = os.path.join(rollout_path, files)
            
    action_untransformed_path = os.path.join(untransformed_path, "actions")
    action_transformed_path = os.path.join(transformed_path, "actions")
    action_similarity_list = []
    print("Calculating action similarity")
    for action in tqdm.tqdm(os.listdir(action_untransformed_path)):
        action_untransformed = np.load(os.path.join(action_untransformed_path, action),
                                       allow_pickle=True).item()
        short_action = action.split("--")[:2]
        short_action = short_action[0] + "--" + short_action[1] + "--"
        
        for transformed_action in os.listdir(action_transformed_path):
            if short_action in transformed_action:
                file_transformed = transformed_action
                break
        action_transformed = np.load(os.path.join(action_transformed_path, file_transformed),
                                     allow_pickle=True).item()
        key = list(action_untransformed.keys())[0]
        # (Assume one key per file)
        action_untransformed = np.asarray(action_untransformed[key])
        action_transformed = np.asarray(action_transformed[key])
        
        # Ensure both arrays have the same length by downsampling
        length = min(len(action_untransformed), len(action_transformed))
        step_untransformed = max(1, len(action_untransformed) // length)
        step_transformed = max(1, len(action_transformed) // length)
        action_untransformed = action_untransformed[::step_untransformed][:length]
        action_transformed = action_transformed[::step_transformed][:length]
        try:
            # Compute L2 norm (Euclidean distance) between actions
            action_similarity = math.sqrt(np.sum((action_untransformed - action_transformed) ** 2))
        except Exception as e:
            print("Error in calculating action similarity:", e)
            continue
        action_similarity_list.append(action_similarity)
    if save_plot:
        _plot_histogram(action_similarity_list, rollout_path,
                        title="action_similarity",
                        xlabel="L2 Norm", ylabel="Frequency")
    return action_similarity_list


def get_entropy(rollout_path, save_plot=False, transformation="untransformed"):
    # Identify subdirectories for untransformed and transformed data
    for files in os.listdir(rollout_path):
        if files == "untransformed":
            untransformed_path = os.path.join(rollout_path, files)
        elif files == "transformed":
            transformed_path = os.path.join(rollout_path, files)
            
    action_untransformed_path = os.path.join(untransformed_path, "action_probs")
    action_transformed_path = os.path.join(transformed_path, "action_probs")
    
    average_entropy_list = []
    print("Calculating action entropy for", transformation)
    for action in tqdm.tqdm(os.listdir(action_untransformed_path)):
        short_action = action.split("--")[:2]
        short_action = short_action[0] + "--" + short_action[1] + "--"
        
        for transformed_action in os.listdir(action_transformed_path):
            if short_action in transformed_action:
                file_transformed = transformed_action
                break
        if transformation == "untransformed":
            action_data = np.load(os.path.join(action_untransformed_path, action),
                                  allow_pickle=True).item()
        elif transformation == "transformed":
            action_data = np.load(os.path.join(action_transformed_path, file_transformed),
                                  allow_pickle=True).item()
        key = list(action_data.keys())[0]
        all_action = np.asarray(action_data[key])
        
        epsilon = 1e-10
        clipped_probs = np.clip(all_action[..., 0, :], epsilon, 1.0)
        entropies = -np.sum(clipped_probs * np.log(clipped_probs), axis=-1)
        average_entropy = np.mean(entropies)
        average_entropy_list.append(average_entropy)

    if save_plot:
        _plot_histogram(average_entropy_list, rollout_path,
                        title="action_entropy", xlabel="Entropy", ylabel="Frequency")
    return average_entropy_list


def get_cross_entropy(rollout_path, save_plot=False):
    # Identify subdirectories for untransformed and transformed data
    for files in os.listdir(rollout_path):
        if files == "untransformed":
            untransformed_path = os.path.join(rollout_path, files)
        elif files == "transformed":
            transformed_path = os.path.join(rollout_path, files)
            
    action_untransformed_path = os.path.join(untransformed_path, "action_probs")
    action_transformed_path = os.path.join(transformed_path, "action_probs")
    
    cross_entropies_list = []
    kl_divergences_list = []
    print("Calculating cross entropy")
    for action in tqdm.tqdm(os.listdir(action_untransformed_path)):
        action_untransformed = np.load(os.path.join(action_untransformed_path, action),
                                       allow_pickle=True).item()
        short_action = action.split("--")[:2]
        short_action = short_action[0] + "--" + short_action[1] + "--"
        
        for transformed_action in os.listdir(action_transformed_path):
            if short_action in transformed_action:
                file_transformed = transformed_action
                break
        action_transformed = np.load(os.path.join(action_transformed_path, file_transformed),
                                     allow_pickle=True).item()
        key = list(action_untransformed.keys())[0]
        action_untransformed = np.asarray(action_untransformed[key])
        action_transformed = np.asarray(action_transformed[key])
        
        # Downsample to the same number of steps
        length = min(action_untransformed.shape[0], action_transformed.shape[0])
        step_untransformed = max(1, action_untransformed.shape[0] // length)
        step_transformed = max(1, action_transformed.shape[0] // length)
        action_untransformed = action_untransformed[::step_untransformed][:length]
        action_transformed = action_transformed[::step_transformed][:length]
        
        epsilon = 1e-10
        transformed_probs = np.clip(action_transformed[..., 0, :], epsilon, 1.0)
        untransformed_probs = np.clip(action_untransformed[..., 0, :], epsilon, 1.0)

        cross_entropies = -np.sum(untransformed_probs * np.log(transformed_probs), axis=-1)
        average_cross_entropy = np.mean(cross_entropies)
        cross_entropies_list.append(average_cross_entropy)
        
        kl_divergences = np.sum(untransformed_probs * (np.log(untransformed_probs) - np.log(transformed_probs)), axis=-1)
        average_kl_divergence = np.mean(kl_divergences)
        kl_divergences_list.append(average_kl_divergence)
        
    if save_plot:
        _plot_histogram(cross_entropies_list, rollout_path,
                        title="cross_entropy", xlabel="Cross Entropy", ylabel="Frequency")
        _plot_histogram(kl_divergences_list, rollout_path,
                        title="kl_divergence", xlabel="KL Divergence", ylabel="Frequency")
    return cross_entropies_list, kl_divergences_list

# (Optional helper function for individual histogram plotting)
def _plot_histogram(data, save_path, title="Histogram", xlabel="Values", ylabel="Frequency", color="blue"):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=15, color=color, edgecolor="black", alpha=0.7)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, title + ".png"), dpi=300)
    plt.close()


# ==============================
# Combined Plotting Functions
# ==============================

def plot_combined_histograms(metrics_dict, output_path):
    """
    Creates a 3x2 figure overlaying histograms for the following metrics:
      [0,0]: Euclidean Distance (action similarity)
      [0,1]: Instruction Similarity (text embedding similarity)
      [1,0]: Untransformed Action Entropy
      [1,1]: Difference in Entropy (untransformed minus transformed)
      [2,0]: Cross Entropy
      [2,1]: KL Divergence
    """
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle("Metrics Distributions Comparison", fontsize=18)
    
    colors = ['grey', 'yellow', 'green', 'orange', 'purple', 'black', 'blue']
    transform_types = list(metrics_dict.keys())
    
    for idx, ttype in enumerate(transform_types):
        data = metrics_dict[ttype]
        eu_distance = np.array(data['eu_distance'])
        similarity = np.array(data['similarity'])
        all_u_H = np.array(data['all_u_H'])
        all_t_H = np.array(data['all_t_H'])
        all_cross_H = np.array(data['all_cross_H'])
        all_kl_H = np.array(data['all_kl_H'])
        diff_entropies = all_u_H - all_t_H
        
        alpha = 0.5
        col = colors[idx % len(colors)]
        axs[0, 0].hist(eu_distance, bins=20, color=col, alpha=alpha, label=ttype)
        axs[0, 1].hist(similarity, bins=20, color=col, alpha=alpha, label=ttype)
        axs[1, 0].hist(all_u_H, bins=20, color=col, alpha=alpha, label=f'{ttype} (U)')
        axs[1, 1].hist(diff_entropies, bins=20, color=col, alpha=alpha, label=ttype)
        axs[2, 0].hist(all_cross_H, bins=20, color=col, alpha=alpha, label=ttype)
        axs[2, 1].hist(all_kl_H, bins=20, color=col, alpha=alpha, label=ttype)
    
    titles_hist = ["Euclidean Distance", "Instruction Similarity", 
                   "Untransformed Action Entropy", "Difference in Entropy (U - T)",
                   "Cross Entropy", "KL Divergence"]
    
    for ax, title_str in zip(axs.flat, titles_hist):
        ax.set_title(title_str, fontsize=14)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.legend(fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(output_path, 'metrics_histograms_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_combined_boxplots(metrics_dict, output_path):
    """
    Creates a 2x3 figure overlaying boxplots for:
      [0]: Untransformed Entropy
      [1]: Transformed Entropy
      [2]: Cross Entropy
      [3]: KL Divergence
      [4]: Euclidean Distance
      [5]: Instruction Similarity
    For each subplot, boxplots from different transformation types are slightly offset.
    """
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.flatten()
    fig.suptitle("Boxplots of Metrics", fontsize=18, y=0.98)
    
    colors = ['grey', 'yellow', 'green', 'orange', 'purple', 'black', 'blue']
    transform_types = list(metrics_dict.keys())
    
    for idx, ttype in enumerate(transform_types):
        data = metrics_dict[ttype]
        eu_distance = np.array(data['eu_distance'])
        similarity = np.array(data['similarity'])
        all_u_H = np.array(data['all_u_H'])
        all_t_H = np.array(data['all_t_H'])
        all_cross_H = np.array(data['all_cross_H'])
        all_kl_H = np.array(data['all_kl_H'])
        
        pos = 1 + idx * 0.3  # Offset position for each transformation type
        alpha = 0.5
        col = colors[idx % len(colors)]
        
        axs[0].boxplot(all_u_H, positions=[pos],
                       patch_artist=True,
                       boxprops=dict(facecolor=col, alpha=alpha))
        axs[1].boxplot(all_t_H, positions=[pos],
                       patch_artist=True,
                       boxprops=dict(facecolor=col, alpha=alpha))
        axs[2].boxplot(all_cross_H, positions=[pos],
                       patch_artist=True,
                       boxprops=dict(facecolor=col, alpha=alpha))
        axs[3].boxplot(all_kl_H, positions=[pos],
                       patch_artist=True,
                       boxprops=dict(facecolor=col, alpha=alpha))
        axs[4].boxplot(eu_distance, positions=[pos],
                       patch_artist=True,
                       boxprops=dict(facecolor=col, alpha=alpha))
        axs[5].boxplot(similarity, positions=[pos],
                       patch_artist=True,
                       boxprops=dict(facecolor=col, alpha=alpha))
    
    titles_box = ["Untransformed Entropy", "Transformed Entropy", "Cross Entropy", 
                  "KL Divergence", "Euclidean Distance", "Instruction Similarity"]
    ylims_box = [[0, 50], [0, 50], [0, 150], [0, 120], [0, 5], [0, 1]]
    
    for ax, title_str, ylim in zip(axs, titles_box, ylims_box):
        ax.set_title(title_str, fontsize=14, pad=10)
        ax.set_ylabel("Value", fontsize=12)
        # ax.set_ylim(ylim)
        ax.set_xticks([])  # Remove x-tick labels for clarity
        ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Build a legend manually
    legend_handles = []
    legend_labels = []
    for idx, ttype in enumerate(transform_types):
        patch = plt.Rectangle((0, 0), 1, 1, fc=colors[idx % len(colors)], alpha=0.5)
        legend_handles.append(patch)
        legend_labels.append(ttype)
    
    fig.legend(legend_handles, legend_labels, bbox_to_anchor=(0.5, 0.92),
               loc='center', ncol=len(transform_types), fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(output_path, 'metrics_boxplots_comparison.png'),
                dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close(fig)


# ==============================
# Main Routine: Aggregate Metrics and Plot
# ==============================

def main():
    # List of transformation types and your model family (adjust paths as needed)
    transformation_type_list = ["interpolation", "synonym", "antonym","negation", "verb_noun_shuffle", "in_set", "out_set", "random_shuffle"]
    model_family = "openvla"
    # Define where to save the combined plots (e.g., current directory)
    output_path = "/home/xilun/openvla/rollouts/openvla"
    
    # Dictionary to hold aggregated metrics for each transformation type
    metrics_dict = {}
    
    for transformation_type in transformation_type_list:
        rollout_path = f"/data/xilun/rollouts/{model_family}_old/{transformation_type}"
        print(f"\nProcessing transformation: {transformation_type}")
        text_embeddings = get_text_embedding_similarity(rollout_path, save_plot=False)
        action_similarity = get_action_similarity(rollout_path, save_plot=False)
        untransformed_entropy = get_entropy(rollout_path, save_plot=False, transformation="untransformed")
        transformed_entropy = get_entropy(rollout_path, save_plot=False, transformation="transformed")
        cross_entropy, kl_divergence = get_cross_entropy(rollout_path, save_plot=False)
        
        # Save metrics for this transformation type
        metrics_dict[transformation_type] = {
            'eu_distance': action_similarity,
            'similarity': text_embeddings,
            'all_u_H': untransformed_entropy,
            'all_t_H': transformed_entropy,
            'all_cross_H': cross_entropy,
            'all_kl_H': kl_divergence
        }
    
    # save the metrics_dict to a file
    np.save(os.path.join(output_path, "metrics_dict.npy"), metrics_dict)
    print("\nMetrics saved.")
    # Create combined figures
    plot_combined_histograms(metrics_dict, output_path)
    plot_combined_boxplots(metrics_dict, output_path)
    print("\nCombined plots saved.")


if __name__ == "__main__":
    main()
