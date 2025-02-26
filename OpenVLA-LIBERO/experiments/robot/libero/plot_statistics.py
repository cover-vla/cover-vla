
# required calculations 
# 1. text embedding similrity between trnasfomed and untransformed text
# 2. action similarity between transformed and untransformed actions
# 3. action entropy 
# 4. action cross-entropy between transformed and untransformed actions


import numpy as np
import os
import tqdm
# import package for cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import math



def plot_box_plot(save_path, transform_type, text_embeddings, action_similarities, transformed_entropy, untransformed_entropy, cross_entropy, kl_divergence):
    
    fig, axs = plt.subplots(1, 5, figsize=(15, 5))
    fig.suptitle(f"Boxplots of Metrics For {transform_type}")

    # 1) Boxplot of [all_u_H, all_t_H]
    axs[0].boxplot([untransformed_entropy, transformed_entropy], labels=['Untransformed', 'Transformed'])
    axs[0].set_title("Untransformed vs. Transformed (Entropy)")
    axs[0].set_ylabel("Value")
    axs[0].set_ylim([0, 120])

    # 2) Boxplot of Cross Entropy
    axs[1].boxplot(cross_entropy)
    axs[1].set_title("Cross Entropy")
    axs[1].set_xticks([1])
    axs[1].set_xticklabels(['cross_entropy'])
    axs[1].set_ylim([0, 35000])

    # 3) Boxplot of KL Divergence
    axs[4].boxplot(kl_divergence)
    axs[4].set_title("KL Divergence")
    axs[4].set_xticks([1])
    axs[4].set_xticklabels(['kl_div'])
    # axs[4].set_ylim([0, 120])

    axs[2].boxplot(action_similarities)
    axs[2].set_title("Euclidean Distance")
    axs[2].set_xticks([1])
    axs[2].set_xticklabels(['action_sim'])
    axs[2].set_ylim([0, 50])

    axs[3].boxplot(text_embeddings)
    axs[3].set_title("Semantic Similarity")
    axs[3].set_xticks([1])
    axs[3].set_xticklabels(['all_sim'])
    axs[3].set_ylim([0.8, 1.2])
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/metrics_boxplots_{transform_type}.png')
    plt.close()

def plot_histogram(data, save_path, title="Histogram", xlabel="Values", ylabel="Frequency", color="blue"):

    # Create the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=15, color=color, edgecolor="black", alpha=0.7)

    # Add labels and title
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.xlim([0, 1])
    

    path = os.path.join(save_path, title + ".png")
    # Save the plot to the specified path
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()  # Close the figure to free memory



def get_text_embedding_similarity(rollout_path, save_plot=False):
    for files in os.listdir(rollout_path):
        if files == "untransformed":
            untransformed_path = os.path.join(rollout_path, files)
        elif files == "transformed":
            transformed_path = os.path.join(rollout_path, files)
            
    text_embedding_untransformed_path = os.path.join(untransformed_path, "text_embeddings")
    text_embedding_transformed_path = os.path.join(transformed_path, "text_embeddings")

    embedding_list = []
    print ("Calculating text embedding similarity")
    for text_embedding in tqdm.tqdm(os.listdir(text_embedding_untransformed_path)):
        short_text_embedding = text_embedding.split("--")[:2]
        short_text_embedding = short_text_embedding[0] + "--" + short_text_embedding[1] + "--"
        
        for transformed_text_embedding in os.listdir(text_embedding_transformed_path):
            if short_text_embedding in transformed_text_embedding:
                text_embedding_transformed = transformed_text_embedding
                break
        assert type(text_embedding_transformed) != np.ndarray, "Error in loading text embedding"
        text_embedding_transformed = np.load(os.path.join(text_embedding_transformed_path, text_embedding_transformed),allow_pickle=True).item()
        text_embedding_untransformed = np.load(os.path.join(text_embedding_untransformed_path, text_embedding),allow_pickle=True).item()
        # print (text_embedding_transformed)
        # input()
        key = list(text_embedding_untransformed.keys())[0]
        text_embedding_untransformed = np.asarray(text_embedding_untransformed[key])[:,0].reshape(-1, 4096)
        text_embedding_transformed = np.asarray(text_embedding_transformed[key])[:,0].reshape(-1, 4096)
        try:
            text_embedding_similarity = cosine_similarity(text_embedding_untransformed, text_embedding_transformed)
            one_similarity = np.mean(np.diag(text_embedding_similarity))
        except:
            print ("Error in calculating cosine similarity")
            continue
        # if type(text_embedding_similarity) == np.ndarray:
        #     text_embedding_similarity = text_embedding_similarity[0][0]
        embedding_list.append(one_similarity)
        # print(f"Text embedding similarity for {text_embedding} is {text_embedding_similarity}")
    if save_plot:
        plot_histogram(embedding_list, rollout_path, title="text_embedding_similarity", xlabel="Cosine Similarity", ylabel="Frequency")
    
    return embedding_list

def get_action_similarity(rollout_path, save_plot=False):
    for files in os.listdir(rollout_path):
        if files == "untransformed":
            untransformed_path = os.path.join(rollout_path, files)
        elif files == "transformed":
            transformed_path = os.path.join(rollout_path, files)
            
    action_untransformed_path = os.path.join(untransformed_path, "actions")
    action_transformed_path = os.path.join(transformed_path, "actions")
    action_similarity_list = []
    print ("Calculating action similarity")
    for action in tqdm.tqdm(os.listdir(action_untransformed_path)):
        action_untransformed = np.load(os.path.join(action_untransformed_path, action), allow_pickle=True).item()
        short_action = action.split("--")[:2]
        short_action = short_action[0] + "--" + short_action[1] + "--"
        
        for transformed_action in os.listdir(action_transformed_path):
            if short_action in transformed_action:
                action_transformed = transformed_action
                break
        action_transformed = np.load(os.path.join(action_transformed_path, action_transformed), allow_pickle=True).item()
        key = list(action_untransformed.keys())[0]
        assert len(action_untransformed.keys()) == 1
        action_untransformed = np.asarray(action_untransformed[key])
        action_transformed = np.asarray(action_transformed[key])
        
        # make sure they have the same length
        length = min(len(action_untransformed), len(action_transformed))
        step_untransformed = max(1, len(action_untransformed) // length)
        step_transformed = max(1, len(action_transformed) // length)

        action_untransformed = action_untransformed[::step_untransformed][:length]
        action_transformed = action_transformed[::step_transformed][:length]

        try:
            # action_similarity = np.linalg.norm(action_untransformed - action_transformed)
            action_similarity = math.sqrt(np.sum((action_untransformed - action_transformed) ** 2))
        except:
            continue
        action_similarity_list.append(action_similarity)
        # print(f"Action similarity for {action} is {action_similarity}")   
    if save_plot:
        plot_histogram(action_similarity_list, rollout_path, title="action_similarity", xlabel="L2 Norm", ylabel="Frequency")    
    
    return action_similarity_list
    
    
def get_entropy(rollout_path, save_plot=False, transformation="untransformed"):
    for files in os.listdir(rollout_path):
        if files == "untransformed":
            untransformed_path = os.path.join(rollout_path, files)
        elif files == "transformed":
            transformed_path = os.path.join(rollout_path, files)
            
    action_untransformed_path = os.path.join(untransformed_path, "action_probs")
    action_transformed_path = os.path.join(transformed_path, "action_probs")
    
    average_entropy_list = []
    print ("Calculating action entropy")
    for action in tqdm.tqdm(os.listdir(action_untransformed_path)):
        short_action = action.split("--")[:2]
        short_action = short_action[0] + "--" + short_action[1] + "--"
        
        for transformed_action in os.listdir(action_transformed_path):
            if short_action in transformed_action:
                action_transformed = transformed_action
                break
        if transformation == "untransformed":
            action_untransformed = np.load(os.path.join(action_untransformed_path, action),allow_pickle=True).item()
            key = list(action_untransformed.keys())[0]
            all_action = np.asarray(action_untransformed[key])
        elif transformation == "transformed":
            action_transformed = np.load(os.path.join(action_transformed_path, action_transformed),allow_pickle=True).item()
            key = list(action_transformed.keys())[0]
            all_action = np.asarray(action_transformed[key])
        
        # Calculate the average entropy across all steps and all action dimensions
        entropies = []
        epsilon = 1e-10
        clipped_probs = np.clip(all_action[..., 0, :], epsilon, 1.0)
        entropies = -np.sum(clipped_probs * np.log(clipped_probs), axis=-1)
        # Compute the average entropy across all steps and action dimensions
        average_entropy = np.mean(entropies)
        average_entropy_list.append(average_entropy)

    if save_plot:
        plot_histogram(average_entropy_list, rollout_path, title="action_entropy", xlabel="Entropy", ylabel="Frequency")
        # print(f"Average transformed action entropy: {average_entropy}")

    return average_entropy_list
def get_cross_entropy(rollout_path, save_plot=False):
    for files in os.listdir(rollout_path):
        if files == "untransformed":
            untransformed_path = os.path.join(rollout_path, files)
        elif files == "transformed":
            transformed_path = os.path.join(rollout_path, files)
            
    action_untransformed_path = os.path.join(untransformed_path, "action_probs")
    action_transformed_path = os.path.join(transformed_path, "action_probs")
    
    cross_entropies_list = []
    kl_divergences_list = []
    print ("Calculating cross entropy")
    for action in tqdm.tqdm(os.listdir(action_untransformed_path)):
        action_untransformed = np.load(os.path.join(action_untransformed_path, action), allow_pickle=True).item()
        short_action = action.split("--")[:2]
        short_action = short_action[0] + "--" + short_action[1] + "--"
        
        for transformed_action in os.listdir(action_transformed_path):
            if short_action in transformed_action:
                action_transformed = transformed_action
                break
        action_transformed = np.load(os.path.join(action_transformed_path, action_transformed), allow_pickle=True).item()
        key = list(action_untransformed.keys())[0]
        action_untransformed = np.asarray(action_untransformed[key])
        action_transformed = np.asarray(action_transformed[key])
        # Handle different number of steps by downsampling to the same length
        length = min(action_untransformed.shape[0], action_transformed.shape[0])  # Minimum number of steps
        step_untransformed = max(1, action_untransformed.shape[0] // length)
        step_transformed = max(1, action_transformed.shape[0] // length)
        
        action_untransformed = action_untransformed[::step_untransformed][:length]
        action_transformed = action_transformed[::step_transformed][:length]

        
        # Calculate the cross-entropy across all steps and all action dimensions
        epsilon = 1e-10
        
        # Clip the probability distributions to avoid log(0) issues
        transformed_probs = np.clip(action_transformed[..., 0, :], epsilon, 1.0)
        untransformed_probs = np.clip(action_untransformed[..., 0, :], epsilon, 1.0)

        # Compute cross-entropy using vectorized operations
        cross_entropies = -np.sum(untransformed_probs * np.log(transformed_probs), axis=-1)

        # Compute the average cross-entropy across all steps and action dimensions
        average_cross_entropy = np.mean(cross_entropies)
        cross_entropies_list.append(average_cross_entropy)
        
        # Compute KL Divergence: D_KL(P || Q) = sum( P * log(P / Q) )
        kl_divergences = np.sum(untransformed_probs * (np.log(untransformed_probs) - np.log(transformed_probs)), axis=-1)

        # Compute the average KL divergence across all steps and action dimensions
        average_kl_divergence = np.mean(kl_divergences)
        kl_divergences_list.append(average_kl_divergence)
        # print(f"Average cross-entropy between untransformed and transformed for {action}: {average_cross_entropy}")
    if save_plot:
        plot_histogram(cross_entropies_list, rollout_path, title="cross_entropy", xlabel="Cross Entropy", ylabel="Frequency")
        plot_histogram(kl_divergences_list, rollout_path, title="kl_divergence", xlabel="KL Divergence", ylabel="Frequency")
    return cross_entropies_list, kl_divergences_list

if __name__ == "__main__":
    
    transformation_type_list = ["synonym", "antonym"]# ["antonym", "synonym"]
    model_family = "openvla" # octo
    save_plot = True
    for transformation_type in transformation_type_list:
        rollout_path = f"/home/xilun/openvla/rollouts/{model_family}/{transformation_type}"
        text_embeddings = get_text_embedding_similarity(rollout_path, save_plot=save_plot)
        action_similarity = get_action_similarity(rollout_path, save_plot=save_plot)
        untransformed_entropy = get_entropy(rollout_path, save_plot=save_plot, transformation="untransformed")
        transformed_entropy = get_entropy(rollout_path, save_plot=save_plot, transformation="transformed")
        cross_entropy, kl_divergence = get_cross_entropy(rollout_path, save_plot=save_plot)
        
        plot_box_plot(rollout_path, transformation_type, text_embeddings, action_similarity, transformed_entropy, untransformed_entropy, cross_entropy, kl_divergence)