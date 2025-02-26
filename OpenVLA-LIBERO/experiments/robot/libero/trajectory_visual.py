import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set matplotlib to use the 'Agg' backend for saving plots without displaying
plt.switch_backend('Agg')

import plotly.graph_objects as go

def plot_and_save_3d_actions_interactive(actions, output_dir):
    """
    Plots all 3D actions in a single interactive plot and saves it as a JSON file for interactive viewing.

    Parameters:
        actions (list of np.ndarray): A list of 3D action arrays (shape: Nx3 per action).
        output_dir (str): Directory to save the JSON file.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    fig = go.Figure()

    for i, action in enumerate(actions):
        # Convert delta values to cumulative positions, starting at zero
        action = np.cumsum(np.vstack(([0, 0, 0], action)), axis=0)

        # Extract x, y, z coordinates from the action
        x, y, z = action[:, 0], action[:, 1], action[:, 2]

        # Add the trajectory as a line in the plot
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines+markers',
            marker=dict(size=4),
            line=dict(width=2),
            name=f"Trajectory {i+1}"
        ))

        # Mark the initial point with a larger red circle
        fig.add_trace(go.Scatter3d(
            x=[x[0]], y=[y[0]], z=[z[0]],
            mode='markers',
            marker=dict(size=8, color='red', symbol='circle'),
            name=f"Start {i+1}"
        ))

    # Update the layout for better visualization
    fig.update_layout(
        title="3D Actions (Interactive)",
        scene=dict(
            xaxis_title="X-axis",
            yaxis_title="Y-axis",
            zaxis_title="Z-axis"
        ),
        legend_title="Trajectories"
    )

    # Save the plot as a JSON file
    output_path = os.path.join(output_dir, "all_trajectories_interactive.json")
    with open(output_path, 'w') as f:
        f.write(fig.to_json())

    print(f"Interactive plot saved as: {output_path}")

def plot_and_save_3d_actions(actions, output_dir):
    """
    Plots all 3D actions in a single image and saves it.

    Parameters:
        actions (list of np.ndarray): A list of 3D action arrays (shape: Nx3 per action).
        output_dir (str): Directory to save the image.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, action in enumerate(actions):
        # Convert delta values to cumulative positions, starting at zero
        action = np.cumsum(np.vstack(([0, 0, 0.8], action * 0.1)), axis=0)
        # Extract x, y, z coordinates from the action
        x, y, z = action[:, 0], action[:, 1], action[:, 2]
        # Normalize x for color mapping
        normalized_x = np.linspace(0, 1, len(x))

        # Mark the initial point with a star
        ax.scatter(x[0], y[0], z[0], color='red', marker='*', s=100, label=f"Start {i+1}" if i == 0 else None)

        # Plot the 3D action with varying color intensity
        for j in range(len(x) - 1):
            color = plt.cm.viridis(normalized_x[j])  # Use colormap to set color based on x
            ax.plot(x[j:j+2], y[j:j+2], z[j:j+2], color=color)

    ax.set_title("3D Actions")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    # ax.legend()

    # Save the plot as a single image
    output_path = os.path.join(output_dir, "trajectory.png")
    plt.savefig(output_path)
    plt.close(fig)

# Example usage
if __name__ == "__main__":
    rollouts_folder = "/home/xilun/openvla/rollouts"
    all_success_trajectories = {}
    for transformation_folder in os.listdir(rollouts_folder): # antonym, synonym
        if transformation_folder == "trajectory_plots":
            continue
        transformation_path = os.path.join(rollouts_folder, transformation_folder)
        for transformed in os.listdir(transformation_path): # transformed, untransformed
            transformed_path = os.path.join(transformation_path, "untransformed")
            action_path = os.path.join(transformed_path, "actions", "2025_01_20")
            for action_file in os.listdir(action_path):
                # check whether the name contains success=True
                if "success=True" in action_file:
                    # load the npz file
                    action_data = np.load(os.path.join(action_path, action_file))
                    text_description = list(action_data.keys())[0]
                    action = np.load(os.path.join(action_path, action_file))[text_description]
                    action_xyz = action[:, :3]
                    # check whether text_description is in the all_success_trajectories dictionary
                    if text_description not in all_success_trajectories.keys():
                        all_success_trajectories[text_description] = []
                    all_success_trajectories[text_description].append(action_xyz)
    
    # print (all_success_trajectories.keys())
    # for key, value in all_success_trajectories.items():
    #     print (key, len(value))
                    
                    
    for key in all_success_trajectories.keys():
        trajectory_plots_folder = os.path.join(rollouts_folder, "trajectory_plots", key)
        # Plot and save the 3D actions
        plot_and_save_3d_actions(all_success_trajectories[key], trajectory_plots_folder)
        print(f"Plots saved in directory: {trajectory_plots_folder}")
        # plot_and_save_3d_actions_interactive(all_success_trajectories[key], trajectory_plots_folder)
        # print(f"Interactive plots saved in directory: {trajectory_plots_folder}")
        