import matplotlib.pyplot as plt
import numpy as np
import os
import networkx as nx
import matplotlib.patches as mpatches

def plot_network(G, save_dir, day):
    attitude_colors = {1: "blue", 2: "green", 3: "orange", 4: "red"}
    # Assign edge colors based on node attitudes
    edge_colors = []
    for edge in G.edges():
        node1, node2 = edge
        if G.nodes[node1]["attitude"] == G.nodes[node2]["attitude"]:
            edge_colors.append(attitude_colors[G.nodes[node1]["attitude"]])
        else:
            edge_colors.append("gray")  # Default color for mixed attitudes
    plt.figure(figsize=(30, 30))
    # Draw the graph
    pos = nx.spring_layout(G)  # Layout for visualization
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color=edge_colors, width=2)

    # Add a legend
    legend_handles = [mpatches.Patch(color=color, label=f"Attitude {attitude}")
                    for attitude, color in attitude_colors.items()]
    plt.legend(handles=legend_handles, loc="upper right")
    plt.savefig(os.path.join(save_dir, f"network_day_{day}.png"))
    plt.close()

def plot_attitudes(attitude_dist, model_type, policy, save_dir):
    against = [a[0] for a in attitude_dist]
    swing = [a[1] for a in attitude_dist]
    support = [a[2] for a in attitude_dist]
    x = np.arange(len(attitude_dist))
    plt.figure(figsize=(10, 6))
    colors = ['#4169E1', '#FFD700', '#FF7F50']
    
    # Plot lines with markers
    plt.plot(x, against, label="Against Percentage", marker='o', color=colors[0])
    plt.plot(x, swing, label='Swing Percentage', marker='o', color=colors[1])
    plt.plot(x, support, label='Support Percentage', marker='o', color=colors[2])

    # Adding annotations for each point
    for i, (a, s, sup) in enumerate(zip(against, swing, support)):
        plt.text(i, a, f'{a:.2f}', ha='right', va='bottom', color=colors[0])
        plt.text(i, s, f'{s:.2f}', ha='left', va='bottom', color=colors[1])
        plt.text(i, sup, f'{sup:.2f}', ha='center', va='top', color=colors[2])

    plt.xlabel('Weeks')
    plt.ylabel('Percentage')
    plt.ylim(0, 1)
    model_disease = save_dir.split('/')[-1]
    plt.title(model_disease + f'/{policy}' + ': Attitudes Over Time')
    plt.legend()
    plt.grid(True)
    # Save the figure
    plt.savefig(os.path.join(save_dir, "attitude.png"))
    plt.close()

def compute_running_average_from_existing_plot():
    """Compute the running average of the existing attitude curves from the plot."""
    lines = plt.gca().get_lines()  # Get all lines on the current plot
    against_data = []
    swing_data = []
    support_data = []

    # Loop through the lines and separate the data for each attitude (Against, Swing, Support)
    for i, line in enumerate(lines):
        y_data = line.get_ydata()
        if 'Average' not in line.get_label():  # Avoid using existing average lines
            if i % 3 == 0:  # "Against" curves
                against_data.append(y_data)
            elif i % 3 == 1:  # "Swing" curves
                swing_data.append(y_data)
            elif i % 3 == 2:  # "Support" curves
                support_data.append(y_data)

    # Compute the averages if there is data
    if len(against_data) > 0:
        avg_against = np.mean(against_data, axis=0)
        avg_swing = np.mean(swing_data, axis=0)
        avg_support = np.mean(support_data, axis=0)
        return avg_against, avg_swing, avg_support
    return [], [], []

def update_or_add_average_curves(x, avg_against, avg_swing, avg_support, run_idx, first_plot):
    """Update the existing average curves or add them if they don't exist, and annotate only the newly plotted ones."""
    ax = plt.gca()
    lines = ax.get_lines()
    
    avg_colors = ['#00008B', '#B8860B', '#CD5C5C']  # Slightly darker colors for averages
    avg_labels = ['Average Against', 'Average Swing', 'Average Support']
    avg_lines = []

    # Check if average curves already exist
    for line in lines:
        if line.get_label() in avg_labels:
            avg_lines.append(line)

    # If average curves exist, update their data without re-annotating
    if len(avg_lines) == 3:
        avg_lines[0].set_ydata(avg_against)  # Update "Average Against"
        avg_lines[1].set_ydata(avg_swing)    # Update "Average Swing"
        avg_lines[2].set_ydata(avg_support)  # Update "Average Support"
        
    else:
        plt.plot(x, avg_against, label='Average Against', linestyle='-', color=avg_colors[0], zorder=5, linewidth=3)
        plt.plot(x, avg_swing, label='Average Swing', linestyle='-', color=avg_colors[1], zorder=5, linewidth=3)
        plt.plot(x, avg_support, label='Average Support', linestyle='-', color=avg_colors[2], zorder=5, linewidth=3)

        # Annotate the average curves (only if they are plotted for the first time in this call)

def plot_policy_attitudes(policy_to_avg_att_data, model_str, policies_heads, seed):
    '''
    attitude_dist_data: list of lists, each sublist contains the attitude distribution for a policy
    model_str: string, the model name
    run_idx: int, the index of the run
    '''
    general_policy_str = policies_heads[0].split("_")[1] if "_" in policies_heads[0] else policies_heads[0]

    plot_name = f"results/{model_str}_{general_policy_str}_attitudes_seed={seed}.png"

    plt.figure(plot_name, figsize=(10, 6))
    plt.ylim(0, 1)
    num_policies = len(policy_to_avg_att_data.items())
    assert num_policies <= 3, "Only 3 policies are supported for plotting"
    # colors_map = {
    #     0: ['#4169E1', '#FFD700', '#FF7F50'],
    #     1: ['#FF4500', '#FF1493', '#00FF00'],
    #     2: ['#8A2BE2', '#FFA500', '#00FFFF'],
    # }
    colors_map = {
        0: "#66c2a5",
        1: "#fc8d62",
        2: "#8da0cb"
    }
    for i, attitude_dist in policy_to_avg_att_data.items():
        against = attitude_dist['against']
        # swing = attitude_dist['swing']
        support = attitude_dist['support']
        x = np.arange(len(against))
        # plt.plot(x, against, label=f"Against, i={i}", color=colors_map[i][0])
        # plt.plot(x, swing, label=f'Swing, i={i}', color=colors_map[i][1])
        plt.plot(x, support, label=f'Support, Policy={policies_heads[i]}', color=colors_map[i])
    plt.legend()
    plt.savefig(plot_name)
    

def add_attitude_curve(attitude_dist, model_str, policy_str, run_idx, average=False, save_dir=None):
    
    if save_dir is not None:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        plot_name = f"{save_dir}/{model_str}_{policy_str}_attitudes.png"
    else:
        if not os.path.isdir("results"):
            os.makedirs("results")
        plot_name = f"results/{model_str}_{policy_str}_attitudes.png"

    plt.figure(plot_name, figsize=(10, 6))
    plt.ylim(0, 1)
    
    linestyles = [':', ':', ':', ':', ':']  # No markers and solid lines for individual curves
    colors = ['#4169E1', '#FFD700', '#FF7F50']
    avg_colors = ['#00008B', '#B8860B', '#CD5C5C']

    against = [a[0] for a in attitude_dist]
    swing = [a[1] for a in attitude_dist]
    support = [a[2] for a in attitude_dist]
    x = np.arange(len(attitude_dist))
    # breakpoint()
    # Plot individual curves with no markers
    plt.plot(x, against, label="Against" if run_idx == 0 else "", color=colors[0], linestyle=linestyles[run_idx % len(linestyles)])
    plt.plot(x, swing, label='Swing' if run_idx == 0 else "", color=colors[1], linestyle=linestyles[run_idx % len(linestyles)])
    plt.plot(x, support, label='Support' if run_idx == 0 else "", color=colors[2], linestyle=linestyles[run_idx % len(linestyles)])
    if average:
        # Compute and update the average curves from the existing lines
        avg_against, avg_swing, avg_support = compute_running_average_from_existing_plot()

        if run_idx == 4:
            for i, avg_y_data in enumerate([avg_against, avg_swing, avg_support]):
                for j, y in enumerate(avg_y_data):
                    plt.text(x[j], y, f'{y:.2f}', ha='center', va='bottom', color=avg_colors[i])

        if avg_against is not None:
            # Check if this is the first time the average curves are being plotted
            first_plot = len(plt.gca().get_lines()) <= 3  # If no average curves exist, it's the first plot
            update_or_add_average_curves(x, avg_against, avg_swing, avg_support, run_idx, first_plot)

    # Only display the unique labels once in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))  # Remove duplicates
    plt.legend(unique_labels.values(), unique_labels.keys())
    
    # Save the figure
    plt.savefig(plot_name)
    if average:
        print("Average Against:", avg_against)
        print("Average Swing:", avg_swing)
        print("Average Support:", avg_support)
        return avg_against, avg_swing, avg_support 
    return None, None, None


def plot_d_curves(d, average_support_data, model_str, policy_heads, temperature=None, file_dir=None):
    """
    Plot the support curves for each policy.
    :param support_data: List of lists, each sublist contains the support curve for a policy
    """

    plot_name = f"{file_dir}/{d}_curves.png"
    plt.figure(plot_name, figsize=(10, 6))
    plt.ylim(0, 1)
    colors = ['#4169E1', '#FFD700', '#FF7F50']
    for i, support in enumerate(average_support_data):
        # breakpoint()
        x = np.arange(len(support))
        plt.plot(x, support, label=f'{d}, Policy={policy_heads[i]}', color=colors[i])
        for j, s in enumerate(support):
            plt.text(j, s, f'{s:.2f}', ha='center', va='bottom', color=colors[i])
            
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_name)