import numpy as np
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
from matplotlib.gridspec import GridSpec
import pandas as pd

def plot_policy():
    # Data
    models = ["Llama-3-8B", "Llama-3-8B-AB", "Llama-3.1-8B", "Qwen-2.5-7B", "GPT-4o", "Claude-3.5-Haiku", "Phi-3.5-mini"]
    policies = ["Incentive", "Ambassador", "Mandate"]

    # Data from table
    weak_data = np.array([
        [1.4, 1.1, 8.73, -3, -2.27, -22.8, -1.6],
        [3.46, 3.3, 5.6, -4.5, np.nan, np.nan, 1.27],
        [1.3, 1.46, 13.2, -1.3, np.nan, np.nan, -0.93]
    ])

    strong_data = np.array([
        [5.4, 3.86, 15.3, 5.1, -2.67, -6.87, -1.33],
        [6.4, 5.8, 17, 0.6, np.nan, np.nan, 4.67],
        [7, 3.86, 20.8, 0.6, np.nan, np.nan, -0.2]
    ])

    difference_data = strong_data - weak_data

    # Define colors
    colors = ["#E85D04", "#F48C06", "#0277BD", "#A9A9A9"]  
    # Create subplots
    # fig, axes = plt.subplots(nrows=3, figsize=(14, 8), sharex=True)
    fig, axes = plt.subplots(nrows=3, figsize=(14, 8), sharex=True)
    bar_width = 0.25
    indices = np.linspace(0, len(models) - 1, len(models))

    for i, ax in enumerate(axes[:1]):
        texts = []
        # Plot bars
        weak_bars = ax.bar(indices - bar_width, weak_data[i], width=bar_width, label="Weak", color=colors[0])
        strong_bars = ax.bar(indices, strong_data[i], width=bar_width, label="Strong", color=colors[1])
        diff_bars = ax.bar(indices + bar_width, difference_data[i], width=bar_width, label="Difference", color=colors[2])

        # Annotate bars with values
        for idx, bars in enumerate([weak_bars, strong_bars, diff_bars]):
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    x_pos = bar.get_x() + bar.get_width() / 2
                    va = 'bottom' if height >= 0 else 'top'
                    offset = 1 if height >= 0 else -1
                    text = ax.text(x_pos, height + offset, f"{height:.1f}", ha='center', va=va, fontsize=18, color="black")
                    texts.append(text)
                # Annotate bars with values, ensuring the middle bar is opposite to the closest bar

        # Handle NaN values (N/A bars)
        for j in range(len(models)):
            if np.isnan(weak_data[i, j]):
                ax.bar(indices[j] - bar_width, 5, width=bar_width, color=colors[3], hatch="////")
                texts.append(ax.text(indices[j] - bar_width, 5.5, "N/A", ha="center", fontsize=18, color="black"))
            if np.isnan(strong_data[i, j]):
                ax.bar(indices[j], 5, width=bar_width, color=colors[3], hatch="////")
                texts.append(ax.text(indices[j], 5.5, "N/A", ha="center", fontsize=18, color="black"))
            if np.isnan(difference_data[i, j]):
                ax.bar(indices[j] + bar_width, 5, width=bar_width, color=colors[3], hatch="////")
                texts.append(ax.text(indices[j] + bar_width, 5.5, "N/A", ha="center", fontsize=18, color="black"))

        ax.set_title(f"{policies[i]} Policy", fontsize=20)
        # ax.set_ylabel("Hesitancy Reduction ($\\Delta H$) ", fontsize=14)
        ax.set_xticks(indices)
        ax.set_xticklabels(models, fontsize=20)
       
        ax.grid(axis='y', linestyle="--", alpha=0.6)

        max_y = np.nanmax([weak_data[i], strong_data[i], difference_data[i]]) + 7
        min_y = np.nanmin([weak_data[i], strong_data[i], difference_data[i]]) - 6.5
        ax.set_ylim(min_y, max_y)
        ax.tick_params(axis='both', which='major', labelsize=18)  # Adjusts major ticks
        ax.set_xlim(indices[0] - bar_width * 2, indices[-1] + bar_width * 2)
    plt.margins(x=0)
    axes[1].set_ylabel("Average Hesitancy Reduction ($\\Delta H$)", fontsize=20)
    axes[0].legend(fontsize=16)
    plt.tight_layout()
    plt.savefig("policy.png", bbox_inches="tight")




def plot_news():
    # Data
    models = ["Llama-3-8B", "Llama-3-8B-AB", "Llama-3.1-8B", "Qwen-2.5-7B"]
    categories = ["Positive News", "Negative News"]
    num_models = len(models)
    num_categories = len(categories)

    # Data
    data = np.array([
        [0.86, -0.93],
        [2.4, -3.0],
        [3.86, -2.53],
        [1.86, -1.73]
    ])

    # Define bar width and positions
    bar_width = 0.3
    ind = np.arange(num_models)  # Positions for the models

    # Define colors (color-weak friendly oranges)
    colors = ["#FFC57E", "#FF8C00"]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot bars
    for i in range(num_categories)[::-1]:
        bars = ax.barh(ind + i * bar_width - bar_width, data[:, i], height=bar_width, label=categories[i], color=colors[i])
        
        # Annotate bars to the right
        for bar in bars:
            width = bar.get_width()
            ax.text(width + (0.05 if width >= 0 else -0.05), 
                    bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}', 
                    va='center', ha='left' if width >= 0 else 'right', fontsize=12)

    # ax.axvline(x=2, color='black', linestyle='--', linewidth=1.5)

    # Increase font sizes
    font_size = 16
    ax.set_yticks(ind)
    ax.set_yticklabels(models, rotation=45, ha="right", fontsize=font_size)
    ax.set_xlabel("Average Hesitancy Reduction ($\\Delta$H)", fontsize=font_size)
    ax.set_title("Hesitancy Reduction ($\\Delta$H) with Different News", fontsize=font_size + 2)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.legend(fontsize=13, loc="lower right")

    ax.set_xlim(left=min(data.min() - 2, 0))
    ax.set_xlim(right=max(data.max() + 4, 0))

    plt.grid(axis='x')
    plt.savefig("news_diff.png", bbox_inches="tight")
    fig.subplots_adjust(hspace=0.05)
    plt.tight_layout()
    plt.show()


def plot_ratings():
    # Data setup
    models = ["Llama-3", "Llama-3-AB", "Llama-3.1", "Qwen-2.5"]
    categories = ["Attitude", "Memory", "Conversation"]
    runs = ["Strong Incentive", "Strong Ambassador", "Strong Mandate"]

    # Generate random example data (4 models × 3 runs x 3 categories)
    data = np.array([
        [3.60, 4.62, 4.95],
        [3.42, 4.65, 4.91],
        [4.50, 4.67, 4.95],
        [4.28, 4.64, 4.96]
    ])

    # Create x-axis labels that include both category and run info
    xtick_labels = [f"{cat}" for cat in categories]

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(5, 3))
    heatmap = sns.heatmap(data, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=xtick_labels, 
                          yticklabels=models, linewidths=0.5, ax=ax, cbar_kws={"label": "Response Quality Score"}, 
                          annot_kws={"size": 18}
                        )
    plt.xticks(fontsize=12)  # Increase category font size
    plt.yticks(fontsize=12)

    heatmap.collections[0].colorbar.set_label("Response Quality Score", fontsize=14)

    # increase annotation font size
    # Formatting
    ax.set_xlabel("Evaluation Categories", fontsize=16)
    ax.set_ylabel("Models", fontsize=16)
    ax.set_title("Heatmap of Response Quality Rated by GPT-4o", fontsize=16)
    plt.savefig("ratings.png", bbox_inches="tight")
    plt.show()

def plot_combined():
    # Data for the top horizontal bar chart (from plot_policy)
    models_policy = ["Llama-3-8B", "Llama-3-8B-AB", "Llama-3.1-8B", "Qwen-2.5-7B", "GPT-4o", "Claude-3.5-Haiku", "Phi-3.5-mini"]
    policies = ["Incentive", "Ambassador", "Mandate"]
    weak_data = np.array([
        [1.4, 1.1, 8.73, -3, -2.27, -22.8, -1.6],
        [3.46, 3.3, 5.6, -4.5, np.nan, np.nan, 1.27],
        [1.3, 1.46, 13.2, -1.3, np.nan, np.nan, -0.93]
    ])
    strong_data = np.array([
        [5.4, 3.86, 15.3, 5.1, -2.67, -6.87, -1.33],
        [6.4, 5.8, 17, 0.6, np.nan, np.nan, 4.67],
        [7, 3.86, 20.8, 0.6, np.nan, np.nan, -0.2]
    ])
    difference_data = strong_data - weak_data
    colors_policy = ["#E85D04", "#F48C06", "#0277BD", "#A9A9A9"]
    bar_width = 0.25
    indices_policy = np.arange(len(models_policy))

    # Data for the bottom-left bar plot (from plot_news)
    models_news = ["Llama-3-8B", "Llama-3-8B-AB", "Llama-3.1-8B", "Qwen-2.5-7B"]
    categories_news = ["Positive News", "Negative News"]
    data_news = np.array([
        [0.86, -0.93],
        [2.4, -3.0],
        [3.86, -2.53],
        [1.86, -1.73]
    ])
    colors_news = ["#FFC57E", "#FF8C00"]
    bar_width_news = 0.3
    indices_news = np.arange(len(models_news))

    # Data for the bottom-right heatmap (from plot_ratings)
    models_ratings = ["Llama-3", "Llama-3-AB", "Llama-3.1", "Qwen-2.5"]
    categories_ratings = ["Attitude", "Memory", "Conversation"]
    data_ratings = np.array([
        [3.60, 4.62, 4.95],
        [3.42, 4.65, 4.91],
        [4.50, 4.67, 4.95],
        [4.28, 4.64, 4.96]
    ])

    # Create figure and GridSpec
    fig = plt.figure(figsize=(30, 10))
    gs = GridSpec(2, 3, height_ratios=[1, 1])
    title_size = 23
    number_label_size = 20
    models_label_size = 16
    annotation_size = 20
    legend_size = 16

    # Top plot: Horizontal bar chart (Policy)
    ax0 = fig.add_subplot(gs[0, 1:])
    i = 0  # Selecting the first policy
    weak_bars = ax0.bar(indices_policy - bar_width, weak_data[i], width=bar_width, label="Weak", color=colors_policy[0])
    strong_bars = ax0.bar(indices_policy, strong_data[i], width=bar_width, label="Strong", color=colors_policy[1])
    diff_bars = ax0.bar(indices_policy + bar_width, difference_data[i], width=bar_width, label="Difference", color=colors_policy[2])
    for idx, bars in enumerate([weak_bars, strong_bars, diff_bars]):
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                x_pos = bar.get_x() + bar.get_width() / 2
                va = 'bottom' if height >= 0 else 'top'
                offset = -0.05 if height >= 0 else -0.1
                ax0.text(x_pos, height + offset, f"{height:.1f}", ha='center', va=va, fontsize=16, color="black")
    ax0.set_title(r"$\bf{(P2)} $ "+ f"Hesitancy Reduction with Various Effort Levels of the {policies[i]} Policy", fontsize=title_size)
    ax0.set_xticks(indices_policy)
    ax0.set_xticklabels(models_policy, fontsize=models_label_size)
    ax0.set_ylabel("Hesitancy Reduction ($\\Delta H$)", fontsize=number_label_size)
    ax0.set_ylim(-28, 20)
    ax0.legend(fontsize=18)
    ax0.grid(axis='y', linestyle="--", alpha=0.6)
    ax0.tick_params(axis='both', which='major', labelsize=18)

    # Bottom-left plot: Horizontal bar chart (News)
    ax1 = fig.add_subplot(gs[1, 1])
    for j in range(len(categories_news)):
        bars = ax1.barh(indices_news + j * bar_width_news - bar_width_news, data_news[:, j], height=bar_width_news, label=categories_news[j], color=colors_news[j])
        for bar in bars:
            width = bar.get_width()
            ax1.text(width + (0.05 if width >= 0 else -0.1),
                     bar.get_y() + bar.get_height()/2,
                     f'{width:.2f}',
                     va='center', ha='left' if width >= 0 else 'right', fontsize=20)
            
    ax1.set_yticks(indices_news)
    ax1.set_yticklabels(models_news, fontsize=models_label_size, rotation=45, ha="right")
    ax1.set_xlabel("Average Hesitancy Reduction ($\\Delta H$)", fontsize=number_label_size)
    ax1.set_title(r"$\bf{(P3)} $ "+"Hesitancy Reduction with Different News", fontsize=title_size)
    ax1.legend(fontsize=16)
    ax1.set_xlim(-4, 5)
    ax1.grid(axis='x', linestyle="--", alpha=0.6)

    # Bottom-right plot: Heatmap (Ratings)
    ax2 = fig.add_subplot(gs[1, 2])
    heatmap = sns.heatmap(data_ratings, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=categories_ratings, yticklabels=models_ratings, linewidths=0.5, ax=ax2, cbar_kws={"label": "Response Quality Score"}, annot_kws={"size": 20})
    heatmap.collections[0].colorbar.set_label("Response Quality Score", fontsize=16)
    heatmap.collections[0].colorbar.ax.tick_params(labelsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.set_xlabel("Evaluation Categories", fontsize=number_label_size)
    ax2.set_title(r"$\bf{(P4)} $ "+"Response Quality Rated by GPT-4o", fontsize=title_size)
    bold_indices = [(0, 2), (1, 2), (2,3)]

    # Iterate over text annotations and set font weight for specific entries
    for text in ax2.texts:
        row, col = map(int, text.get_position())
        if (row, col) in bold_indices:
            text.set_fontweight('bold')

    ax2.set_yticklabels(models_ratings, fontsize=14, rotation=45, ha="right")

    data = {
        'Model': [
            'Llama-3-8B', 'Llama-3-8B-AB', 'Llama-3.1-8B',
            'Llama-3.2-3B', 'Phi-3.5-mini', 'Qwen-2.5-7B',
            'GPT-4o mini', 'GPT-4o', 'Claude-3.5-Haiku'
        ],
        0.1: [32.8, 22.2, 4.2, -34.2, -2.8, 1.2, 51.4, 23.6, 6.8],
        0.5: [29.0, 13.8, 3.6, -37.4, 0.6, 1.6, 49.0, 24.8, 12.6],
        0.7: [22.6, 11.8, 0.2, -32.0, 4.6, 0.6, 45.0, 21.2, 11.6],
        1.0: [20.0, 14.0, 2.6, -30.0, 2.2, 1.2, 39.0, 19.4, 12.0],
        1.5: [16.0, 5.2, 3.2, -22.4, 3.2, 3.0, 32.0, 10.0, 9.2],
        2.0: [13.2, 3.6, 0.2, -20.4, 4.2, 3.0, 26.6, 9.8, 11.2]
    }

    df = pd.DataFrame(data)
    df.set_index('Model', inplace=True)
    ax3 = plt.subplot(gs[:, 0])
    heatmap = sns.heatmap(df, cmap='Spectral', annot=True, center=0, fmt=".2f", linewidths=0.5, cbar_kws={'label': 'Mean Absolute Error (MAE)'}, annot_kws={"size": 20})
    heatmap.collections[0].colorbar.set_label("Mean Error", fontsize=18)
    heatmap.collections[0].colorbar.ax.tick_params(labelsize=14)
    
    bold_indices = [(5, 0), (5, 1), (2,2), (5,2), (5,3), (1, 4), (2, 5), (5,6), (5,7), (0, 8)]  # Example: bold annotations at positions (0,1) and (2,2)

    # Iterate over text annotations and set font weight for specific entries
    for text in ax3.texts:
        row, col = map(int, text.get_position())
        if (row, col) in bold_indices:
            text.set_fontweight('bold')

    ax3.tick_params(axis='both', which='major', labelsize=14)
    ax3.set_title(r"$\bf{(P1)} $ "+'Attitude Modulation', fontsize=title_size)
    ax3.set_xlabel('Modulating Temperature', fontsize=number_label_size)
    ax3.set_xticklabels(df.columns, fontsize=number_label_size)
    ax3.set_ylabel('')
    ax3.set_yticklabels(data['Model'], fontsize=16, rotation=45, ha='right')
    plt.savefig("attitude_modulation_heatmap.png", bbox_inches="tight")



def plot_average(trimmed_simulation_avg, trimmed_weeks, simu_weeks, average_color, label="Simulation Average"):    
    # Plot the simulation average with markers only at specified indices
    plt.plot(trimmed_weeks, trimmed_simulation_avg, label=label,
             linestyle='-', linewidth=3, color=average_color)
    plt.scatter(simu_weeks, [trimmed_simulation_avg[i] for i in simu_weeks], 
                color=average_color, marker="o", zorder=3)
    
    # # Annotate Simulation Average (selected key points)
    # for i in simu_weeks:  
    #     offset = -0.02 if i == 1 else 0.02
    #     plt.text(i, trimmed_simulation_avg[i] + offset, 
    #              f"{trimmed_simulation_avg[i]:.2f}", 
    #              fontsize=12, color=average_color, ha='center')

def plot_llama_qwen():
    # Data for the plots
    data = {
        'llama-weak-incentive': [0.91, 0.73, 0.6, 0.56, 0.52, 0.45, 0.44, 0.45, 0.42, 0.42, 0.39, 0.41, 0.39, 0.39, 0.37, 0.4, 0.4, 0.4, 0.37, 0.36, 0.36], 
        'llama-weak-ambassador': [0.91, 0.74, 0.58, 0.55, 0.51, 0.46, 0.47, 0.45, 0.43, 0.41, 0.42, 0.4, 0.39, 0.37, 0.39, 0.39, 0.4, 0.42, 0.41, 0.4, 0.42], 
        'llama-weak-mandate': [0.91, 0.73, 0.58, 0.53, 0.51, 0.48, 0.42, 0.42, 0.39, 0.39, 0.38, 0.36, 0.36, 0.36, 0.38, 0.37, 0.39, 0.39, 0.36, 0.36, 0.33], 
        'llama-strong-mandate': [0.91, 0.73, 0.58, 0.53, 0.51, 0.48, 0.4, 0.39, 0.34, 0.32, 0.3, 0.3, 0.28, 0.29, 0.29, 0.26, 0.3, 0.29, 0.27, 0.27, 0.27], 
        'llama-strong-ambassador': [0.91, 0.74, 0.58, 0.55, 0.51, 0.46, 0.4, 0.38, 0.37, 0.32, 0.35, 0.31, 0.31, 0.34, 0.33, 0.31, 0.31, 0.3, 0.3, 0.28, 0.3], 
        'llama-strong-incentive': [0.91, 0.74, 0.59, 0.54, 0.49, 0.46, 0.39, 0.37, 0.36, 0.32, 0.34, 0.34, 0.31, 0.34, 0.33, 0.3, 0.31, 0.32, 0.33, 0.29, 0.3], 
        'qwen-weak-mandate': [0.75, 0.54, 0.51, 0.52, 0.49, 0.45, 0.45, 0.48, 0.46, 0.43, 0.46, 0.46, 0.49, 0.51, 0.47, 0.46, 0.48, 0.48, 0.47, 0.46, 0.46], 
        'qwen-weak-ambassador': [0.75, 0.55, 0.51, 0.51, 0.47, 0.46, 0.48, 0.48, 0.48, 0.47, 0.5, 0.51, 0.48, 0.5, 0.5, 0.54, 0.55, 0.54, 0.53, 0.5, 0.48], 
        'qwen-weak-incentive': [0.75, 0.55, 0.51, 0.5, 0.47, 0.44, 0.48, 0.47, 0.46, 0.47, 0.5, 0.5, 0.48, 0.53, 0.53, 0.51, 0.54, 0.54, 0.51, 0.52, 0.54], 
        'qwen-strong-incentive': [0.74, 0.57, 0.52, 0.51, 0.51, 0.48, 0.47, 0.44, 0.4, 0.43, 0.43, 0.4, 0.41, 0.42, 0.44, 0.41, 0.49, 0.42, 0.44, 0.42, 0.41], 
        'qwen-strong-ambassador': [0.75, 0.55, 0.51, 0.51, 0.47, 0.46, 0.47, 0.48, 0.46, 0.44, 0.47, 0.48, 0.44, 0.48, 0.47, 0.44, 0.49, 0.47, 0.46, 0.47, 0.44], 
        'qwen-strong-mandate': [0.75, 0.54, 0.51, 0.52, 0.49, 0.45, 0.43, 0.45, 0.44, 0.41, 0.45, 0.41, 0.4, 0.44, 0.43, 0.44, 0.48, 0.45, 0.46, 0.42, 0.44]
    }

    # Splitting data into two models
    llama_data = {k: v for k, v in data.items() if "llama" in k}
    qwen_data = {k: v for k, v in data.items() if "qwen" in k}

    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(10, 8.5), sharex=True)
    fig.set_constrained_layout(True)

    # Larger font size
    plt.rcParams.update({'font.size': 18, 'xtick.labelsize': 18, 'ytick.labelsize': 18})


    colors = ["#1f77b4", "#d62728", "#2ca02c"] # Blue, Red, Green

    colors_map = {
        "llama-weak-incentive": colors[0],
        "llama-weak-ambassador": colors[1],
        "llama-weak-mandate": colors[2],
        "llama-strong-incentive": colors[0],
        "llama-strong-ambassador": colors[1],
        "llama-strong-mandate": colors[2],
        "qwen-weak-incentive": colors[0],
        "qwen-weak-ambassador": colors[1],
        "qwen-weak-mandate": colors[2],
        "qwen-strong-incentive": colors[0],
        "qwen-strong-ambassador": colors[1],
        "qwen-strong-mandate": colors[2]
    }

    # LLaMA-3.1-8B-Instruct plot
    for key, values in llama_data.items():
        if "weak" in key:
            axes[0].plot(values, label=key.replace("llama-", "").replace("_", " ").title(), linestyle='--', color=colors_map[key], linewidth=3)
        else:
            axes[0].plot(values, label=key.replace("llama-", "").replace("_", " ").title(), color=colors_map[key], linewidth=3)
    
    # set x lim
    axes[0].set_xlim(0, 20)
    axes[0].set_title("LLaMA-3.1-8B-Instruct", fontsize=20)
    # axes[0].set_ylabel("Hesitancy (H)", fontsize=20)
    axes[0].legend(fontsize=16, loc="upper right", frameon=False)
    axes[0].grid(True)
    axes[0].axvline(x=5, color='black', linestyle='--', linewidth=2)
    axes[0].text(5.2, plt.ylim()[1] * 0.9, 'Warmup Cutoff', rotation=90, fontsize=16, color='black', verticalalignment='top')
    # Qwen-2.5-7B-Instruct plot
    for key, values in qwen_data.items():
        if "weak" in key:
            axes[1].plot(values, label=key.replace("qwen-", "").replace("_", " ").title(), linestyle='--', color=colors_map[key], linewidth=3)
        else:
            axes[1].plot(values, label=key.replace("qwen-", "").replace("_", " ").title(), color=colors_map[key], linewidth=3)
    
    axes[1].set_xlim(0, 20)
    axes[1].set_ylim(0.38, 0.8)
    axes[1].set_title("Qwen-2.5-7B-Instruct", fontsize=20)
    axes[1].set_xlabel("Timesteps (Weeks)", fontsize=20)
    
    axes[1].legend(fontsize=14.5, loc="upper right", frameon=False)
    axes[1].grid(True)
    axes[1].axvline(x=5, color='black', linestyle='--', linewidth=2)
    axes[1].text(5.2, plt.ylim()[1] * 0.95, 'Warmup Cutoff', rotation=90, fontsize=16, color='black', verticalalignment='top')
    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=16)  # Adjusts major ticks
    fig.supylabel("Average Hesitancy Percentage (H)", fontsize=20)
    fig.subplots_adjust(hspace=0.05)
    plt.tight_layout()
    
   
    plt.savefig("llama_qwen.png", bbox_inches="tight", pad_inches=0.01)


def plot_curve():
    incentive_run_1 = [0.87, 0.77, 0.61, 0.57, 0.53, 0.46, 0.36, 0.37, 0.38, 0.29, 0.31, 0.35, 
                0.32, 0.28, 0.38, 0.29, 0.38, 0.27, 0.31, 0.25, 0.25, 0.27, 0.25, 0.24, 
                0.19, 0.23, 0.16, 0.22, 0.21, 0.24, 0.25, 0.2, 0.12, 0.21, 0.23, 0.18]

    incentive_run_2 = [0.91, 0.74, 0.61, 0.56, 0.57, 0.49, 0.43, 0.45, 0.43, 0.41, 0.4, 0.36, 
                0.32, 0.34, 0.4, 0.36, 0.43, 0.36, 0.39, 0.27, 0.37, 0.25, 0.28, 0.27, 
                0.25, 0.26, 0.24, 0.25, 0.28, 0.25, 0.27, 0.29, 0.32, 0.25, 0.2, 0.21]

    incentive_run_3 = [0.95, 0.76, 0.55, 0.52, 0.51, 0.44, 0.4, 0.3, 0.33, 0.33, 0.26, 0.3, 
                0.31, 0.26, 0.31, 0.32, 0.34, 0.26, 0.29, 0.3, 0.26, 0.28, 0.3, 0.27, 
                0.26, 0.23, 0.19, 0.18, 0.24, 0.21, 0.22, 0.23, 0.16, 0.2, 0.22, 0.15]

    incentive_run_4 = [0.9, 0.68, 0.52, 0.49, 0.43, 0.4, 0.34, 0.38, 0.36, 0.36, 0.33, 0.37, 
                0.31, 0.33, 0.31, 0.33, 0.32, 0.34, 0.28, 0.31, 0.32, 0.27, 0.3, 0.26, 
                0.22, 0.27, 0.23, 0.25, 0.25, 0.2, 0.19, 0.28, 0.18, 0.17, 0.22, 0.21]

    incentive_run_5 = [0.93, 0.74, 0.57, 0.52, 0.48, 0.47, 0.42, 0.38, 0.31, 0.31, 0.34, 0.3, 
                0.26, 0.35, 0.26, 0.27, 0.25, 0.26, 0.32, 0.27, 0.27, 0.19, 0.24, 0.16, 
                0.19, 0.19, 0.23, 0.17, 0.2, 0.28, 0.17, 0.2, 0.22, 0.21, 0.25, 0.21]
    
    ambassador_run_1 = [0.87, 0.77, 0.61, 0.57, 0.53, 0.46, 0.42, 0.36, 0.39, 0.43, 0.32, 0.36, 0.36, 0.29, 0.36, 0.29, 0.3, 0.28, 0.3, 0.26, 0.26, 0.26, 0.29, 0.26, 0.18, 0.27, 0.15, 0.21, 0.2, 0.26, 0.24, 0.16, 0.25, 0.28, 0.25, 0.2]
    ambassador_run_2 =[0.91, 0.74, 0.61, 0.56, 0.57, 0.49, 0.43, 0.43, 0.42, 0.45, 0.43, 0.41, 0.36, 0.38, 0.4, 0.32, 0.37, 0.34, 0.29, 0.32, 0.36, 0.33, 0.28, 0.2, 0.26, 0.28, 0.2, 0.23, 0.29, 0.24, 0.26, 0.22, 0.26, 0.26, 0.19, 0.21]
    ambassador_run_3 =[0.95, 0.76, 0.55, 0.52, 0.51, 0.44, 0.41, 0.37, 0.31, 0.41, 0.32, 0.3, 0.37, 0.3, 0.35, 0.37, 0.35, 0.36, 0.3, 0.35, 0.32, 0.32, 0.37, 0.31, 0.26, 0.31, 0.3, 0.27, 0.24, 0.25, 0.26, 0.27, 0.23, 0.24, 0.2, 0.28]
    ambassador_run_4 =[0.9, 0.68, 0.52, 0.49, 0.43, 0.4, 0.38, 0.39, 0.38, 0.33, 0.33, 0.36, 0.33, 0.32, 0.32, 0.34, 0.34, 0.29, 0.34, 0.31, 0.29, 0.3, 0.26, 0.25, 0.22, 0.22, 0.24, 0.26, 0.22, 0.21, 0.31, 0.21, 0.26, 0.23, 0.29, 0.28]
    ambassador_run_5 =[0.93, 0.74, 0.57, 0.52, 0.48, 0.47, 0.41, 0.41, 0.34, 0.33, 0.33, 0.3, 0.3, 0.36, 0.35, 0.32, 0.32, 0.4, 0.33, 0.28, 0.33, 0.29, 0.26, 0.19, 0.23, 0.22, 0.21, 0.23, 0.25, 0.25, 0.19, 0.28, 0.3, 0.25, 0.26, 0.24]
    
    mandate_run_1 = [0.87, 0.77, 0.61, 0.57, 0.53, 0.46, 0.39, 0.39, 0.35, 0.33, 0.33, 0.28, 0.33, 0.25, 0.34, 0.32, 0.32, 0.3, 0.3, 0.24, 0.27, 0.22, 0.24, 0.26, 0.21, 0.25, 0.25, 0.21, 0.17, 0.27, 0.18, 0.17, 0.14, 0.2, 0.23, 0.16]
    mandate_run_2 = [0.91, 0.74, 0.61, 0.56, 0.57, 0.49, 0.44, 0.42, 0.39, 0.37, 0.44, 0.38, 0.29, 0.35, 0.33, 0.25, 0.33, 0.32, 0.28, 0.21, 0.32, 0.2, 0.23, 0.25, 0.22, 0.22, 0.18, 0.23, 0.25, 0.23, 0.22, 0.25, 0.29, 0.19, 0.2, 0.23]
    mandate_run_3 = [0.95, 0.76, 0.55, 0.52, 0.51, 0.44, 0.43, 0.33, 0.34, 0.32, 0.33, 0.3, 0.3, 0.25, 0.33, 0.3, 0.3, 0.32, 0.27, 0.33, 0.24, 0.27, 0.25, 0.22, 0.28, 0.18, 0.27, 0.22, 0.22, 0.17, 0.26, 0.25, 0.17, 0.22, 0.19, 0.21]
    mandate_run_4 = [0.9, 0.68, 0.52, 0.49, 0.43, 0.4, 0.38, 0.39, 0.31, 0.34, 0.37, 0.34, 0.3, 0.32, 0.27, 0.35, 0.31, 0.32, 0.3, 0.29, 0.32, 0.31, 0.23, 0.23, 0.21, 0.17, 0.22, 0.2, 0.26, 0.22, 0.27, 0.18, 0.15, 0.17, 0.18, 0.19]
    mandate_run_5 = [0.93, 0.74, 0.57, 0.52, 0.48, 0.47, 0.41, 0.38, 0.36, 0.31, 0.3, 0.28, 0.28, 0.34, 0.27, 0.25, 0.27, 0.22, 0.33, 0.27, 0.24, 0.23, 0.17, 0.21, 0.23, 0.18, 0.2, 0.15, 0.21, 0.26, 0.12, 0.24, 0.26, 0.18, 0.18, 0.2]

    plt.figure(figsize=(10, 6))

    incentive_simulation_avg = np.mean([incentive_run_1, incentive_run_2, incentive_run_3, incentive_run_4, incentive_run_5], axis=0)
    ambassador_simulation_avg = np.mean([ambassador_run_1, ambassador_run_2, ambassador_run_3, ambassador_run_4, ambassador_run_5], axis=0)
    mandate_simulation_avg = np.mean([mandate_run_1, mandate_run_2, mandate_run_3, mandate_run_4, mandate_run_5], axis=0)
    

    trimmed_weeks = np.arange(0, 31)
    trimmed_incentive_avg = incentive_simulation_avg[5:]
    trimmed_ambassador_avg = ambassador_simulation_avg[5:]
    trimmed_mandate_avg = mandate_simulation_avg[5:]

    simu_weeks = [0, 4, 8, 13, 17, 21, 26, 30]

    incentive_color = "#1E88E5"  # Deep Blue
    real_color = "#E6851F"     
    ambassador_color = "#4FC3F7"  
    mandate_color = "#0B346E"  


    plot_average(trimmed_incentive_avg, trimmed_weeks, simu_weeks, incentive_color, label="Strong Incentive (MAE=3.60%)")
    plot_average(trimmed_ambassador_avg, trimmed_weeks, simu_weeks, ambassador_color, label="Strong Ambassador (MAE=4.20%)")
    plot_average(trimmed_mandate_avg, trimmed_weeks, simu_weeks, mandate_color, label="Strong Mandate (MAE=2.82%)")
    
    real_data = [0.45, 0.38, 0.31, 0.24, 0.2, 0.19, 0.18, 0.18]
    real_weeks = [0, 4, 8, 13, 17, 21, 26, 30]

    # Plot the real data
    plt.plot(real_weeks, real_data, label="Real Data",
             linestyle='--', linewidth=3, color=real_color, marker='o')

    # Labels and legend
    plt.xlabel("Weeks", fontsize=20)
    plt.ylabel("Average Hesitancy Percentage (H)", fontsize=20)
    plt.title("Hesitancy Trends Over Time, Simulated with Llama-3.1-8B-Instruct", fontsize=19)
    plt.tick_params(axis='both', which='major', labelsize=18)  # Adjusts major ticks
    plt.legend(fontsize=18)
    plt.grid(True)

    plt.savefig("curve.png", bbox_inches="tight")

    # print MAE with the real data
    selected_trimmed_incentive_avg = [trimmed_incentive_avg[i] for i in simu_weeks]
    selected_trimmed_ambassador_avg = [trimmed_ambassador_avg[i] for i in simu_weeks]
    selected_trimmed_mandate_avg = [trimmed_mandate_avg[i] for i in simu_weeks]
    print("Incentive MAE:", np.mean(np.abs(np.array(selected_trimmed_incentive_avg) - np.array(real_data))))
    print("Ambassador MAE:", np.mean(np.abs(np.array(selected_trimmed_ambassador_avg) - np.array(real_data))))
    print("Mandate MAE:", np.mean(np.abs(np.array(selected_trimmed_mandate_avg) - np.array(real_data))))


    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plotting script for policy and news data")
    parser.add_argument("plot_type", type=int, help="Type of plot to generate: policy, news")
    args = parser.parse_args()
    if args.plot_type == 0:
        plot_policy()
    elif args.plot_type == 1:
        plot_news()
    elif args.plot_type == 2:
        plot_ratings()
    elif args.plot_type == 3:
        plot_curve()
    elif args.plot_type == 4:
        plot_llama_qwen()
    elif args.plot_type == 5:
        plot_combined()