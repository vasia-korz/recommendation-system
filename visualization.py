"""
Module for visualizing the performance of hybrid models compared to baseline models.

Includes functions for generating visualizations of model performance metrics
and comparing them across different user record thresholds and cluster sizes.

#### Functions:
- hybrid_vs_baselines_visualization: Generates bar plots comparing MAE, MSE, and accuracy of different models.
- hybrid_metrics_by_user_records: Plots MSE, MAE, and accuracy against the number of records in the training set.
- hybrid_metrics_by_cluster_size: Plots MSE and MAE against the number of clusters for hybrid models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def hybrid_vs_baselines_visualization(results: dict):
    """
    Generates bar plots comparing MAE, MSE, and accuracy of different models.

    #### Parameters:
    - results: dict
        Dictionary containing model names as keys and another dictionary as values,
        where the inner dictionary has metric names ('mae', 'mse', 'accuracy') as keys and their values.

    #### Returns:
    None. Displays a matplotlib plot.
    """
    result_names = results.keys()
    results = results.values()

    data = []
    for name, result in zip(result_names, results):
        for metric, value in result.items():
            data.append({"Model": name, "Metric": metric, "Value": value})

    df = pd.DataFrame(data)

    sns.set(style="whitegrid")
    sns.set_context("talk")

    colors = sns.color_palette("Set2", len(result_names))

    fig, axs = plt.subplots(1, 3, figsize=(12, 5))  # Adjust width for horizontal layout

    sns.barplot(x="Model", y="Value", hue="Model", data=df[df["Metric"] == "mae"], palette=colors, width=0.5, ax=axs[0], dodge=False)
    axs[0].set_title("MAE", fontsize=20)
    axs[0].set_ylabel("MAE", fontsize=16)
    axs[0].set_xlabel("")
    axs[0].tick_params(axis='y', labelsize=14)
    axs[0].set_xticks([])

    sns.barplot(x="Model", y="Value", hue="Model", data=df[df["Metric"] == "mse"], palette=colors, width=0.5, ax=axs[1], dodge=False)
    axs[1].set_title("MSE", fontsize=20)
    axs[1].set_ylabel("MSE", fontsize=16)
    axs[1].set_xlabel("")
    axs[1].tick_params(axis='y', labelsize=14)
    axs[1].set_xticks([])

    sns.barplot(x="Model", y="Value", hue="Model", data=df[df["Metric"] == "accuracy"], palette=colors, width=0.5, ax=axs[2], dodge=False)
    axs[2].set_title("Accuracy", fontsize=20)
    axs[2].set_ylabel("Accuracy", fontsize=16)
    axs[2].set_xlabel("")
    axs[2].tick_params(axis='y', labelsize=14)
    axs[2].set_xticks([])

    handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors]
    labels = result_names
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=18, title="Models", title_fontsize=20, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout(rect=(0.0, 0.1, 1.0, 1.0))
    plt.show()


def hybrid_metrics_by_user_records(results, thresholds):
    """
    Plots MSE, MAE, and accuracy against the number of records in the training set.

    #### Parameters:
    - results: dict
        Dictionary where keys are thresholds and values are another dictionary with 'mse', 'mae', and 'accuracy' as keys.
    - thresholds: list
        List of thresholds for the number of records in the training set.

    #### Returns:
    None. Displays a matplotlib plot.
    """
    sns.set_theme(style="whitegrid")
    mse_values = [results[thresh]['mse'] for thresh in thresholds]
    mae_values = [results[thresh]['mae'] for thresh in thresholds]
    accuracy_values = [results[thresh]['accuracy'] for thresh in thresholds]

    bar_width = 0.3
    thresholds_indices = np.arange(len(thresholds))

    fig, ax1 = plt.subplots(figsize=(10, 6))

    bars1 = ax1.bar(thresholds_indices - bar_width/2, mse_values, bar_width, label='MSE', color='tab:blue', alpha=0.7)
    bars2 = ax1.bar(thresholds_indices + bar_width/2, mae_values, bar_width, label='MAE', color='tab:orange', alpha=0.7)

    ax2 = ax1.twinx()
    ax2.plot(thresholds_indices, accuracy_values, marker='o', label='Accuracy', color='tab:green', linewidth=2)

    ax1.set_xlabel('Number of records in the training set', fontsize=14)
    ax1.set_ylabel('MSE & MAE', color='black', fontsize=14)
    ax2.set_ylabel('Accuracy', color='black', fontsize=14)

    range_labels = [f'{thresholds[i]} - {thresholds[i+1]}' for i in range(len(thresholds)-1)]
    range_labels.append(f'> {thresholds[-1]}')
    ax1.set_xticks(thresholds_indices)
    ax1.set_xticklabels(range_labels, rotation=45)

    plt.title('Performance vs Number of records in the training set', fontsize=16)
    lines_labels = [bars1, bars2, ax2.get_lines()[0]]
    labels = [line.get_label() for line in lines_labels]
    ax1.legend(lines_labels, labels, loc='upper left')

    ax1.grid(True, color='#e8e8e8')
    ax2.grid(False)
    fig.tight_layout()
    plt.show()


def hybrid_metrics_by_cluster_size(hybrid_results: dict):
    """
    Plots MSE and MAE against the number of clusters for hybrid models.

    #### Parameters:
    - hybrid_results: dict
        Dictionary where keys are the number of clusters and values are another dictionary
        with 'mse' and 'mae' as keys.

    #### Returns:
    None. Displays a matplotlib plot.
    """
    sns.set_theme(style="whitegrid")
    metrics_to_plot = ['mse', 'mae']
    fig, axs = plt.subplots(len(metrics_to_plot), 1, figsize=(5, 8))
