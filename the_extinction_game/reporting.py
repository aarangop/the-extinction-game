
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def process_survival_data(df):
    # Calculate survival metrics per century
    survival_stats = pd.DataFrame({
        'century': range(1, len(df) + 1),
        'avg_survival': df.mean(axis=1),
        'q25': df.quantile(0.25, axis=1),
        'q75': df.quantile(0.75, axis=1)
    })

    return survival_stats


def plot_survival_analysis(df):
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Average Survival Rate
    survival_stats = process_survival_data(df)
    axes[0, 0].fill_between(survival_stats.century,
                            survival_stats.q25,
                            survival_stats.q75,
                            alpha=0.3)
    axes[0, 0].plot(survival_stats.century,
                    survival_stats.avg_survival,
                    'b-')
    axes[0, 0].set_title('Survival Rate Over Time')
    axes[0, 0].set_xlabel('Century')
    axes[0, 0].set_ylabel('Survival Rate')

    # 2. Heatmap of Survival
    sns.heatmap(df.iloc[:, :100],  # Sample of branches
                ax=axes[0, 1],
                cmap='viridis')
    axes[0, 1].set_title('Survival Heatmap (Sample)')
    axes[0, 1].set_xlabel('Branch')
    axes[0, 1].set_ylabel('Century')

    # 3. Distribution of Survival Times
    final_survival = df.iloc[-1]
    axes[1, 0].hist(final_survival, bins=30)
    axes[1, 0].set_title('Distribution of Final Survival')
    axes[1, 0].set_xlabel('Survival State')
    axes[1, 0].set_ylabel('Count')

    # 4. Box Plot of Survival by Century
    survival_long = df.melt(var_name='Branch',
                            value_name='Survival',
                            ignore_index=False)
    survival_long['Century'] = survival_long.index + 1
    sns.boxplot(data=survival_long,
                x='Century',
                y='Survival',
                ax=axes[1, 1])
    axes[1, 1].set_title('Survival Distribution by Century')

    plt.tight_layout()
    return fig
