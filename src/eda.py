import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src import config

def create_class_distribution_plot():
    """
    Analyzes the dataset to find the number of images per class
    and saves a bar plot visualizing the distribution.
    """
    print("Performing EDA: Analyzing class distribution...")
    
    
    class_counts = {}
    for class_name in os.listdir(config.DATA_DIR):
        class_dir = os.path.join(config.DATA_DIR, class_name)
        if os.path.isdir(class_dir):
            num_images = len(os.listdir(class_dir))
            class_counts[class_name] = num_images
    
    
    counts_df = pd.DataFrame(list(class_counts.items()), columns=['Class', 'Count']).sort_values('Count', ascending=False)
    
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(x='Class', y='Count', data=counts_df)
    ax.set_title('Class Distribution in TrashNet Dataset')
    ax.set_xlabel('Material Class')
    ax.set_ylabel('Number of Images')
    
    # Add counts on top of the bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points')

    # Save the plot
    plt.savefig(config.CLASS_DISTRIBUTION_PATH)
    print(f"Class distribution plot saved to: {config.CLASS_DISTRIBUTION_PATH}")
    

if __name__ == '__main__':
    create_class_distribution_plot()