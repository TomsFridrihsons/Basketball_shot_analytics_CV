import pandas as pd
import matplotlib.pyplot as plt
import sys
import os


def plot_training_results(csv_path):
    """
    Read YOLO training CSV and plot losses, metrics, and learning rate.
    """
    # Check if file exists
    if not os.path.isfile(csv_path):
        print(f"Error: File '{csv_path}' not found.")
        sys.exit(1)

    # Load data
    df = pd.read_csv(csv_path)

    # Ensure epoch column is integer and set as index
    df['epoch'] = df['epoch'].astype(int)
    df.set_index('epoch', inplace=True)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('YOLO Training Results', fontsize=16)

    # 1. Training and validation losses
    ax = axes[0, 0]
    ax.plot(df.index, df['train/box_loss'], label='train/box_loss', color='blue')
    ax.plot(df.index, df['train/cls_loss'], label='train/cls_loss', color='orange')
    ax.plot(df.index, df['train/dfl_loss'], label='train/dfl_loss', color='green')
    ax.plot(df.index, df['val/box_loss'], label='val/box_loss', linestyle='--', color='blue')
    ax.plot(df.index, df['val/cls_loss'], label='val/cls_loss', linestyle='--', color='orange')
    ax.plot(df.index, df['val/dfl_loss'], label='val/dfl_loss', linestyle='--', color='green')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Losses')
    ax.legend(fontsize=8)
    ax.grid(True)

    # 2. Precision, Recall, mAP50, mAP50-95
    ax = axes[0, 1]
    ax.plot(df.index, df['metrics/precision(B)'], label='Precision', color='purple')
    ax.plot(df.index, df['metrics/recall(B)'], label='Recall', color='brown')
    ax.plot(df.index, df['metrics/mAP50(B)'], label='mAP50', color='red')
    ax.plot(df.index, df['metrics/mAP50-95(B)'], label='mAP50-95', color='pink')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Precision, Recall, and mAP')
    ax.legend()
    ax.grid(True)

    # 3. Learning rate (all three columns are identical, plot only one)
    ax = axes[1, 0]
    ax.plot(df.index, df['lr/pg0'], label='Learning Rate', color='black')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('LR')
    ax.set_title('Learning Rate')
    ax.legend()
    ax.grid(True)

    # 4. Validation mAP50-95 (key metric)
    ax = axes[1, 1]
    ax.plot(df.index, df['metrics/mAP50-95(B)'], label='mAP50-95', color='darkgreen')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP50-95')
    ax.set_title('Validation mAP50-95')
    ax.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Use the filename from command line argument or default to 'results (2).csv'
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = r"C:\Users\fridr\Documents\HooperAI\data\raw\csv\results (2).csv"  # change this to your actual file name
        print(filename)
    plot_training_results(filename)