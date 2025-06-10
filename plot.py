import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_loss_logs():
    files = os.listdir('.')
    loss_log_files = [f for f in files if f.startswith('loss_log')]

    if not loss_log_files:
        print("No files starting with 'loss_log' and ending with '.csv' found in the current directory.")
        return

    for file_name in loss_log_files:
        try:
            df = pd.read_csv(file_name)

            if 'step' not in df.columns or 'train_loss' not in df.columns or 'val_loss' not in df.columns:
                print(f"Skipping {file_name}: Missing one or more required columns (step, train_loss, val_loss).")
                continue

            plt.figure(figsize=(10, 6))
            plt.plot(df['step'], df['train_loss'], label='Train Loss', marker='o', markersize=4)
            plt.plot(df['step'], df['val_loss'], label='Validation Loss', marker='x', markersize=4)

            plt.title(f'Training and Validation Loss for {file_name}')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True) 
            output_file_name = f"{os.path.splitext(file_name)[0]}_loss_plot.png"

            plt.savefig(output_file_name)
            plt.close() 

            print(f"Successfully plotted and saved: {output_file_name}")

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

plot_loss_logs()
