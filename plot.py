import os
import pandas as pd
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.abspath(__file__))

for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    
    if os.path.isdir(folder_path):
        loss_file = os.path.join(folder_path, "loss_log.txt")
        val_loss_file = os.path.join(folder_path, "val_loss_log.txt")
        
        if os.path.exists(loss_file) and os.path.exists(val_loss_file):
            try:
                loss_df = pd.read_csv(loss_file)
                val_loss_df = pd.read_csv(val_loss_file)

                plt.figure()
                plt.plot(loss_df["step"], loss_df["loss"], label="Loss")
                plt.plot(val_loss_df["step"], val_loss_df["val_loss"], label="Val Loss")
                plt.xlabel("Step")
                plt.ylabel("Loss")
                plt.title(f"Loss Curve - {folder}")
                plt.legend()
                plt.grid(True)

                plot_path = os.path.join(folder_path, "loss_plot.png")
                plt.savefig(plot_path)
                plt.close()
                print(f"Saved plot to: {plot_path}")
            except Exception as e:
                print(f"Failed to process {folder}: {e}")
