import matplotlib.pyplot as plt
import os

def plot_training_history(gnn_hist, base_hist, save_dir="plots"):
    """Generate visual training curves for Loss and WRMSSE."""
    os.makedirs(save_dir, exist_ok=True)
    
    epochs_gnn = range(len(gnn_hist['train_loss']))
    epochs_base = range(len(base_hist['train_loss']))

    # Plot 1: WRMSSE
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_gnn, gnn_hist['val_wrmsse'], label='DTNetGNN Val WRMSSE', color='blue', linewidth=2)
    plt.plot(epochs_base, base_hist['val_wrmsse'], label='Baseline Val WRMSSE', color='red', linestyle='dashed')
    plt.axvline(x=gnn_hist['best_epoch'], color='blue', alpha=0.3, linestyle=':', label='GNN Best Epoch')
    plt.title('Validation WRMSSE over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('WRMSSE Score (Lower is Better)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{save_dir}/val_wrmsse_curve.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Train vs Val Loss (GNN)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_gnn, gnn_hist['train_loss'], label='Train Loss (RMSSE Proxy)', color='green')
    plt.plot(epochs_gnn, gnn_hist['val_loss'], label='Val Loss (MSE)', color='orange')
    plt.title('DTNetGNN Train vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{save_dir}/gnn_loss_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[plot] 📊 Visual training curves saved to '{save_dir}/val_wrmsse_curve.png' and 'gnn_loss_curve.png'")
