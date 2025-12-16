import matplotlib.pyplot as plt
import wandb


class MetricsTracker:
    def __init__(self, log_wandb=True):
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
        self.log_wandb = log_wandb

    def update(self, train_loss, train_acc, val_loss, val_acc, lr, epoch):
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['lr'].append(lr)

        if self.log_wandb:
            wandb.log({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss,
                'val_acc': val_acc, 'lr': lr})

    def plot(self, model_name):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(self.history['train_loss'], label='Train')
        axes[0].plot(self.history['val_loss'], label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(self.history['train_acc'], label='Train')
        axes[1].plot(self.history['val_acc'], label='Val')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(self.history['lr'])
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'logs/training_curves_{model_name}.png', dpi=150)
        plt.show()
