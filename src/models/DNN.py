import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import logging
import wandb

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    y_true_mean = np.mean(y_true)
    ss_total = np.sum((y_true - y_true_mean) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_res / ss_total)


class DNNRegressor(nn.Module):
    def __init__(self, input_size, output_size, num_neurons1, num_neurons2, num_neurons3):
        super(DNNRegressor, self).__init__()

        # Define layers according to the new DNN setup
        self.fc1 = nn.Linear(input_size, num_neurons1)
        self.fc2 = nn.Linear(num_neurons1, num_neurons2)
        self.fc3 = nn.Linear(num_neurons2, num_neurons3)
        self.fc4 = nn.Linear(num_neurons3, output_size)  # Output size is 2 for the two target columns

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ModelTrainer:
    def __init__(self, model, loss_fn, optimizer, data_set):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.data_set = data_set
        self.train_losses = []
        self.test_losses = []
        self.mae_array = []
        self.mse_array = []
        self.r2_array = []
        self.epochs_array = []

    def train(self, n_epochs, x_train, x_test, y_train, y_test, track):
        if track:
            wandb.init(
                # Set the project where this run will be logged
                project="p4p",
                config={
                    "learning_rate": self.optimizer.param_groups[0]['lr'],
                    "dataset": self.data_set,
                    "epochs": n_epochs,
                    "goal": "learning lambert arc velocities"
                },
            )

        for epoch in tqdm(range(n_epochs), desc="Training Model"):
            self.model.train()
            permutation = torch.randperm(x_train.size()[0])

            for i in range(0, x_train.size()[0], 32):  # Batch size = 32
                self.optimizer.zero_grad()

                indices = permutation[i:i + 32]
                batch_x, batch_y = x_train[indices], y_train[indices]

                # Forward pass
                y_pred = self.model(batch_x)
                loss = self.loss_fn(y_pred, batch_y)
                loss.backward()
                self.optimizer.step()

            self.train_losses.append(loss.item())

            # Validation
            with torch.no_grad():
                self.model.eval()
                test_pred = self.model(x_test)
                test_loss = self.loss_fn(test_pred, y_test)
                self.test_losses.append(test_loss.item())

                # Convert predictions and true values to numpy arrays
                test_pred_np = test_pred.cpu().numpy()
                y_test_np = y_test.cpu().numpy()

                # Calculate additional metrics
                mae = mean_absolute_error(y_test_np, test_pred_np)
                mse = mean_squared_error(y_test_np, test_pred_np)
                r2 = r2_score(y_test_np, test_pred_np)

                if track:
                    # Log metrics to wandb
                    wandb.log({
                        "train_loss": loss.item(),
                        "test_loss": test_loss.item(),
                        "mean_absolute_error": mae,
                        "mean_squared_error": mse,
                        "r2_score": r2,
                    })

            self.mae_array.append(mae)
            self.mse_array.append(mse)
            self.r2_array.append(r2)
            self.epochs_array.append(epoch)

        logging.info("Training complete.")
        if track:
            wandb.finish()

    def plot_training_curves(self):
        plt.ion()
        plt.figure()
        plt.xscale('log')
        plt.yscale('log')
        plt.plot(self.epochs_array, self.train_losses, label='Training Loss')
        plt.plot(self.epochs_array, self.test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        logging.info(
            "Training loss is : " + str(self.train_losses[-1]) + ",    Test loss is : " + str(self.test_losses[-1]))
