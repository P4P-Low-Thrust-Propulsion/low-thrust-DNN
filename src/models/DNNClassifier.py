import torch
import torch.nn as nn
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


class DNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.n = 300
        self.layer1 = nn.Linear(in_features=3, out_features=self.n)
        self.layer2 = nn.Linear(in_features=self.n, out_features=self.n)
        self.layer3 = nn.Linear(in_features=self.n, out_features=self.n)
        self.layer4 = nn.Linear(in_features=self.n, out_features=self.n)
        self.layer5 = nn.Linear(in_features=self.n, out_features=self.n)
        self.layer6 = nn.Linear(in_features=self.n, out_features=self.n)
        self.layer7 = nn.Linear(in_features=self.n, out_features=self.n)
        self.layer8 = nn.Linear(in_features=self.n, out_features=self.n)
        self.layer9 = nn.Linear(in_features=self.n, out_features=4)
        self.activation = nn.SELU()

    def forward(self, x):
        return self.layer9(self.activation(self.layer8(self.activation(self.layer7(self.activation(self.layer6(self.activation(self.layer5(
            self.activation(self.layer4(self.activation(self.layer3(self.activation(self.layer2(self.activation(self.layer1(x)))))))))))))))))


class ModelTrainer:
    def __init__(self, model, loss_fn, optimizer, data_set):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.data_set = data_set
        self.train_losses = []
        self.test_losses = []
        self.epochs_array = []

    def train(self, n_epochs, x_train, x_test, y_train, y_test, track):
        if track:
            wandb.init(
                # Set the project where this run will be logged
                project="p4p",
                # Track hyperparameters and run metadata
                config={
                    "learning_rate": self.optimizer.param_groups[0]['lr'],
                    "dataset": self.data_set,
                    "epochs": n_epochs,
                    "goal": "learning lambert arc velocities"
                },
            )

        for epoch in tqdm(range(n_epochs), desc="Training Model"):
            self.model.train()
            # Forward pass
            y_pred = self.model(x_train)
            # Calculate the loss
            loss = self.loss_fn(y_pred, y_train)
            self.train_losses.append(loss.item())
            # Zero gradients
            self.optimizer.zero_grad()
            # Backward pass
            loss.backward()
            # Step the optimizer (perform gradient descent)
            self.optimizer.step()

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

            self.epochs_array.append(epoch)

        logging.info("Training complete.")
        if track: wandb.finish()

        return self.epochs_array

    def plot_training_curves(self):
        plt.figure()
        plt.plot(self.epochs_array, self.train_losses, label='Training Loss')
        plt.plot(self.epochs_array, self.test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        logging.info(
            "Training loss is : " + str(self.train_losses[-1]) + ",    Test loss is : " + str(self.test_losses[-1]))
