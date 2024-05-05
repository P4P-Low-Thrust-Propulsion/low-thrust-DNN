import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.n = 300
        self.layer1 = nn.Linear(in_features=4, out_features=self.n)
        self.layer2 = nn.Linear(in_features=self.n, out_features=self.n)
        self.layer3 = nn.Linear(in_features=self.n, out_features=self.n)
        self.layer4 = nn.Linear(in_features=self.n, out_features=3)
        self.relu = nn.Softshrink()

    def forward(self, x):
        return self.layer4(self.relu(self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))))


class ModelTrainer:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.test_losses = []
        self.epochs_array = []

    def train(self, n_epochs, X_train, X_test, y_train, y_test):
        for epoch in tqdm(range(n_epochs), desc="Training Model"):
            self.model.train()
            # Forward pass
            y_pred = self.model(X_train)
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
                test_pred = self.model(X_test)
                test_loss = self.loss_fn(test_pred, y_test)
                self.test_losses.append(test_loss.item())

            self.epochs_array.append(epoch)

        logging.info("Training complete.")

    def plot_training_curves(self):
        plt.figure()
        plt.plot(self.epochs_array, self.train_losses, label='Training Loss')
        plt.plot(self.epochs_array, self.test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.show()
