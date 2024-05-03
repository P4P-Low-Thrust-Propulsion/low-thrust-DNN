import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import time
from IPython import display

df = pd.read_csv('data_generation/data/transfer_data.csv')
scaler = MinMaxScaler()
# Fit the scaler to your data (optional, depending on the scaler)
scaler.fit(df)

# Transform the data using the fitted scaler and keep it as a DataFrame
df = pd.DataFrame(scaler.transform(df), columns=df.columns)

df_Features = df.iloc[:, :4]
df_Labels = df.iloc[:, -3:]

data_Features = df_Features.values
data_Labels = df_Labels.values

# Fit and transform the features
X = torch.from_numpy(data_Features).type(torch.float)
y = torch.from_numpy(data_Labels).type(torch.float)

######### CHEcking state of GPU
print(f"PyTorch version: {torch.__version__}")

# Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
print(f"Is MPS available? {torch.backends.mps.is_available()}")

# Set the device
device = "mps" if torch.backends.mps.is_available() else "cpu"
device = "cpu"
print(f"Using device: {device}")

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
len(X_train), len(X_test), len(y_train), len(y_test)
# y_train = torch.unsqueeze(y_train, dim=1)
# y_test = torch.unsqueeze(y_test, dim=1)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
# %%
# 1. Construct a model class that subclasses nn.Module
from torch import nn
import torch.nn.functional as F


class Model_0v1(nn.Module):
    n = 150

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=4, out_features=n)
        self.layer2 = nn.Linear(in_features=n, out_features=n)
        self.layer3 = nn.Linear(in_features=n, out_features=n)
        self.layer4 = nn.Linear(in_features=n, out_features=3)
        self.relu = nn.Softshrink()

    def forward(self, x):
        # return self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))
        return self.layer4(self.relu(self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))))


torch.manual_seed(42)
# create an instance of the mdoel
model_01 = Model_0v1()
model_01.state_dict()

# %%
print(f"Model on device:\n{next(model_01.parameters()).device}")
X_test.to(device)
print(f"DATA on device:\n{X_test.device}")
# %%
# train the model,

# the whole idea is to move the red dots to the green, unknown pararemeters to known parameters
# loss function
# optomizer


# setting up a loss function
loss_fn = nn.MSELoss()

# setting up a optomizer
optomizer = torch.optim.SGD(params=model_01.parameters(),
                            lr=0.05)  # lr = learning rate

# %%
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)

# Assuming X_train, y_train, X_test, y_test are already on CPU
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

# Assuming model_01, loss_fn, optimizer are already defined
# Move model and optimizer to GPU
model_01 = model_01.to(device)
# optimizer = optomizer.to(device)

# Number of epochs
epochs = 10000

# Lists to store losses
losses = []
test_losses = []
epochs_array = []

start_time = time.time()

# Training loop
for epoch in range(epochs):
    epochs_array.append(epoch)
    model_01.train()

    # Forward pass
    y_preds = model_01(X_train)

    # Calculate the loss
    loss = loss_fn(y_preds, y_train)
    losses.append(loss.item())

    # Zero gradients
    optomizer.zero_grad()

    # Backward pass
    loss.backward()

    # Step the optimizer (perform gradient descent)
    optomizer.step()

    # Testing
    model_01.eval()

    with torch.no_grad():  # Ensure no gradients are computed
        # Forward pass for testing
        test_preds = model_01(X_test)

        # Calculate test loss
        test_loss = loss_fn(test_preds, y_test)
        test_losses.append(test_loss.item())

    if epoch % 5 == 0:
        print(
            f"Status: {epoch * 100 / epochs}% | Epoch: {epoch} | Training Loss: {loss:.8f} | Test Loss: {test_loss:.8f}")

print(f"Training took : {time.time() - start_time}")

# Plotting
plt.figure()
plt.plot(epochs_array, losses, label='Training Loss')
plt.plot(epochs_array, test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()

# %%
##making estimations
model_01.eval()  # tunrs off diffrence setting sin th emodel not needed evaluating/testing
with torch.inference_mode():
    y_preds = model_01(X_test)


# %%
##unscale the y_test and y_preds
def unscale(scaled_value, i):
    unscaled_value = scaled_value * (scaler.data_max_[i] - scaler.data_min_[i]) + (scaler.data_min_[i])
    return unscaled_value


df_result_ytest_scaled = pd.concat([pd.DataFrame(X_test.numpy()), pd.DataFrame(y_test.numpy())], ignore_index=True,
                                   axis='columns')
df_result_ypreds_scaled = pd.concat([pd.DataFrame(X_test.numpy()), pd.DataFrame(y_preds.numpy())], ignore_index=True,
                                    axis='columns')

# Apply to unscale function to each column of inputs array
df_result_ytest = pd.DataFrame()
df_result_ypreds = pd.DataFrame()

for column in df_result_ytest_scaled.columns:
    df_result_ytest[column] = unscale(df_result_ytest_scaled[column], column)
    df_result_ypreds[column] = unscale(df_result_ypreds_scaled[column], column)
df_result_ytest, df_result_ypreds
# %%
n = 10
plt.scatter(epochs_array[:n], np.linalg.norm(df_result_ytest.iloc[:, -3:], axis=1)[:n], c='green', label='y_test')
plt.scatter(epochs_array[:n], np.linalg.norm(df_result_ypreds.iloc[:, -3:], axis=1)[:n], c='blue', label='y_preds')

# Set labels and title
plt.xlabel('Epochs')
plt.ylabel('Magnitude of velocity [km/s]')
plt.title('Scatter Plot of Magnitudes Scaled back')
plt.legend()

# %%
n = 10
plt.scatter(epochs_array[:n], np.linalg.norm(df_result_ytest_scaled, axis=1)[:n], c='green', label='y_test')
plt.scatter(epochs_array[:n], np.linalg.norm(df_result_ypreds_scaled, axis=1)[:n], c='blue', label='y_preds')

# Set labels and title
plt.xlabel('Epochs')
plt.ylabel('Magnitude of velocity [km/s]')
plt.title('Scatter Plot of Magnitudes unscaled')
plt.legend()

save
# %%
# saving a model in pytorch

from pathlib import Path

# create models directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
# create model save path
MODEL_NAME = "LeakyRelu.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
print(f"Saving model to : {MODEL_SAVE_PATH}")

# 3. SAVE THE MODEL SAVE DICT
torch.save(obj=model_01.state_dict(), f=MODEL_SAVE_PATH)

## Loading the model of just the state dict
# new instance of the linear regression model class
# loaded_model_0 = LinearRegressionModel()
# load the saved state_dict of model_0 into the new instance
# loaded_model_0.load_state_dict(torch.load(MODEL_SAVE_PATH))

# print(loaded_model_0.state_dict())