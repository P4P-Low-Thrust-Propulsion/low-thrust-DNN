# %%
from DNNClassifier import DNNClassifier, ModelTrainer
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch.nn.functional
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
import torch.nn as nn

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

# Check if CUDA (NVIDIA GPU) is available
cuda_available = torch.cuda.is_available()

print("CUDA (NVIDIA GPU) available:", cuda_available)

# Move your model and data tensors to the GPU (if available)
device = torch.device("cuda" if cuda_available else "mps")

# %% Setting up training params
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
len(X_train), len(X_test), len(y_train), len(y_test)

# %% Construct a model class that subclasses nn.Module
torch.manual_seed(42)

# create an instance of the model
model_01 = DNNClassifier()
model_01.state_dict()

# Your model and data setup
model_01.to(device)
X_train, X_test, y_train, y_test = X_train.to(device), X_test.to(device), y_train.to(device), y_test.to(device)

# %% Train model
# Setting up a loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(params=model_01.parameters(), lr=0.05)  # lr = learning rate

model_01_trainer = ModelTrainer(model_01, loss_fn, optimizer)
epochs_array = model_01_trainer.train(2, X_train, X_test, y_train, y_test)

# %% Make estimates
model_01.eval()  # turns off difference setting sin th emodel not needed evaluating/testing
with torch.inference_mode():
    y_pred = model_01(X_test)


# %%
# unscale the y_test and y_preds
def unscale(scaled_value, i):
    unscaled_value = scaled_value * (scaler.data_max_[i] - scaler.data_min_[i]) + (scaler.data_min_[i])
    return unscaled_value


df_result_y_test_scaled = pd.concat([pd.DataFrame(X_test.cpu().numpy()), pd.DataFrame(y_test.cpu().numpy())], ignore_index=True,
                                    axis='columns')
df_result_y_pred_scaled = pd.concat([pd.DataFrame(X_test.cpu().numpy()), pd.DataFrame(y_pred.cpu().numpy())], ignore_index=True,
                                    axis='columns')

# Apply to unscale function to each column of inputs arrays
df_result_y_test = pd.DataFrame()
df_result_y_pred = pd.DataFrame()

for column in df_result_y_test_scaled.columns:
    df_result_y_test[column] = unscale(df_result_y_test_scaled[column], column)
    df_result_y_pred[column] = unscale(df_result_y_pred_scaled[column], column)

# %% Plotting
n = 10
plt.figure()
plt.scatter(epochs_array[:n], np.linalg.norm(df_result_y_test.iloc[:, -3:], axis=1)[:n], c='green', label='y_test')
plt.scatter(epochs_array[:n], np.linalg.norm(df_result_y_pred.iloc[:, -3:], axis=1)[:n], c='blue', label='y_pred')

# Set labels and title
plt.xlabel('Epochs')
plt.ylabel('Magnitude of velocity [km/s]')
plt.title('Scatter Plot of Magnitudes Scaled back')
plt.legend()
plt.show()

plt.figure()
plt.scatter(epochs_array[:n], np.linalg.norm(df_result_y_test_scaled, axis=1)[:n], c='green', label='y_test')
plt.scatter(epochs_array[:n], np.linalg.norm(df_result_y_pred_scaled, axis=1)[:n], c='blue', label='y_preds')

# Set labels and title
plt.xlabel('Epochs')
plt.ylabel('Magnitude of velocity [km/s]')
plt.title('Scatter Plot of Magnitudes unscaled')
plt.legend()
plt.show()

# %% Saving model

# create models directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
# create model save path
MODEL_NAME = "LeakyRelu.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
print(f"Saving model to : {MODEL_SAVE_PATH}")

# 3. SAVE THE MODEL SAVE DICT
torch.save(obj=model_01.state_dict(), f=MODEL_SAVE_PATH)

