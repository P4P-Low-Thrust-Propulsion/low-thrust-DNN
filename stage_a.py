import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
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


# X = torch.from_numpy(X).type(torch.float)
# y = torch.from_numpy(y).type(torch.float)
# tensor_Features = torch.tensor(data_Features, dtype=torch.float32)
# tensor_Labels   = torch.tensor(data_Labels  , dtype=torch.float32)
X = torch.from_numpy(data_Features).type(torch.float)
y = torch.from_numpy(data_Labels).type(torch.float)
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
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=4, out_features=32)
        self.layer2 = nn.Linear(in_features=32, out_features=32)
        self.layer3 = nn.Linear(in_features=32, out_features=3)
        self.relu = nn.ReLU()

    def forward(self, x):
        # return self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))
        return self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))


torch.manual_seed(42)
# create an instance of the mdoel
model_01 = Model_0v1()
model_01.state_dict()
# %%
model_01

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

torch.manual_seed(42)
epochs = 5000

plt.figure()

losses = []
test_losses = []
epochs_array = []

for epoch in range(epochs):
    epochs_array.append(epoch)
    model_01.train()
    # 1 forward pass
    y_preds = model_01(X_train)
    # 2 calculate the loss
    loss = loss_fn(y_preds, y_train)
    losses.append(loss.item())
    # 3 optomizer zero grad
    optomizer.zero_grad()
    # 4 perfrom back propogation on the loss with respect to the parameters of the model
    loss.backward()
    # 5. step the optimizer (perform gradeint descent )
    optomizer.step()  # by deful how the optimzer changes will acculumate through the loop so we have to zero them above in step 3
    # TESTING
    model_01.eval()  # tunrs off diffrence setting sin th emodel not needed evaluating/testing

    with torch.inference_mode():
        # 1. forward
        test_pred = model_01(X_test)
        # 2. lest
        test_loss = loss_fn(test_pred, y_test)
        test_losses.append(test_loss.item())

    if epoch % 5 == 0:
        # print(f"Loss: {y_preds}")
        print(f" Status : {epoch * 100 / epochs}%  Epoch: {epoch} | Test: {loss:.8f} | Test loss: {test_loss:.8f}")

plt.plot(epochs_array, losses, label='Training Loss')
plt.plot(epochs_array, test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()

# %%
# making estimations
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
n = 40
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