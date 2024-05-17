#%%
from DNNClassifier import DNNClassifier, ModelTrainer
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch.nn.functional
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
import torch.nn as nn
import matplotlib as mpl

MODEL_PATH = Path("venv/models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
#create model save path
MODEL_NAME= "17MAY.pth"
MODEL_SAVE_PATH = MODEL_PATH/MODEL_NAME
## Loading the model of just the state dict
#new instance of the linear regression model class
model_01 = DNNClassifier()
model_01.cpu()
model_01.load_state_dict(torch.load(MODEL_SAVE_PATH))
print (model_01.state_dict())
#%%
df = pd.read_csv('data_generation/data/transfer_data.csv')
scaler = MinMaxScaler()
scaler.fit(df)

# %%
# Transform the data using the fitted scaler and keep it as a DataFrame
df = pd.DataFrame(scaler.transform(df), columns=df.columns)

df_Features = df.iloc[:, :4]
df_Labels = df.iloc[:, -3:]

data_Features = df_Features.values
data_Labels = df_Labels.values

# Fit and transform the features
X = torch.from_numpy(data_Features).type(torch.float)
y = torch.from_numpy(data_Labels).type(torch.float)
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "mps")

# %% Setting up training params
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# %% Send data
model_01.eval()  # turns off difference setting sin th emodel not needed evaluating/testing
with torch.inference_mode():
    y_pred = model_01(X_test)

 # %%
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

 # %%
mpl.use('macosx')
y_pred_array = df_result_y_pred.iloc[:, -2:].values
y_test_array = df_result_y_test.iloc[:, -2:].values
errors_array =abs(y_test_array-y_pred_array)

#%%  Plot 1 (Prediction vs Actual plot)
#Plotting actual vs predicted values for x
n = 1500
fig, (ax1,ax2) = plt.subplots(1,2,figsize = (12,6))
ax1.scatter(y_test_array[:n, 0], y_pred_array[:n, 0],color = 'blue', alpha = 0.5)
ax1.plot([min(y_test_array[:n, 0]),max(y_test_array[:n, 0])],[min(y_test_array[:n, 0]),max(y_test_array[:n, 0])],color = 'red', linestyle = '--')
ax1.set_title("Actual Vs Predicted Values X Values  ")
ax1.set_xlabel("Actual Values ")
ax1.set_ylabel("Predicted Values ")

ax2.scatter(y_test_array[:n, 1], y_pred_array[:n, 1],color = 'blue', alpha = 0.5)
ax2.plot([min(y_test_array[:n, 1]),max(y_test_array[:n, 1])],[min(y_test_array[:n, 1]),max(y_test_array[:n, 1])],color = 'red', linestyle = '--')
ax2.set_title("Actual Vs Predicted Values Y values ")
ax2.set_xlabel("Actual Values ")
ax2.set_ylabel("Predicted Values ")

plt.tight_layout()
plt.show()

#%% Plot 2 (Residual plots)
# Plotting actual vs predicted values for x
n = 1500
fig, (ax1,ax2) = plt.subplots(1,2,figsize = (12,6))
ax1.scatter(y_pred_array[:n, 0],errors_array[:n, 0],color = 'blue', alpha = 0.5)
ax1.axhline(y=0,color = 'red',linestyle = '--')
ax1.set_title("Residueal Plot x ")
ax1.set_xlabel("Predicted Values")
ax1.set_ylabel("Residules ")



ax2.scatter(y_pred_array[:n, 1],errors_array[:n, 1],color = 'blue', alpha = 0.5)
ax2.axhline(y=0,color = 'red',linestyle = '--')
ax2.set_title("Residueal Plot y ")
ax2.set_xlabel("Predicted Values")
ax2.set_ylabel("Residules ")


plt.tight_layout()
plt.show()

#%% Plot 3 (Bar Chart )
n = 1500
width = 50
fig, (ax1,ax2) = plt.subplots(1,2,figsize = (12,6))

ax1.bar(range(len(errors_array[:n,0])),errors_array[:n,0],width = width,color = 'skyblue')
ax1.set_title("Absolute erros X ")
ax1.set_xlabel("Data point")
ax1.set_ylabel("Absolute errors ")




ax2.bar(range(len(errors_array[:n,1])),errors_array[:n,1],width = width,color = 'skyblue')
ax2.set_title("Absolute erros Y ")
ax2.set_xlabel("Data point")
ax2.set_ylabel("Absolute errors ")



plt.tight_layout()
plt.show()

#%% Plot 4 (QQ plot )

plt.figure(figsize=(10, 6))
stats.probplot(errors_array[:n,0], dist="norm", plot=plt)
plt.title('Q-Q Plot')
plt.grid(True)
plt.show()
#%% Plot 5 Prediction with confidence intervals
sigma=1.96
n = 200
mean_residualsx = np.mean(errors_array[:n,0])
std_residualsx = np.std(errors_array[:n,0])
mean_residualsy = np.mean(errors_array[:n,1])
std_residualsy = np.std(errors_array[:n,1])

lower_boundx = y_pred_array[:n,0] - sigma * std_residualsx
upper_boundx = y_pred_array[:n,0] + sigma * std_residualsx
lower_boundy = y_pred_array[:n,1] - sigma * std_residualsy
upper_boundy = y_pred_array[:n,1] + sigma * std_residualsy

fig, (ax1,ax2) = plt.subplots(1,2,figsize = (12,6))
ax1.scatter(y_test_array[:n,0], y_pred_array[:n,0], alpha=0.5, label='Predictions')
ax1.fill_between(y_test_array[:n,0], lower_boundx, upper_boundx, color='red', alpha=0.2, label=f'{sigma*100:.0f}% Prediction Interval')
ax1.set_xlabel('Actual')
ax1.set_ylabel('Predicted')
ax1.set_title('Prediction Intervals x')
ax1.legend()


ax2.scatter(y_test_array[:n,1], y_pred_array[:n,1], alpha=0.5, label='Predictions')
ax2.fill_between(y_test_array[:n,1], lower_boundy, upper_boundy, color='red', alpha=0.2, label=f'{sigma*100:.0f}% Prediction Interval')
ax2.set_xlabel('Actual')
ax2.set_ylabel('Predicted')
ax2.set_title('Prediction Intervals y')
ax2.legend()

plt.tight_layout()
plt.show()
#%% Plot 6 Fature importance (ABS of the weight)
state_dict = model_01.state_dict()
# Extracting weights for the first layer (input to first hidden layer)
# Extract weights for the first four nodes in the first layer
weights_first_layer_x = abs(state_dict['layer1.weight'][:, 0].numpy())
weights_first_layer_y = abs(state_dict['layer1.weight'][:, 1].numpy())
weights_first_layer_z = abs(state_dict['layer1.weight'][:, 2].numpy())
weights_first_layer_t = abs(state_dict['layer1.weight'][:, 3].numpy())

# Number of subplots (rows, columns)
fig, axs = plt.subplots(2, 2, figsize=(20, 8))

# Plotting each weight series in its own subplot
axs[0,0].bar(range(len(weights_first_layer_x)), weights_first_layer_x)
axs[0,0].set_xlabel('Feature Index')
axs[0,0].set_ylabel('Feature Importance')
axs[0,0].set_title('Feature Importance Plot rel X')

axs[0,1].bar(range(len(weights_first_layer_y)), weights_first_layer_y)
axs[0,1].set_xlabel('Feature Index')
axs[0,1].set_ylabel('Feature Importance')
axs[0,1].set_title('Feature Importance Plot rel Y')

axs[1,0].bar(range(len(weights_first_layer_z)), weights_first_layer_z)
axs[1,0].set_xlabel('Feature Index')
axs[1,0].set_ylabel('Feature Importance')
axs[1,0].set_title('Feature Importance Plot rel Z')

axs[1,1].bar(range(len(weights_first_layer_t)), weights_first_layer_t)
axs[1,1].set_xlabel('Feature Index')
axs[1,1].set_ylabel('Feature Importance')
axs[1,1].set_title('Feature Importance Plot tof')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()