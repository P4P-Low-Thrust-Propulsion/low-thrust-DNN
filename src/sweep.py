import wandb
import pprint
from src.models.DNNClassifier import DNNClassifier, ModelTrainer
import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, PowerTransformer, QuantileTransformer
)
import torch.nn.functional
from sklearn.model_selection import train_test_split
from pathlib import Path
import torch.nn as nn
from datetime import date
import logging


def main():
    wandb.init(project="my-first-sweep")

    # Parameters
    DATA_SET = wandb.config.DATA_SET
    RECORD = False
    LEARNING_RATE = wandb.config.LEARNING_RATE
    EPOCHS = wandb.config.EPOCHS
    TEST_SIZE = wandb.config.TEST_SIZE
    INPUT_SIZE = 3
    OUTPUT_SIZE = 4
    NUM_LAYERS = wandb.config.NUM_LAYERS
    NUM_NEURONS = wandb.config.NUM_NEURONS

    activation_function = wandb.config.ACTIVATION_FUNCTION
    if activation_function == 'ELU':
        ACTIVATION = nn.ELU
    elif activation_function == 'Hardshrink':
        ACTIVATION = nn.Hardshrink
    elif activation_function == 'Hardsigmoid':
        ACTIVATION = nn.Hardsigmoid
    elif activation_function == 'Hardtanh':
        ACTIVATION = nn.Hardtanh
    elif activation_function == 'Hardswish':
        ACTIVATION = nn.Hardswish
    elif activation_function == 'LeakyReLU':
        ACTIVATION = nn.LeakyReLU
    elif activation_function == 'LogSigmoid':
        ACTIVATION = nn.LogSigmoid
    elif activation_function == 'PReLU':
        ACTIVATION = nn.PReLU
    elif activation_function == 'ReLU':
        ACTIVATION = nn.ReLU
    elif activation_function == 'ReLU6':
        ACTIVATION = nn.ReLU6
    elif activation_function == 'RReLU':
        ACTIVATION = nn.RReLU
    elif activation_function == 'SELU':
        ACTIVATION = nn.SELU
    elif activation_function == 'CELU':
        ACTIVATION = nn.CELU
    elif activation_function == 'GELU':
        ACTIVATION = nn.GELU
    elif activation_function == 'Sigmoid':
        ACTIVATION = nn.Sigmoid
    elif activation_function == 'SiLU':
        ACTIVATION = nn.SiLU
    elif activation_function == 'Mish':
        ACTIVATION = nn.Mish
    elif activation_function == 'Softplus':
        ACTIVATION = nn.Softplus
    elif activation_function == 'Softshrink':
        ACTIVATION = nn.Softshrink
    elif activation_function == 'Softsign':
        ACTIVATION = nn.Softsign
    elif activation_function == 'Tanh':
        ACTIVATION = nn.Tanh
    elif activation_function == 'Tanhshrink':
        ACTIVATION = nn.Tanhshrink
    else:
        raise ValueError(f"Unknown activation function '{activation_function}'")

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Display plots in separate window
    # mpl.use('macosx')

    DATA_PATH = Path("data/lambert/processed")
    DATA_NAME = "transfer_data_" + DATA_SET + ".csv"

    # create saved_models directory
    MODEL_PATH = Path("src/models/saved_models")
    today = date.today()
    MODEL_NAME = str(today) + "_" + DATA_SET + ".pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    df = pd.read_csv(DATA_PATH / DATA_NAME)

    method = wandb.config.SCALING_TECHNIQUE
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    elif method == 'maxabs':
        scaler = MaxAbsScaler()
    elif method == 'yeo-johnson':
        scaler = PowerTransformer(method='yeo-johnson')
    elif method == 'quantile-uniform':
        scaler = QuantileTransformer(output_distribution='uniform')
    elif method == 'quantile-normal':
        scaler = QuantileTransformer(output_distribution='normal')
    else:
        raise ValueError("Unknown scaling method")

    # Fit and transform the scaler to each column separately
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    df_Features = df.iloc[:, :INPUT_SIZE]
    df_Labels = df.iloc[:, -OUTPUT_SIZE:]

    data_Features = df_Features.values
    data_Labels = df_Labels.values

    # Fit and transform the features
    x = torch.tensor(data_Features, dtype=torch.float32)
    y = torch.tensor(data_Labels, dtype=torch.float32)

    # Check if CUDA (NVIDIA GPU) is available
    cuda_available = torch.cuda.is_available()

    logging.info("CUDA (NVIDIA GPU) available: " + str(cuda_available))

    # Move your model and processed tensors to the GPU (if available)
    device = torch.device("cuda" if cuda_available else "mps")

    # %% Setting up training params
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=42)

    # %% Construct a model class that subclasses nn.Module
    torch.manual_seed(42)

    # create an instance of the model
    model_01 = DNNClassifier(INPUT_SIZE, OUTPUT_SIZE, NUM_LAYERS, NUM_NEURONS, ACTIVATION)
    model_01.state_dict()

    # Your model and processed setup
    model_01.to(device)
    x_train, x_test, y_train, y_test = x_train.to(device), x_test.to(device), y_train.to(device), y_test.to(device)

    # %% Train model
    # Setting up a loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(params=model_01.parameters(), lr=LEARNING_RATE, momentum=0.75)  # lr = learning rate

    model_01_trainer = ModelTrainer(model_01, loss_fn, optimizer, DATA_NAME)
    model_01_trainer.train(EPOCHS, x_train, x_test, y_train, y_test, RECORD)
    wandb.log({"Training_Loss": model_01_trainer.train_losses[-1]})


# %% SWEEP config and setup
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "Training_Loss"},
    "parameters": {
        "NUM_NEURONS": {"values": [16, 32, 64, 128, 1256]},
        "NUM_LAYERS": {"values": [3, 6, 9, 12, 15]},
        "LEARNING_RATE": {"values": [0.001, 0.01, 0.1, 0.5]},
        "EPOCHS": {"values": [100, 300, 500, 900, 1500, 3000]},
        "TEST_SIZE": {"values": [0.05, 0.1, 0.2, 0.3]},
        "DATA_SET": {"values": ['10K_01', '10K_02', '10K_05', '10K_10']},
        "SCALING_TECHNIQUE": {"values": ['minmax', 'standard', 'robust', 'maxabs', 'yeo-johnson', 'quantile-uniform',
                                         'quantile-normal']},
        "ACTIVATION_FUNCTION": {"values": ['ELU', 'Hardshrink', 'Hardsigmoid', 'Hardtanh', 'Hardswish', 'LeakyReLU',
                                           'LogSigmoid', 'PReLU', 'ReLU', 'ReLU6', 'RReLU', 'SELU', 'CELU', 'GELU',
                                           'Sigmoid', 'SiLU', 'Mish', 'Softplus', 'Softshrink', 'Softsign', 'Tanh',
                                           'Tanhshrink']},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="my second sweep")
wandb.agent(sweep_id, main, count=150)
