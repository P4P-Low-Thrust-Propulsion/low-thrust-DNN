import wandb
import pprint
from src.models.DNNClassifier import DNNClassifier, ModelTrainer
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch.nn.functional
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
import torch.nn as nn
from datetime import date
import matplotlib as mpl
import numpy as np
from scipy import stats
import seaborn as sns
import logging

def main():
    wandb.init(project="my-first-sweep")

    # Parameters
    DATA_SET = "10K_01"
    RECORD = False
    LEARNING_RATE = wandb.config.LEARNING_RATE
    EPOCHS = wandb.config.EPOCHS
    TEST_SIZE = wandb.config.TEST_SIZE

    INPUT_SIZE = 3
    OUTPUT_SIZE = 4
    NUM_LAYERS = wandb.config.NUM_LAYERS
    NUM_NEURONS = wandb.config.NUM_NEURONS
    ACTIVATION = nn.SELU

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Display plots in separate window
    # mpl.use('macosx')

    DATA_PATH = Path("data/processed")
    DATA_NAME = "transfer_data_" + DATA_SET + ".csv"

    # create saved_models directory
    MODEL_PATH = Path("src/models/saved_models")
    today = date.today()
    MODEL_NAME = str(today) + "_" + DATA_SET + ".pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    df = pd.read_csv(DATA_PATH / DATA_NAME)

    # Initialize the scaler
    scaler = StandardScaler()  # or MaxAbsScaler()

    # Fit and transform the scaler to each column separately
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # pd.set_option('display.max_columns', None)
    # logging.info("\n" + str(df.describe(include='all')))

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
    model_01 = DNNClassifier(INPUT_SIZE, OUTPUT_SIZE, NUM_LAYERS, NUM_NEURONS,ACTIVATION)
    model_01.state_dict()

    # Your model and processed setup
    model_01.to(device)
    x_train, x_test, y_train, y_test = x_train.to(device), x_test.to(device), y_train.to(device), y_test.to(device)

    # %% Train model
    # Setting up a loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(params=model_01.parameters(), lr=LEARNING_RATE, momentum=0.75)  # lr = learning rate

    model_01_trainer = ModelTrainer(model_01, loss_fn, optimizer, DATA_NAME)
    epochs_array = model_01_trainer.train(EPOCHS, x_train, x_test, y_train, y_test, RECORD)
    wandb.log({"loss": model_01_trainer.train_losses[-1]})

    # %% SWEEP config and setup

#Defining sweep config
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "loss"},
    "parameters": {
        "NUM_NEURONS": {"values": [16, 32, 64,128,1256]},
        "NUM_LAYERS": {"values": [3, 6, 9,12,15]},
        "LEARNING_RATE": {"values": [0.001,0.01,0.1,0.5]},
        "EPOCHS": {"values": [100,300,500,900]},
        "TEST_SIZE": {"values": [0.05,0.1,0.2,0.3]},
    },
}


pprint.pprint(sweep_configuration)
sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")
wandb.agent(sweep_id,main,count =50)
