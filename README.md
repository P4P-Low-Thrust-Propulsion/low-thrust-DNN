# Research Compendium: Learning How to Hop Between Space Objects Using Low-Thrust Propulsion

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
   - [Data](#data)
   - [Machine Learning](#machine-learning)
   - [Plots and Results](#plots-and-results)
   - [Documentation](#documentation)
3. [How to Run the Code](#how-to-run-the-code)
   - [Dependencies](#dependencies)
   - [Setup Instructions](#setup-instructions)
   - [Running the Models](#running-the-models)
4. [Experiments and Procedures](#experiments-and-procedures)
   - [Data Generation](#data-generation)
   - [Model Training](#model-training)
   - [Evaluation Metrics](#evaluation-metrics)
   - [Hyper-Parameter Tuning](#hyper-parameter-tuning)
5. [Additional Notes](#additional-notes)

---

## Overview
This research compendium accompanies the University of Auckland Part IV Engineering Project carried our by Stephen Ng and Theo McIntosh: `Project #55: Learning how to hop between space objects using low-thrust propulsion`.  It focuses on developing a deep neural network (DNN) model to optimise low-thrust space transfers. It includes all the code, datasets, plots, and supplementary materials used throughout the research and experimentation process. The project was inspired by the Global Trajectory Optimisation Competition (GTOC) and aimed at computing the feasibility for low-thrust spacecraft trajectories using DNNs rather than solving optimal control problem (OCP), as this would reduce computational costs.

The compendium includes:
- Code for generating datasets, preprocessing, training neural networks and other machine learning models, and generating plots.
- Details of the experimental procedures, including data generation, hyper-parameter tuning, and model evaluations.
- Datasets used in the project, including both the raw and processed data.
- Plots and results that were generated for analysis and included in the project report.
- Documentation of experimental setups, tools, and conditions.

---

## Project Structure

### Data
- **`data/`**: All the data used in the project, including raw and processed datasets along with the code files used to develop and manipulate the data.
  - **`lambert/`** 
    - **`datasets/`**: Contains the generated and manipulated Lambert transfer datasets.
      - **`processed/`**: Processed datasets used for training and testing the DNN model. This includes datasets transformed into the transfers reference frame.
    - **`data_creation.py`**: Script to generate Lambert transfers.
  - **`low_thrust/`**
    - **`datasets/`**: Contains the generated initial low-thrust datasets as well as the manipulated datasets. 
      - **`initial/`**: Original generated datasets of low-thrust transfers.
      - **`processed/`**: Manipulated low-thrust transfer datasets, that have been rotated into the RTN frame.
    - **`data_manipulation.py`**: Script used to rotate and manipulate the original datasets.
    - **`random_data_selection.py`**: Script used to generate the original datasets.
    - **`ratio_dataset.py`**: Script used to test whether a dataset that held ratios would work better.
    - **`reachability_data.py`**: Script used to generate the datasets for the reachability analysis.
    - **`reachability_lambert.py`**: Script used to generate the Lambert approximated cost.  For the delta-v cost analysis.
  
### Machine Learning
- **`src/`**: Contains all the source code files used for training and analysis, as well as generated models.
  - **`dnn/`**
    - **`nn_prediction.py`**: Script for making predictions using the trained DNN model.
    - **`nn_training.py`**: Script for training the DNN model, including data loading, model initialisation, and optimisation steps.
    - **`sweep.py`**: Script for running hyper-parameter sweeps using WandB to optimise the DNN architecture.
  - **`models/`**
    - **`saved_models/`**: Directory where trained models, scalers, and related assets are stored for reuse.
    - **`DNN.py`**: Python file containing the definition of the DNN architecture used for training and predictions.
  - **`random_forest/`**
    - **`random_forest_prediction.py`**: Script for making predictions using the trained Random Forest model.
    - **`random_forest_training.py`**: Script for training the Random Forest model, including data preprocessing and grid search for hyper-parameters.
  - **`results/`**
    - **`dv_estimate.py`**: Script to estimate and visualise the delta-v requirements based on transfer predictions.
    - **`reachability_visualisations.py`**: Script to generate reachability plots, showing feasible regions of space for low-thrust transfers.

### Plots and Results
- **`plots/`**: Contains all generated plots and visualisations, including:
  - **`data_creation/`**: Contains plots related to the generation and preprocessing of the dataset used for machine learning models.
  - **`neural_network/`**
    - **`dv_estimate/`**: Plots showing delta-v estimates for spacecraft transfers predicted by the neural network model.
    - **`lambert/`**: Visualisations of Lambert transfer solutions compared to low-thrust transfer predictions.
    - **`low_thrust/`**: Plots focusing on the performance of the neural network in predicting low-thrust masses, including comparisons with true values.
    - **`reachability_analysis/`**: Plots showing the reachability regions of low-thrust transfers, indicating feasible and infeasible transfer zones predicted by the neural network.
  - **`random_forest_regression/`**
    - **`low_thrust/`**: Plots displaying the performance of the Random Forest model in predicting low-thrust masses, including comparisons with actual values.
    - **`reachability_analysis/`**: Visualisations of reachability analysis using the Random Forest model, showing regions of space reachable by low-thrust transfers.

### Documentation
- **`docs/`**: Contains additional documentation, such as:
  - Future work: Details future work that could be done in this project.
  - Experiment log (wandb files): A detailed record of each experiment, including hyper-parameters sweeps using wandb.  These are stored locally and are included in the **`.gitignore`** 
  - Would contain configuration files used for setting up experiments (e.g., YAML files for WandB experiments).

---

## How to Run the Code

### Dependencies
This project uses the following libraries and tools:
- **Python 3.11**: The programming language used for the entire project.
- **NumPy**: For efficient numerical operations and array manipulation.
- **Pandas**: For data manipulation and analysis, especially for handling data in DataFrames.
- **Argparse**: For parsing command-line arguments.
- **Matplotlib**: For generating plots and visualisations.
- **Astropy**: For astronomical calculations and handling time and unit conversions.
- **Poliastro**: For orbital mechanics and trajectory calculations.
- **PyTorch**: For building and training the deep neural network.
- **Scikit-learn**: For machine learning utilities such as data preprocessing, model training, and evaluation.
- **Joblib**: For saving and loading Python objects efficiently, including models and scalers.
- **WandB**: For tracking experiments and hyper-parameter tuning during the training process.
- **TQDM**: For progress bars during loops, particularly useful in training machine learning models.
- **SciPy**: For scientific computations and advanced mathematical functions.
  
Other dependencies can be found in the provided `requirements.txt` file.

### Setup Instructions
1. Clone this repository:
    ```bash
    git clone https://github.com/P4P-Low-Thrust-Propulsion/low-thrust-DNN.git
    cd low-thrust-DNN
    ```
2. Install the required dependencies (Please set up a Conda environment): 
    ```bash
    conda list -e > requirements.txt
    ```

### Running the Models
The following instructions show how to generate data and train a model. The training scripts allow for configuration of hyper-parameters through changing the code in the file.  However, like the data generation scripts they will be updated so that the command line or a configuration file (**`config.yaml`**) can be used.

Example commands to generate data and run a model:

#### Data Generation Lambert 
1. Lambert dataset generation requires running the following script with the required number of transfers you want to create along with the eccentricity limit:
    ```
    python data/lambert/data_creation.py --NUM_TRANSFERS 1000 --ECC 1
    ```
   
#### Data Generation low-thrust 
1. For low-thrust you first have to generate a large number of transfers via the following command:
    ```
    python data/low_thrust/random_data_selection.py --NUM_TRANSFERS 2000
    ```
2. You then have to manipulate that created dataset so that it is in the correct frame of reference for training:
    ```
    python data/low_thrust/data_manipulation.py --DATA_NAME transfer_statistics_2K.csv
    ```
    

#### Train Models
1. To train the models you must define in the code which dataset (file) you are wanting to train.  You can also modify the hyper-parameters. Then run the following:.
    ```
    python src/dnn/nn_training.py
    ```
2. To evaluate the model performance you must define in the code both the dataset you trained on and the model that was saved after the training process. Then run the following::
    ```
    python src/random_forest/random_forest_training.py
    ```
#### Evaluate Models Performance
1. To train the models you must define in the code which dataset (file) you are wanting to train.  You can also modify the hyper-parameters. Then run the following::
    ```
    python src/dnn/nn_prediction.py
    ```
2. To evaluate the model performance you must define in the code both the dataset you trained on and the model that was saved after the training process. Then run the following:
    ```
    python src/random_forest/random_forest_prediction.py
    ```
Generated plots will be displayed interactively and model checkpoints will be saved in the **`/models`** directory.

---

## Experiments and Procedures

### Data Generation
Low-thrust transfer datasets were generated using numerical solutions to the OCP. These datasets consist of spacecraft positions, velocities, and mass at intermediate points during the transfer. The RTN (Radial, Transverse, Normal) reference frame was used for the analysis, and a total of 469 low-thrust transfers were computed and then sampled to generate larger datasets to train on.

### Model Training
The DNN model was trained to predict maximum initial and final mass of a spacecraft for a feasible low-thrust transfer. The training process involved:
- Splitting data into training and testing sets.
- Scaling and normalising data for better model performance.
- Using MSE as the loss function

### Evaluation Metrics
The primary metrics used to evaluate the model were:
- **Mean Squared Error (MSE)**
- **R2**
- **Mean Percentage Absolute Error (MPAE)**

### Hyper-Parameter Tuning
Various hyper-parameters were explored, including the number of layers, learning rate, and batch size. Random search and WandB were used to track and compare different configurations. The final model was chosen based on a balance of performance and computational efficiency.

---

## Additional Notes
- The project focused on finding an alternative to the OCP problem using machine learning techniques but encountered challenges with high mass transfer predictions. This remains an area for future research.
- Datasets and model architecture were inspired by previous GTOC challenges.  However, a noval approach was taken, that was able to achieve the output of being able to determine feasibility and transfer cost via a single DNN.