# Future Works

Several key avenues can be explored to enhance the accuracy, robustness, and applicability of the DNN model developed in this research. These improvements can contribute to better predictions, more versatile applications, and practical integration into real-world mission planning.

## 1. Address Dataset Imbalance
One of the primary areas for potential improvement is handling dataset imbalance. Techniques such as ADASYN (Adaptive Synthetic Sampling) or synthetic data generation could be employed to augment the dataset, particularly in sparsely represented mass ranges. By improving the representation in these regions, the model would be more capable of accurately predicting extreme conditions and reduce errors seen in underrepresented data points.

## 2. Explore Advanced Neural Network Architectures
Further investigation into more advanced neural network architectures could significantly improve performance. Architectures such as Recurrent Neural Networks (RNNs) or Transformers may be better suited to capture temporal and spatial dependencies, especially for multi-target transfers where spacecraft experience dynamic interactions over time. These models could offer more nuanced insights into the complexities of space transfers.

## 3. Apply Transfer Learning
Another promising direction is the application of transfer learning to generalise the model across various types of space missions. This would enable the DNN to handle a broader range of transfer scenarios, including interplanetary missions, asteroid tours, or complex orbital maneuvers. Transfer learning would also make the model more versatile and reduce the amount of mission-specific data required for accurate predictions.

## 4. Integration with Mission Design Tools
Integrating the DNN model with real-time mission design software, such as [Poliastro](https://docs.poliastro.space/en/stable/) or NASA’s General Mission Analysis Tool (GMAT), could provide immediate feasibility assessments during the mission planning phase. This integration would allow mission designers to quickly evaluate the feasibility of different trajectories and improve the overall efficiency of the mission planning process.

## 5. Configuration File for Hyper-Parameter Tuning
The current training scripts require manual adjustment of hyper-parameters within the code itself. To improve usability, a **`config.yaml`** file should be introduced to allow for easy configuration of model hyper-parameters and training settings. This will enable users to modify parameters such as the learning rate, batch size, and model architecture directly from the command line or via configuration files, streamlining the training process.

