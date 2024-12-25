# Neural Network for Boston Housing Dataset

This README provides instructions on how to execute the neural network code for the Boston Housing Dataset.

## Prerequisites

Before running the code, ensure you have the following installed:
- Python 3.x
- NumPy
- Pandas

You can install the required packages using pip:

```
pip install numpy pandas
```

## Dataset

The code expects a file named `housing.csv` in the same directory as the script. This file should contain the Boston Housing Dataset.

## Running the Code

1. Place the `P4_Neural_Networks.ipynb` file and the `housing.csv` file in the same directory.

2. Open a terminal or command prompt and navigate to the directory containing the files.

3. Run the script using Python:

   ```
   python neural_networks.py
   ```

4. The script will prompt you for input three times, each for a different configuration (a, b, and c). For each configuration:
   - Enter the number of neurons in the hidden layer when prompted.
   - Enter the learning rate when prompted.

5. After each set of inputs, the script will run:
   - 5-fold cross-validation
   - 10-fold cross-validation

6. The results for each fold will be displayed, showing the loss and accuracy (R²) for that fold.

7. After all folds are complete, the average loss and accuracy across all folds will be displayed.

## Example Input/Output

Here's an example of what you might see when running the script:

```
Enter the number of neurons in the hidden layer: 10
Enter the learning rate: 0.01
5-Fold Cross-Validation for Case (a1): Hidden Layer Size = 10, Learning Rate = 0.01
Fold 1, Loss: 0.0123, Accuracy (R²): 0.8765
...
Average Loss: 0.0135, Average Accuracy (R²): 0.8701

10-Fold Cross-Validation for Case (a2): Hidden Layer Size = 10, Learning Rate = 0.01
...

Enter the number of neurons in the hidden layer: 20
Enter the learning rate: 0.005
...
```

## Modifying the Code

If you want to modify the neural network architecture or change the features used from the dataset, you can edit the `P4_Neural_Networks.ipynb` file. The main areas you might want to modify are:

- The `preprocess_data` function to change which features are used.
- The `NeuralNetwork` class to modify the network architecture.
- The `k_fold_cross_validation` function to change the number of epochs or other training parameters.

