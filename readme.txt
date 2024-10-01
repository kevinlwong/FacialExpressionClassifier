Kaggle username: kevinlwong
Final Kaggle leaderboard score: 0.82244
Leaderboard Placement: 1st Place as of 8/5/2024 10:36 PM

###README: CS4210 Machine Learning Assignment 3 - Task 1

---

#### Project Overview:
This Jupyter Notebook is part of **Assignment 3 - Task 1** for the **CS4210 Machine Learning and its Applications** course during the Summer Semester 2024. It focuses on implementing and training a Convolutional Neural Network (CNN) for image classification tasks using the PyTorch framework.

The CNN is trained to classify facial expressions into three categories: **Angry, Happy, and Neutral**. The notebook covers the following key steps:
- Data preprocessing
- Building the CNN model
- Training and validating the model
- Making predictions on the test data

---

#### Key Components:

1. **Data Preprocessing**:
   - **Transformations** are applied to the training and validation datasets to augment the data (random flips, rotations) and normalize it.
   - Data is loaded from CSV files representing pixel values and reshaped into 48x48 grayscale images.
   - The data is split into training and validation sets using `train_test_split`.
   - Custom datasets are created using PyTorch's `Dataset` class, and `DataLoader` objects are used for efficient batch processing.

2. **Model Architecture**:
   - The model is a **Convolutional Neural Network (CNN)** designed for image classification.
   - The architecture includes four convolutional layers with batch normalization, followed by fully connected layers with dropout for regularization.
   - **Leaky ReLU** activation functions are used along with max-pooling for downsampling.
   - The final layer outputs predictions for 3 classes.

3. **Training and Validation**:
   - The model is trained using the **Adam optimizer** and **cross-entropy loss**.
   - **Gradient clipping** is applied to prevent exploding gradients.
   - Training results are printed per epoch, showing the loss and validation accuracy.
   - The notebook trains the model for 20 epochs, with results improving as training progresses.

4. **Making Predictions**:
   - After training, the model makes predictions on the test dataset, which is not labeled.
   - Predictions are saved to a CSV file in a format ready for submission.

---

#### Files:
- **train_data.csv**: Training data containing pixel values.
- **train_target.csv**: Labels corresponding to the training data.
- **test_data.csv**: Test data with pixel values (no labels).
- **submission.csv**: Output file with predictions.

---

#### Requirements:
- PyTorch
- Scikit-learn
- Pandas
- NumPy

---

#### Instructions to Run the Notebook:
1. Install the required libraries using `pip install torch scikit-learn pandas numpy`.
2. Ensure the datasets (`train_data.csv`, `train_target.csv`, `test_data.csv`) are available in the same directory as the notebook.
3. Run the notebook cells sequentially to train the model and generate the predictions.
4. The output will be saved in `submission.csv`.

---

#### Contact Information:
- Author: **Kevin Wong**
- Course: CS4210 Machine Learning and its Applications - Summer Semester 2024

