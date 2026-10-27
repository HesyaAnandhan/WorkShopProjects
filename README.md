# WorkShopProjects
# Dog-Cat Image Classification Using CNN

## Project Overview

This project implements a **Convolutional Neural Network (CNN)** to classify images into two categories: **Cat** and **Dog**.

The model is developed using Python, TensorFlow, and Keras in Google Colab. It uses three convolutional layers to automatically learn visual features from images and make predictions.

## Project Objectives

- Build a CNN model for binary image classification.
- Understand how convolutional and pooling layers extract image features.
- Train and evaluate a deep learning model.
- Predict whether an uploaded image contains a cat or a dog.
- Explore image preprocessing and data augmentation techniques.

## Model Architecture

The CNN contains **exactly three convolutional layers**.

| Layer | Configuration | Purpose |
|---|---|---|
| Input | 128 × 128 × 3 | Accepts RGB images |
| Data Augmentation | Flip, rotation, zoom | Improves training variety |
| Convolution 1 | 32 filters, 3 × 3 | Learns basic image features |
| Max Pooling 1 | 2 × 2 | Reduces spatial dimensions |
| Convolution 2 | 64 filters, 3 × 3 | Learns more complex features |
| Max Pooling 2 | 2 × 2 | Reduces feature-map dimensions |
| Convolution 3 | 128 filters, 3 × 3 | Learns higher-level features |
| Max Pooling 3 | 2 × 2 | Reduces spatial dimensions |
| Flatten | — | Converts feature maps into a vector |
| Dense | 128 neurons, ReLU | Learns classification patterns |
| Dropout | 0.5 | Helps reduce overfitting |
| Output | 1 neuron, Sigmoid | Predicts the probability of a dog |

### Model Compilation

- **Optimizer:** Adam
- **Loss Function:** Binary Crossentropy
- **Evaluation Metric:** Accuracy
- **Hidden-layer activation:** ReLU
- **Output activation:** Sigmoid

## Dataset

The project uses the **Cats vs Dogs image dataset** through TensorFlow Datasets.

Dataset source: [TensorFlow Datasets — Cats vs Dogs](https://www.tensorflow.org/datasets/catalog/cats_vs_dogs)

The images are divided into three subsets:

- **Training set:** 80%
- **Validation set:** 10%
- **Test set:** 10%

The dataset is used to train the model, monitor its performance, and evaluate predictions on unseen images. Some corrupted images may be skipped during dataset preparation.

## Data Preprocessing

The preprocessing pipeline includes:

1. Resizing images to 128 × 128 pixels.
2. Converting images to floating-point values.
3. Normalizing pixel values to the range 0–1.
4. Applying data augmentation during training.
5. Batching and prefetching data for efficient processing.

## How to Run the Project

### Option 1: Google Colab

1. Open the notebook in this repository.
2. Click **Open in Colab**, or upload the `.ipynb` file to [Google Colab](https://colab.research.google.com/).
3. Select a GPU runtime if available.
4. Run the dataset preparation cell.
5. Build and compile the CNN model.
6. Train the model using the training dataset.
7. Evaluate the model using the test dataset.
8. Upload a new image to test the prediction.

### Option 2: Run Locally

Clone the repository:

```bash
git clone https://github.com/HesyaAnandhan/dog-cat-cnn-classification.git
```

Move into the project directory:

```bash
cd dog-cat-cnn-classification
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the project notebook and execute the cells in order.

**Note:** Local execution requires a compatible Python and TensorFlow environment. Google Colab is an alternative if local setup causes issues.

## Model Evaluation

The model can be evaluated using:

- Training accuracy
- Validation accuracy
- Test accuracy
- Training and validation loss
- Classification report, including precision, recall, and F1-score

**Results:** Add the actual values obtained from your completed model evaluation.

| Metric | Result |
|---|---|
| Training Accuracy | To be measured |
| Validation Accuracy | To be measured |
| Test Accuracy | To be measured |
| Precision | To be measured |
| Recall | To be measured |
| F1-Score | To be measured |

Results depend on the training configuration, dataset, and model performance.

## Prediction

After training, the model can predict the class of a new image.

- **Cat:** The model predicts that the image contains a cat.
- **Dog:** The model predicts that the image contains a dog.

The sigmoid output represents the model's estimated probability for the dog class, assuming the labels are encoded as Cat = 0 and Dog = 1.

A prediction probability is not a guarantee that the classification is correct.

## Technologies Used

- Python
- TensorFlow
- Keras
- TensorFlow Datasets
- NumPy
- Matplotlib
- Scikit-learn
- Google Colab
- Jupyter Notebook
- Git and GitHub

## Project Structure

```text
dog-cat-cnn-classification/
│
├── dog_cat_cnn_three_layers.ipynb
├── README.md
├── requirements.txt
└── outputs/
    └── results.json
```

The structure above is a suggested layout. Keep the actual notebook filename and folders consistent with the files in your repository.

## Key Learning Outcomes

- Understanding CNN architecture and image classification.
- Implementing convolutional and max-pooling layers.
- Using data augmentation and normalization.
- Training and evaluating a binary classification model.
- Making predictions on new images.
- Organizing a machine learning project for GitHub.

## Future Improvements

- Experiment with different image sizes and hyperparameters.
- Improve generalization by tuning data augmentation and regularization.
- Compare the custom CNN with transfer-learning models.
- Develop a simple web interface for image uploads and predictions.
- Save and load the trained model for reuse.

## References

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)
- [TensorFlow Datasets — Cats vs Dogs](https://www.tensorflow.org/datasets/catalog/cats_vs_dogs)
- [Google Colab](https://colab.research.google.com/)

## Author

**Hesya Anandhan**

GitHub: [@HesyaAnandhan](https://github.com/HesyaAnandhan)

# Artificial Neural Network (ANN) for Binary Classification Using the make_moons Dataset

## Project Overview

This project demonstrates the development of an Artificial Neural Network (ANN) for binary classification using the `make_moons` dataset from Scikit-learn. The model is implemented using TensorFlow and Keras to learn nonlinear decision boundaries and classify data points into two classes.

The project covers dataset generation, data preprocessing, ANN architecture design, model training, performance evaluation, visualization, and prediction on new samples.

## Objectives

- Generate a synthetic binary classification dataset using Scikit-learn.
- Build an ANN with two hidden layers.
- Train the model using the Adam optimizer and binary cross-entropy loss.
- Evaluate the model using accuracy, precision, recall, F1-score, and test loss.
- Visualize model accuracy, loss, confusion matrix, and decision boundary.
- Save and reload the trained model and preprocessing scaler.
- Predict the class of a new input sample.

## Technologies Used

- Python
- Google Colab
- TensorFlow
- Keras
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Joblib

## Dataset

The project uses the `make_moons` function from Scikit-learn to generate a synthetic dataset containing two interleaving moon-shaped classes.

Dataset configuration:

- Total samples: 2,000
- Number of features: 2
- Number of classes: 2
- Noise: 0.25
- Training samples: 1,600
- Testing samples: 400
- Random state: 42

The dataset is divided into training and testing sets using an 80:20 split. Feature scaling is performed using `StandardScaler`.

## ANN Architecture

The model is developed using the Keras Sequential API.

| Layer | Neurons | Activation Function |
|---|---:|---|
| Input Layer | 2 features | None |
| Hidden Layer 1 | 16 | ReLU |
| Hidden Layer 2 | 8 | ReLU |
| Output Layer | 1 | Sigmoid |

Total trainable parameters: 193

The model uses the Adam optimizer and binary cross-entropy loss. Training is configured for up to 100 epochs with a batch size of 32 and early stopping based on validation loss.

## Project Workflow

1. Import the required Python libraries.
2. Generate the synthetic `make_moons` dataset.
3. Visualize the distribution of the two classes.
4. Split the dataset into training and testing sets.
5. Scale the input features using `StandardScaler`.
6. Build and compile the ANN model.
7. Train the model and monitor training and validation performance.
8. Plot the accuracy and loss curves.
9. Evaluate the model using the test dataset.
10. Generate the confusion matrix and decision boundary visualization.
11. Predict the class of a new sample.
12. Save the trained model, scaler, and evaluation results.
13. Reload the saved model and verify its prediction.

## Model Evaluation Results

The trained model achieved the following results on the test dataset.

| Metric | Result |
|---|---:|
| Test Loss | 0.1637 |
| Test Accuracy | 92.50% |
| Precision | 91.67% |
| Recall | 93.50% |
| F1-Score | 92.57% |

The model correctly classified 370 out of 400 test samples.

These results indicate that the ANN learned the nonlinear classification pattern in the synthetic dataset and achieved similar performance across both classes.

## Visualizations

The project generates the following visualizations:

- Dataset distribution plot
- Model accuracy graph
- Model loss graph
- Confusion matrix
- ANN decision boundary

These visualizations help evaluate the model's learning process, classification performance, and ability to separate the two classes.

## Sample Prediction

The trained ANN was tested using a new input sample.

Input:

```python
[0.5, 0.2]
```

Prediction results:

| Output | Value |
|---|---:|
| Class 0 Probability | 0.3155 |
| Class 1 Probability | 0.6845 |
| Predicted Class | 1 |

The model predicts Class 1 because its estimated probability is greater than the classification threshold of 0.5.

## Project Files

```text
ANN-make-moons-Binary-Classification/
├── ANN_make_moons_Binary_Classification.ipynb
├── README.md
├── requirements.txt
├── evaluation_results.csv
├── ann_make_moons.keras
└── scaler.joblib
```

The notebook filename may differ depending on the name used when downloading it from Google Colab.

## Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/HesyaAnandhan/ANN-make-moons-Binary-Classification.git
cd ANN-make-moons-Binary-Classification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

Open the Jupyter Notebook:

```bash
jupyter notebook
```

Select the project notebook and run the cells in order.

Alternatively, upload the notebook to Google Colab and execute it there.

## Saved Model

The trained model is saved in Keras format as `ann_make_moons.keras`. The fitted `StandardScaler` is saved as `scaler.joblib`, allowing new samples to be transformed using the same preprocessing method used during training.

The saved model and scaler were reloaded successfully, and the sample prediction was reproduced with a Class 1 probability of 0.6845.

## Conclusion

This project demonstrates how an Artificial Neural Network can solve a nonlinear binary classification problem using a synthetic dataset. The model achieved 92.50% test accuracy and was successfully saved, reloaded, and tested with a new sample.

The project provides practical experience with neural network architecture, model training, feature scaling, evaluation metrics, data visualization, and model persistence.

## Author

Hesya Anandhan

GitHub: [HesyaAnandhan](https://github.com/HesyaAnandhan)

Project Repository: [ANN-make-moons-Binary-Classification](https://github.com/HesyaAnandhan/ANN-make-moons-Binary-Classification)
