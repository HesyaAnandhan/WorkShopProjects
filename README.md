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

