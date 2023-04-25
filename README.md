# Alzheimer's Classification using 2D MRI Images with Deep Learning Techniques

This repository contains the code and Jupyter Notebook for a deep learning model that classifies MRI brain images into three classes: Very Mild Demented, Mild Demented, Non Demented. The model is based on Convolutional Neural Networks (CNNs) and was developed using the Keras library with TensorFlow backend.

## Dataset

The dataset used in this study consists of MRI images of the brain. It has 3 different classes: Alzheimer's disease, Mild Cognitive Impairment (MCI), and non-dementia. The original dataset is available on Kaggle at the following link: https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images. The dataset was preprocessed and split into training, validation, and testing sets with a ratio of 80:10:10. The patient’s ages were not specified, and no additional information about them was supplied in the dataset. The image size in the dataset is 227*227. For this study, we split the dataset into three different sets. In the training set, it has 4480 images, the validation set consist of 1280 images and the test set consist of 640 images. The split ratio was 7:2:1 in between training, validation and test set.

The data was preprocessed and split into training, validation, and testing sets with a ratio of 80:10:10. Figure 1 shows some sample MRI images from the dataset.

## Model Architecture

The notebook contains multiple model architectures, including:

- AlzheimerNet
- Pre-trained VGG19 
- VGG19 Feature extraction with DNN 
- Pre-trained ResNet50 
- ResNet50 Feature extraction with DNN 

Each model was trained using the categorical cross-entropy loss function and the Adam optimizer.

## Techniques Used

The following deep learning techniques were implemented in this project:

- Convolutional Neural Networks (CNNs) for image classification
- Data preprocessing techniques including image resizing, normalization, and augmentation
- Transfer learning to leverage pre-trained models for feature extraction
- Categorical cross-entropy loss function for multi-class classification
- Adam optimizer for model training

## Requirements

- Access to Google Colaboratory with GPU runtime
- This GitHub repository: https://github.com/MJHossainS/Alzheimer-classification-using-2D-MRI-images-with-deep-learning-techniques-

## Usage

1. Open Google Colaboratory and select "File" -> "Open notebook"
2. In the "GitHub" tab, enter the following repository URL: https://github.com/MJHossainS/Alzheimer-classification-using-2D-MRI-images-with-deep-learning-techniques-
3. Open the "Alzheimer's classification.ipynb" notebook and run the cells in order to train and evaluate the model.

## Results

The models achieved promising results for the classification of Alzheimer's disease, MCI, and non-dementia classes. The exact accuracy and other metrics depend on the specific model used and are reported in the notebook.


## Acknowledgments

- The Keras and TensorFlow communities for providing the tools and resources to develop deep learning models.
