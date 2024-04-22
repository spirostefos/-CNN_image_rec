# Overview: Histopathological Lung and Colon Cancer Image Classification using Convolutional Neural Networks (CNNs)

This project aims to classify histopathological lung and colon cancer images using Convolutional Neural Networks (CNNs), which is crucial for early diagnosis and effective patient care. The dataset consists of 25,000 images resized to 120 x 120 pixels, covering both benign and malignant tissues. Initially, a baseline CNN model is trained and evaluated. Subsequently, hyperparameter tuning and architecture adjustments are performed, drawing inspiration from existing literature. Transfer learning techniques are then explored, leveraging the pre-trained VGG16 model. The goal is to develop precise models for medical image classification, aiding in accurate diagnosis and treatment planning.

This repository contains Python scripts for training and evaluating neural network models for image classification tasks using the TensorFlow and Keras libraries. Three different models have been implemented and evaluated:

1. **Custom Convolutional Neural Network (CNN)**
2. **Transfer Learning with VGG16**
3. **Enhanced Model from Literature**

## File Structure
- `data120.npy`: Numpy array containing image data.
- `labels120.npy`: Numpy array containing corresponding labels.
- `README.md`: This file providing an overview of the project.
- `requirements.txt`: Text file listing all the required dependencies for the project.
- `main.npy`: Main file to run.

## Dataset
- First, download the dataset from the following link as a .zip file: [Histopathological Lung and Colon Cancer Dataset](https://academictorrents.com/details/7a638ed187a6180fd6e464b3666a6ea0499af4af)
- Extract the zip file into a folder. You can then upload this dataset to your Google Drive folder or use it on any platform you want.
- For more information about the dataset, you can refer to this article: [Article Link](https://arxiv.org/abs/1912.12142)


### Requirements:
- Python 3.x
- TensorFlow
- Keras
- numpy
- matplotlib
- scikit-learn

## Usage
1. Clone the repository to your local machine: `git clone https://github.com/spirostefos/CNN_image_rec.git`
2. Navigate to the project directory: `cd CNN_image_rec`
3. Install the required dependencies: `pip install -r requirements.txt`
4. Run the main script to execute the project: `python main.py`
5. Execute the code cells in your preferred environment, ensuring the dataset is accessible.
6. Follow the provided code comments for each step, including data preprocessing, model training, evaluation, and visualization.
7. Experiment with different hyperparameters, architectures, and optimization techniques as described in the code.
8. Utilize the provided evaluation functions to assess the performance of the models, including ROC curves, classification reports, confusion matrices, etc.

## Instructions:

#### 1. Custom Convolutional Neural Network (CNN):
- **create_model_final function**: Defines the architecture of the custom CNN model.
- **Training**: Trains the model using the defined architecture with specified hyperparameters.
- **Evaluation**: Evaluates the model's performance on the test dataset and generates a detailed report.

#### 2. Transfer Learning with VGG16:
- **Loading VGG16 Model**: Loads the pre-trained VGG16 model without the output layer and specifies a new input shape for images.
- **Fine-tuning**: Adds new classifier layers on top of the pre-trained VGG16 layers and trains the model with frozen pre-trained layers.
- **Evaluation**: Evaluates the fine-tuned VGG16 model's performance on the test dataset and generates a detailed report.

#### 3. Enhanced Model from Literature:
- **Model Architecture**: Constructs a sequential model with enhanced architecture based on literature.
- **Training**: Compiles and trains the model using the specified optimizer and hyperparameters.
- **Evaluation**: Evaluates the enhanced model's performance on the test dataset and generates a detailed report.



