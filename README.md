# Deep Learning for Emotion Detection

## Overview

This project revolves around identifying facial emotions using Convolutional Neural Networks (CNNs). The main aim is to develop a system capable of recognizing emotions from faces in a video stream. The key tasks involved are:

1. Emotion categorization
2. Face-mesh creation for feature extraction
3. Facial tracking

## Setting up conda environment from conda_env.yml

To create a new conda environment from the `conda_env.yml` file, run the following command:

```sh
conda env create -f conda_env.yml
```

To activate the newly created environment, use:

```sh
conda activate emotion-detection
```

## Running the Code

### 1. Training the Emotion Model

To train the emotion detection model, execute the `train.py` script:

```sh
python ./scripts/train.py
```

### 2. Testing Emotion Prediction

To evaluate emotions on the test dataset and compute accuracy, run the `predict.py` script:

```sh
python ./scripts/predict.py
```

### 3. Live Video Emotion Detection

To detect emotions from a live video feed, use the `predict_live_stream.py` script:

```sh
python ./scripts/predict_live_stream.py
```

## Emotion Recognition Model Architecture

I employed a Sequential model, which proved more effective. This approach was inspired by [this project on GitHub](https://www.kaggle.com/code/farneetsingh24/ck-facial-emotion-recognition-96-46-accuracy).

### Key Features of the Model

🔍 **Deep Convolutional Layers**: Utilizing multiple convolutional layers, particularly high-depth layers (512 and 256 filters in early stages), the model captures intricate and subtle facial features. High-depth filters enhance the model’s ability to learn detailed nuances essential for precise emotion classification.

### Fine-Tuned Parameters

🧩 **ELU Activation > RELU**  
ELU (Exponential Linear Unit) activation functions are preferred over the traditional ReLU, as they facilitate faster learning and convergence. ELU also mitigates the vanishing gradient problem, which is advantageous for training deeper networks.

🧩 **Dropout Set to 22%**  
Strategically placed dropout layers with a 23% rate help prevent overfitting by randomly omitting units during training, fostering a more generalized model. MaxPooling is employed to reduce the spatial dimensions of output volumes, lowering computational load while extracting dominant features invariant to minor input changes.

🧩 **Data Augmentation - Rotation at 17 Degrees**  
The image data generator enhances training images via rotations, shifts, and flips, creating a robust model less sensitive to input variations, thus mirroring real-world scenarios where facial expressions vary significantly.

🧩 **Nadam Optimizer > Adam**  
The Nadam optimizer is selected for its efficiency in managing sparse gradients and avoiding premature convergence. Learning rate adjustments via the ReduceLROnPlateau callback ensure fine-tuning as the model nears optimal performance, boosting accuracy without overshooting during training.

🧩 **Batch Size Set to 64**  
🧩 **2x 128 Layers in the Middle Function as Attention Mechanisms**  
🧩 **Smaller Test Set with Stratified Sampling (`stratify=y_train`)**

## Model Architecture Details

![Model Architecture](./images/model_architecture.png)
