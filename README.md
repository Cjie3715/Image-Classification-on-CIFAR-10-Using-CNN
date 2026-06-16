# Image Classification on CIFAR-10 Using CNN

**Author:** Xi-Jie Zeng (Yuan Ze University)

## Project Description

This repository contains the source code for a final project that implements an image classification model on the CIFAR-10 dataset using a custom Convolutional Neural Network (CNN) built with PyTorch.

The trained model achieves an overall test accuracy of **83.23%** on the CIFAR-10 dataset.

---

## Environment & Requirements

This project was developed and tested using the following environment:

| Component   | Version                      |
| ----------- | ---------------------------- |
| IDE         | Visual Studio Code (VS Code) |
| Python      | 3.9                          |
| PyTorch     | 2.7.1                        |
| Torchvision | Latest compatible version    |
| Matplotlib  | Latest compatible version    |

---

## Dataset Information

This project uses the **CIFAR-10** dataset, which contains:

* **60,000** color images
* Image size: **32 × 32**
* **10 classes**
* **6,000 images per class**
* **50,000 training images**
* **10,000 test images**

### Classes

| Label |
| ----- |
| Plane |
| Car   |
| Bird  |
| Cat   |
| Deer  |
| Dog   |
| Frog  |
| Horse |
| Ship  |
| Truck |

> **Note:** The CIFAR-10 dataset will be automatically downloaded using `torchvision.datasets.CIFAR10` when running the training or evaluation scripts. No manual download is required.

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/Cjie3715/Image-Classification-on-CIFAR-10-Using-CNN.git
cd Image-Classification-on-CIFAR-10-Using-CNN
```

### Install Dependencies

```bash
pip install torch==2.7.1 torchvision matplotlib
```

> If you plan to use GPU acceleration, please ensure that your CUDA version is compatible with PyTorch 2.7.1.

---

## Project Structure

```text
Image-Classification-on-CIFAR-10-Using-CNN/
│
├── train_model.py
├── evaluate_model.py
├── dataset/
│
├── CIFAR_10_20_acc_0.8323.pth
├── loss_curves.png
├── accuracy_bar_chart.png
│
└── README.md
```

### File Descriptions

#### train_model.py

Responsible for:

* Loading CIFAR-10 dataset
* Data augmentation
* CNN model definition
* Training loop
* Saving model checkpoints
* Generating training loss curves

#### evaluate_model.py

Responsible for:

* Loading saved model weights
* Evaluating test accuracy
* Computing per-class accuracy
* Generating accuracy bar charts

#### dataset/

Directory where CIFAR-10 will be automatically downloaded.

#### .pth Files

Saved model weights generated during training.

Example:

```text
CIFAR_10_20_acc_0.8323.pth
```

#### Output Images

* `loss_curves.png`
* `accuracy_bar_chart.png`

---

## Usage

### 1. Train the Model

Run the following command:

```bash
python train_model.py
```

The script will:

* Train the CNN for 20 epochs
* Save model weights after each epoch
* Generate `loss_curves.png`

---

### 2. Evaluate the Model

Run:

```bash
python evaluate_model.py
```

When prompted, enter the name of the trained model file:

```text
CIFAR_10_20_acc_0.8323.pth
```

The script will:

* Calculate overall test accuracy
* Calculate per-class accuracy
* Generate `accuracy_bar_chart.png`

---

## Model Architecture

The CNN architecture includes:

* Multiple Convolutional Layers
* Batch Normalization
* ReLU Activation Functions
* Max Pooling Layers
* Fully Connected Layers
* Dropout Regularization

The architecture is designed to balance classification accuracy and computational efficiency for the CIFAR-10 dataset.

---

## Results

After training for **20 epochs**, the model achieved:

### Overall Accuracy

```text
83.23%
```

### Per-Class Performance

| Class | Accuracy |
| ----- | -------: |
| Car   |   96.10% |
| Ship  |   92.30% |
| Bird  |   70.60% |
| Cat   |   66.20% |

### Best Performing Classes

* Car (96.10%)
* Ship (92.30%)

### Most Challenging Classes

* Cat (66.20%)
* Bird (70.60%)

---

## Generated Outputs

### Training Loss Curve

```text
loss_curves.png
```

Displays the training loss progression over epochs.

### Accuracy Bar Chart

```text
accuracy_bar_chart.png
```

Displays classification accuracy for each CIFAR-10 class.

---

## Future Improvements

Potential improvements include:

* Deeper CNN architectures
* Residual connections (ResNet)
* Learning rate scheduling
* Additional data augmentation
* Hyperparameter tuning
* Transfer learning techniques

---

## License

This project is intended for educational and academic purposes.
