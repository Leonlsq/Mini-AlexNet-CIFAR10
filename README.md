# Mini-AlexNet for CIFAR-10 Classification

This repository contains a modular implementation of a "Mini-AlexNet," specifically re-engineered to handle low-resolution images from the CIFAR-10 dataset. This project serves as a practical exploration of Convolutional Neural Network (CNN) architectures and spatial dimension dynamics.

## Motivation & Architecture Adaptation

The original AlexNet (2012) was designed for ImageNet images ($224 \times 224$). Applying the original architecture directly to CIFAR-10 ($32 \times 32$) results in rapid information loss due to large kernel sizes and strides.

### The Mathematical Challenge
In the original AlexNet, the first layer uses an $11 \times 11$ kernel with a stride of 4. For a $32 \times 32$ input:
$$O = \lfloor \frac{32 - 11 + 2(0)}{4} \rfloor + 1 = 6$$
The spatial dimension drops from 32 to 6 in a single layer, which is too aggressive for small images.

### My Solution: Mini-AlexNet
To preserve spatial information while maintaining depth, I adapted the architecture:
- **First Convolution:** Reduced kernel size to $3 \times 3$, stride to $1$, and added padding of $1$. This keeps the output at $32 \times 32$.
- **Pooling Strategy:** Used three $2 \times 2$ Max-Pooling layers to gradually reduce dimensions: $32 \rightarrow 16 \rightarrow 8 \rightarrow 4$.
- **Classifier:** Downsized fully connected layers to $1024$ neurons to prevent overfitting on a smaller dataset and reduce computational overhead.

##  Tech Stack
- **Framework:** PyTorch
- **Environment:** Python 3.x
- **Libraries:** Torchvision (Data), Matplotlib (Visualization), NumPy

## Project Structure
- `model.py`: Definition of the `MiniAlexNet` class.
- `data_loader.py`: Automated pipeline for CIFAR-10 download and preprocessing.
- `train.py`: Training loop with real-time loss tracking and validation.
- `.gitignore`: Configured to exclude heavy datasets and local caches.

## Performance & Results
*(Optional: Insert your training loss curve or accuracy here after running the training)*

- **Target Accuracy:** ~70% (on CIFAR-10 test set)
- **Training Device:** MPS (Apple Silicon) / CUDA / CPU

## How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/Leonlsq/Mini-AlexNet-CIFAR10.git
   cd MiniAlexNet
   ```
2.Install dependencies:
   ```bash
   pip install torch torchvision matplotlib numpy
   ```

3.Start training:
   ```bash
   python train.py
   ```


## About Me
I am a 2nd-semester Informatics student at Technical University of Munich (TUM). 
I am passionate about deep learning and actively self-teaching milestone architectures in Computer Vision and NLP.
