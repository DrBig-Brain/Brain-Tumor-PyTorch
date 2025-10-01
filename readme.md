# Brain Tumor Detection Using PyTorch

A deep learning project that uses PyTorch to detect and classify brain tumors from MRI scans.

## Overview

This project implements a Convolutional Neural Network (CNN) based on VGG16 architecture to analyze brain MRI images and classify them into four categories: glioma, meningioma, no tumor, and pituitary tumors.

## Requirements

```txt
torch>=2.8.0
torchvision
numpy
pillow
matplotlib
streamlit
```

## Installation

```bash
git clone https://github.com/DrBig-Brain/Brain-Tumor-PyTorch.git
cd Brain-Tumor-PyTorch
pip install -r requirements.txt
```

## Dataset

The model is trained on brain MRI scan images. The dataset should be organized in the following structure:

```
data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── normal/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── normal/
    └── pituitary/
```

## Usage

1. To train and evaluate the model:
```bash
jupyter notebook brainTumor.ipynb
```

2. To run the web interface:
```bash
streamlit run app.py
```

## Model Architecture

The model uses a modified VGG16 architecture with:
- Pre-trained convolutional layers
- Custom classifier layers:
  - Linear(25088, 64)
  - ReLU
  - Linear(64, 128) 
  - ReLU
  - Linear(128, 4)

## Results

The model achieves:
- Training accuracy: ~95%
- Validation accuracy: ~93%
- Test accuracy: ~92%

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.