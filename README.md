# AI_Gen - Face Generator with GANs (DCGAN + CelebA)

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) to generate realistic human face images using the CelebA dataset.

## 🚀 Features

- Uses **PyTorch** to build and train the GAN architecture
- Generates 64x64 pixel face images
- Training progress shown live with loss tracking
- Optionally displays generated images after training
- Compatible with CUDA (NVIDIA GPUs) or CPU
- Suitable for AMD GPUs via ROCm (experimental)


## 📦 Dependencies

```bash
pip install torch torchvision matplotlib pillow
