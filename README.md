# ForgeryNet

# Overview

Signature forgery remains a major challenge in banking, legal, and corporate domains. This project leverages Deep Learning to detect genuine vs. forged handwritten signatures, improving the speed and accuracy of document authentication.

It includes experimented with multiple CNN-based architectures including AlexNet, VGG19, ResNet-50, and a Siamese Network — combined with data augmentation and GAN-generated synthetic data to boost model robustness.

The final deliverable is a Streamlit web application that allows users to upload signature images and get real-time predictions.

# Objectives

1. Develop a robust deep learning model to classify signatures as genuine or forged.
2. Use data augmentation and GANs to improve generalization and handle limited data.
3. Compare multiple CNN architectures for performance on signature verification tasks.
4. Implement Siamese ResNet-50 for similarity learning between signature pairs.
5. Provide an easy-to-use web app for interactive inference.

# Dataset

1. Source: CEDAR Signature Dataset (Kaggle)
2. Total Images: 2,642 (1,321 Genuine + 1,321 Forged)
3. Subjects: 55 individuals, 24 genuine + 24 forged signatures each
4. Image Size: 256×256 (Grayscale)
5. Split: 70% Train, 15% Validation, 15% Test

# Preprocessing & Augmentation

1. Grayscale conversion
2. Random horizontal flips, rotations (±20°), brightness jitter
3. Normalization
4. GAN-based synthetic generation to increase dataset diversity

# Methodology

1. Baseline CNN
   - 3 convolutional layers (16 → 32 → 64 feature maps)
   - ReLU activations + MaxPooling
   - Fully connected layers: 512 → 2 (genuine/forged)
   - Perfect train accuracy observed, but risk of overfitting without regularization.

2. Optimized CNN
   - Added dropout layers (0.5) for regularization
   - Applied data augmentation to avoid overfitting
   - Training stabilized, validation accuracy improved
   
4. Advanced Architectures
   - AlexNet: Training accuracy 80.24%, Validation 76.89%, Good Baseline
   - ResNet-50: Validation accuracy 93.75%, Test accuracy 95.83%, overfitting tendency
   - Validation accuracy peaked at 95.27%, overfitting tendency
   - Siamese Network: Used contrastive loss to learn similarity between signature pairs. Effective but more variable, 61% to 95% accuracy depending on the base models used, on test set due to high sensitivity to pair generation.
   
6. GAN-based Data Augmentation
   - Implemented a Generative Adversarial Network to synthesize new signature images.
   - Generator: Fully connected → ReLU → TanH, outputs 128×128 images.
   - Discriminator: LeakyReLU → Sigmoid classifier (real vs fake).
   - Trained with Binary Cross Entropy and Adam optimizer.
   - Synthetic images were added back into the dataset to improve model robustness.

# Evaluation

- Accuracy
- Precision / Recall / F1 Score
- Confusion Matrix
- ROC-AUC

| Model             | Accuracy | Precision | Recall | F1 Score |
| ----------------- | -------- | --------- | ------ | -------- |
| Baseline CNN      | 78.09%   | 0.79      | 0.78   | 0.78     |
| AlexNet           | 80.24%   | 0.81      | 0.77   | 0.79     |
| VGG19             | 95.27%   | 0.95      | 0.95   | 0.95     |
| ResNet-50         | 95.83%   | 0.96      | 0.96   | 0.96     |
| Siamese ResNet-50 | 95.19%%   | 0.94     | 0.95  | 0.95     |

# Running the Application
  1. Install Dependencies
     Dependencies include:
      - streamlit
      - torch, torchvision
      - Pillow
      - numpy, matplotlib
    
  <pre>
    python -m venv .venv
    source .venv/bin/activate      # Windows: .venv\Scripts\activate
    pip install --upgrade pip
    pip install -r requirements.txt
  </pre>
     
  2. Download Dataset
     Download the CEDAR Dataset and unzip it into

     <pre>
       data/
        full_org/
        full_forg/
      </pre>
      
  3. Run the App
     Navigate to the app folder and launch Streamlit:
     <pre> 
       cd src/App
       streamlit run app.py
      </pre>

      - Upload a signature image (genuine or forged)
      - The app loads the pre-trained model res_network.pth
      - Returns prediction: Genuine / Forged

# Key Findings
- ResNet-50 architecture with >95% accuracy with a risk of overfitting
- VGG19 achieved strong results but was computationally heavier
- Siamese Networks are promising for pairwise verification but need improved pair generation and contrastive loss tuning
- GAN augmentation improved diversity, especially for CNN baselines
- Proper data preprocessing & regularization significantly boosted generalization

# References
- CEDAR Signature Dataset – Kaggle
- He et al. – Deep Residual Learning for Image Recognition (ResNet)
- Simonyan & Zisserman – Very Deep Convolutional Networks (VGG)
- Goodfellow et al. – Generative Adversarial Nets (2014)
- PyTorch Documentation – CNN & Transfer Learning Tutorials
     
    
