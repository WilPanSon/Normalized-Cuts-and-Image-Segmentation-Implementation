# Normalized Cuts Image Segmentation

An implementation of the **Shi & Malik (2000) - "Normalized Cuts and Image Segmentation"** algorithm in Python, using `scikit-image`, `scipy`, and `numpy`.  
This script performs **2-way normalized cut segmentation** on an input image with automatic parameter scaling for robust results.

---

## 📜 Overview

Normalized cuts is a graph-theoretic approach to image segmentation that:
- Models an image as a weighted graph of pixels
- Uses the **Fiedler vector** (second smallest eigenvector of the normalized Laplacian) to partition the image
- Balances both **intra-group similarity** and **inter-group dissimilarity**

This project:
- Builds an **affinity matrix** using KD-trees for efficiency
- Automatically scales parameters based on image size and intensity
- Performs a **2-way segmentation** using median thresholding on the Fiedler vector

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/normalized-cuts.git
cd normalized-cuts
```
### 2. Install dependencies
pip install numpy scipy scikit-image matplotlib

### 3. Run Script
python test.py --image images/your_image.jpg --max_side 200
