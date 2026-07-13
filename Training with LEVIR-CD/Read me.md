readme_content = """# Satellite Image Change Detection Using 2D-CNN (U-Net)

![Python](https://img.shields.io/badge/Python-3.8.20-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10.0-orange.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-98.09%25-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An end-to-end deep learning framework and interactive web prototype for detecting building changes in temporal satellite imagery using a customized 2D-CNN (U-Net) architecture trained on the **LEVIR-CD** dataset.

---

## 📌 Performance Overview

| Metric | Target | Result | Status |
| :--- | :---: | :---: | :---: |
| **Accuracy** | $>70.00\%$ | **98.09%** | Pass |
| **Precision** | - | **88.24%** | Pass |
| **Recall** | - | **72.03%** | Pass |
| **F1-Score** | - | **79.31%** | Pass |
| **IoU** | $>50.00\%$ | **65.72%** | Pass |

---

## 📂 Project Directory Structure

```text
.
├── Training/                               # Model Training & Processing
│   ├── LEVIR CD/                           # Raw dataset (train split)
│   ├── LEVIR CD Processed/                 # Preprocessed dataset
│   ├── logs/                               # Training history logs
│   ├── outputs/                            # Training outputs & plots
│   ├── Step 1 PreProcessing.py            # Data preprocessing script
│   ├── Step 2 Training.py                 # U-Net training script
│   └── Visualization.py                   # Plotting & metric analysis
│
├── Website steps/                          # Web Application
│   ├── Project/
│   │   ├── models/                         # Model weights directory
│   │   ├── templates/                      # Flask HTML templates
│   │   │   └── index.html
│   │   └── app.py                          # Flask web application
│   ├── Step 1.html                         # Step-by-step documentation
│   ├── Step 2.html
│   └── Step 3.html
│
└── requirements.txt                        # Python dependencies


🚀 How to Run
1. Model Training
To train the model from scratch:

Run Step 1 PreProcessing.py inside the Training folder to preprocess the dataset.

Run Step 2 Training.py to train the U-Net model and save the weights.

Bash
cd Training
python "Step 1 PreProcessing.py"
python "Step 2 Training.py"
2. Running the Web Application
To run the web interface:

Ensure the trained model weights (.h5 file) are placed inside Website steps/Project/models/.

Run app.py inside the Website steps/Project/ directory.

Bash
cd "Website steps/Project"
python app.py
Then open http://localhost:5000 in your web browser.

📜 License & Citation
This project is open-source under the MIT License.

Dataset: LEVIR-CD Dataset (Chen et al.)

