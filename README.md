Emotion-Based Smile Classifier Using ResNet-50


Project Overview
This project is a deep learning-based web application that classifies facial expressions into four categories: Real Smile, Fake Smile, Neutral, and Anxiety/Depression. It uses the power of ResNet-50 with transfer learning and provides an intuitive interface built using Streamlit for real-time image classification and emotional analysis.

Key Features
Developed using Python, PyTorch, OpenCV, and Streamlit.

Uses pre-trained ResNet-50 for accurate classification.

Allows users to upload images and instantly view predictions.

Displays dynamic graphs and visual analytics for better interpretation.

Supports mental health awareness through emotion detection.

📁 Project Structure
Emotion-BasedSmileClassifier/
├── scripts/ resnet50.ipynb if we train model then the model will be saved
├── test/ anxiety_depression, real_smiles, fake_smiles, neutral datasets
├── train/ anxiety_depression, real_smiles, fake_smiles, neutral datasets
├── requirements.txt → List of required Python libraries
├── app.py → Main Streamlit application
└── README.md → Project documentation

📥 Dataset (Not Included)
The training and testing datasets are not included in this repository due to large file size.
You can download the dataset separately (e.g., from Kaggle or Google Drive) and place it inside a /data folder for training.

🚀 How to Run the Project
Clone the Repository
git clone https://github.com/your-username/EmotionSmileClassifier.git
cd EmotionSmileClassifier

Install Required Libraries
pip install -r requirements.txt

Run the Web Application
streamlit run app.py

💡 Model Information
Architecture: ResNet-50 (pre-trained on ImageNet)

Training: Transfer Learning applied with custom emotion categories

Output Classes: Real Smile, Fake Smile, Neutral, Anxiety/Depression

Frozen Layers: All but the final classification layers

Loss Function: CrossEntropyLoss

Optimizer: Adam

Data Augmentation: Resize, Normalize, Horizontal Flip, Rotation

Hardware: Trained using GPU (CUDA) for fast performance

📊 Sample Output & Visualization
Users can upload an image and the app displays:

Predicted class (e.g., Fake Smile)

Prediction Confidence (e.g., 93.2%)

Dynamic Bar Chart of all class probabilities

Side-by-side layout for image and analysis

📌 System Requirements
Python 3.8+
PyTorch
torchvision
OpenCV
Streamlit
A GPU-enabled machine (optional but recommended)

Accuracy
![WhatsApp Image 2025-04-12 at 08 32 58_3cccb9c8](https://github.com/user-attachments/assets/7fef9502-7928-4d59-af19-7b479dcbf505)


Output
![WhatsApp Image 2025-04-11 at 23 29 45_db8f7a02](https://github.com/user-attachments/assets/81cf8f13-e80f-4938-aef9-2c3c070ad9d1)![WhatsApp Image 2025-04-11 at 23 29 45_5ad59289](https://github.com/user-attachments/assets/b0c72b81-13dc-4528-8d51-ac5c1f811f09)
