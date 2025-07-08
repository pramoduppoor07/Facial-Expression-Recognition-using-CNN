#  Facial Expression Recognition (FER-2013) using CNN

This project uses deep learning techniques to classify human facial expressions into emotion categories like Happy, Sad, Angry, and more. The model is trained on the FER-2013 dataset using Convolutional Neural Networks (CNNs) and can predict emotions from grayscale facial images.

---

##  Project Overview

This is an image-based classification problem focused on identifying emotions from facial images using a CNN model.

- Problem Type: Multi-class Image Classification  
- Model Used: Convolutional Neural Network (CNN)  
- Dataset: FER-2013 (from Kaggle)  
- Output Labels:
  - 0 = Angry
  - 1 = Disgust
  - 2 = Fear
  - 3 = Happy
  - 4 = Sad
  - 5 = Surprise
  - 6 = Neutral

---

##  Dataset Description

- Images: 48×48 grayscale facial images  
- Total Samples: ~35,000  
- Dataset Split:
  - train/
  - test/  
- Each image is stored under a folder named after its emotion label.

---

##  Tech Stack & Libraries Used

- Python  
- NumPy, Pandas – Data processing  
- OpenCV – Image loading and preprocessing  
- TensorFlow / Keras – CNN model building and training  
- Matplotlib – Visualization  
- Scikit-learn – Data splitting and metrics

---

##  Model Architecture

A CNN with three convolutional blocks followed by dense layers:

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])
```
---
##  Accuracy Results

| Dataset       | Accuracy |
|---------------|----------|
| Training Set  | ~56%     |
| Validation    | ~54%     |
| Test Set      | ~55%     |


---

##  Prediction Example

After training, the model can predict the emotion from a given image:

```python
predict_image('/content/FER-2013/test/happy/PrivateTest_10613684.jpg')
# Output: happy
```
---
##  How to Run This Project

1. Clone the repository:

```bash
git clone https://github.com/pramoduppoor07/Facial-Expression-Recognition-using-CNN.git
cd Facial-Expression-Recognition-using-CNN
```

2. Run the Jupyter Notebook:
```bash
jupyter notebook
```
---

##  Future Improvements

-  Add Data Augmentation using `ImageDataGenerator`
-  Try transfer learning with MobileNet or EfficientNet
-  Deploy the model using Streamlit or Flask
-  Add real-time facial expression detection using OpenCV
-  Add confusion matrix and classification report

---

##  Acknowledgements

- FER-2013 Dataset – [Kaggle: Facial Expression Recognition Challenge](https://www.kaggle.com/datasets/msambare/fer2013)
- Keras and TensorFlow teams
- OpenCV community
