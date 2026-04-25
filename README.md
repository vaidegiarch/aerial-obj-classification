# aerial-obj-classification
drone or bird

Project Title--Aerial Object Classification & Detection

Skills take away From This Project

Deep Learning
Computer Vision
Image Classification & Object Detection
Python
TensorFlow/Keras or PyTorch
Data Preprocessing & Augmentation



Model Evaluation
Streamlit Deployment
Domain
Aerial Surveillance, Wildlife Monitoring, Security & Defense Applications

📌 Problem Statement
This project aims to develop a deep learning-based solution that can classify aerial images into two categories — Bird or Drone 
— and optionally perform object detection to locate and label these objects in real-world scenes.
The solution will help in security surveillance, wildlife protection, and airspace safety where accurate identification between drones and birds is critical. 
The project involves building a Custom CNN classification model, leveraging transfer learning, and optionally implementing YOLOv8 for real-time object detection. The final solution will be deployed using Streamlit for interactive use.

📌 Real-Time Business Use Cases
Wildlife Protection
Detect birds near wind farms or airports to prevent accidents.
Security & Defense Surveillance
Identify drones in restricted airspace for timely alerts.
Airport Bird-Strike Prevention
Monitor runway zones for bird activity.
Environmental Research
Track bird populations using aerial footage without misclassification.

📌 Project Workflow
1. Understand the Dataset
Inspect dataset folder structure
Check number of images per class
Identify class imbalance
Visualize sample images

2. Data Preprocessing
Normalize pixel values to [0, 1]
Resize images to a fixed size (224×224 for classification)
For TensorFlow-based transfer learning models, use the model-specific preprocess_input function.
For PyTorch-based pretrained models, apply ImageNet normalization using torchvision.transforms.Normalize(mean, std) as per the model’s training configuration.

3. Data Augmentation
Apply transformations: rotation, flipping, zoom, brightness, cropping

4. Model Building (Classification)
Custom CNN: Conv layers, pooling, dropout, batch normalization, dense output layer


Transfer Learning: Load models like ResNet50, MobileNet, EfficientNetB0 and fine-tune

5. Model Training
Train both models
Use EarlyStopping & ModelCheckpoint
Track metrics: Accuracy, Precision, Recall, F1-score

6. Model Evaluation
Evaluate test results with confusion matrix & classification report
Plot accuracy/loss graphs

7. Model Comparison
Compare accuracy, training time, and generalization performance
Save the best performing model for Streamlit deployment
