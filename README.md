# 🍕 Pizza Classification Using YOLO11n

A deep learning image classification project using **YOLO11n-Classification (YOLO11n-cls)** to classify pizza images into four different categories.

## 👩‍💻 Project Submitted By

- **Sunkara Harshitha**
- **P. Charvitha**

---

## 📌 Project Overview

This project implements an image classification system using the **Ultralytics YOLO11n classification model**.

The trained model classifies pizza images into four categories:

- Bianca
- Hawaiian
- Manager Choice
- Pepporoni

The project covers the complete workflow:

**Dataset → Dataset Organization → Train/Validation/Test Split → Model Selection → Training → Validation → Testing → Evaluation → New Image Prediction**

---

## 🎯 Objective

The main objective of this project is to develop a deep learning model that can automatically identify the category of a pizza image.

The model is evaluated using:

- Training Loss
- Validation Loss
- Top-1 Accuracy
- Top-5 Accuracy
- Confusion Matrix
- Normalized Confusion Matrix
- Independent New-Image Prediction

---

## 🧠 Model Used

### YOLO11n Classification

The project uses the **YOLO11n-cls** model for image classification.

YOLO11n is the nano-sized YOLO11 model and was selected as a lightweight model suitable for efficient image classification.

| Parameter | Value |
|---|---|
| Model | YOLO11n-cls |
| Task | Image Classification |
| Input Size | 224 × 224 |
| Ultralytics | 8.4.121 |
| Python | 3.12.10 |
| PyTorch | 2.13.0 |

The pretrained model used for training is:

`yolo11n-cls.pt`

The model weight file is excluded from GitHub using `.gitignore`.

---

# 📂 Dataset

The dataset contains four pizza classes.

| Class | Images |
|---|---:|
| Bianca | 104 |
| Pepporoni | 100 |
| Hawaiian | 100 |
| Manager Choice | 100 |
| **Total** | **404** |

## Dataset Split

The complete dataset was divided into training, validation, and testing sets.

| Split | Images |
|---|---:|
| Training | 282 |
| Validation | 80 |
| Testing | 42 |
| **Total** | **404** |

### Training Distribution

| Class | Images |
|---|---:|
| Bianca | 72 |
| Pepporoni | 70 |
| Hawaiian | 70 |
| Manager Choice | 70 |
| **Total** | **282** |

### Validation Distribution

| Class | Images |
|---|---:|
| Bianca | 20 |
| Pepporoni | 20 |
| Hawaiian | 20 |
| Manager Choice | 20 |
| **Total** | **80** |

### Test Distribution

| Class | Images |
|---|---:|
| Bianca | 12 |
| Pepporoni | 10 |
| Hawaiian | 10 |
| Manager Choice | 10 |
| **Total** | **42** |

The training, validation, and test images are kept separate to prevent test images from being used during training.

---

# 🔄 Project Workflow

```text
                    Pizza Dataset
                         │
                         ▼
                Dataset Organization
                         │
                         ▼
            Train / Validation / Test Split
                         │
                         ▼
               YOLO11n Classification
                         │
                         ▼
                      Training
                         │
                         ▼
                    Validation
                         │
                         ▼
                       Testing
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
      Confusion Matrix       Accuracy Metrics
              │
              ▼
       New Image Prediction

       # ⚙️ Environment Setup

## Python Version

Python 3.12.10

## Main Libraries

- Ultralytics 8.4.121
- PyTorch 2.13.0
- OpenCV
- Pillow
- NumPy
- Matplotlib
- PyYAML

---

# 🚀 Installation

## 1. Clone the Repository

    git clone https://github.com/Sunkara-Harshitha09/CVAT_Image_Annotation_Classification.git

## 2. Open the Project

    cd CVAT_Image_Annotation_Classification

## 3. Create a Virtual Environment

    python -m venv .venv

## 4. Activate the Virtual Environment

Windows PowerShell:

    .\.venv\Scripts\Activate.ps1

## 5. Install Dependencies

    pip install -r requirements.txt

## 6. Verify the Installation

    yolo version

You can also verify the installation using Python:

    python -c "from ultralytics import YOLO; print('Ultralytics YOLO: OK')"

---

# 🗂️ Dataset Structure

The classification dataset follows a folder-based classification structure:

    dataset/
    │
    ├── train/
    │   ├── bianca/
    │   ├── hawaiian/
    │   ├── manager_choice/
    │   └── pepporoni/
    │
    ├── val/
    │   ├── bianca/
    │   ├── hawaiian/
    │   ├── manager_choice/
    │   └── pepporoni/
    │
    └── test/
        ├── bianca/
        ├── hawaiian/
        ├── manager_choice/
        └── pepporoni/

For image classification, the folder name represents the class label.

The dataset is not included in this GitHub repository because the image files are large.

---

# 🛠️ Dataset Splitting

The project contains:

    split_dataset.py

This script divides the original dataset into:

- Training set
- Validation set
- Test set

Final dataset split:

    Training:    282 images
    Validation:   80 images
    Testing:      42 images
    Total:       404 images

The test images are kept separate from the training images so that they are not used during model training.

---

# 🏋️ Model Training

The YOLO11n classification model is trained using the Ultralytics YOLO framework.

Example training code:

    from ultralytics import YOLO

    model = YOLO("yolo11n-cls.pt")

    model.train(
        data="dataset",
        epochs=20,
        imgsz=224,
        batch=16,
        project="runs/classify",
        name="food_classifier"
    )

---

# ⚙️ Important Training Parameters

| Parameter | Meaning |
|---|---|
| `data` | Location of the classification dataset |
| `epochs` | Number of complete passes through the training dataset |
| `batch` | Number of images processed together before updating model weights |
| `imgsz` | Input image size |
| `project` | Directory where training results are stored |
| `name` | Name of the training experiment |

## Epoch

One epoch means that the model has processed the complete training dataset once.

For example, if the training configuration uses:

    epochs = 20

the model goes through the training dataset 20 times.

## Batch Size

Batch size determines how many images are processed together before the model updates its weights.

For example:

    batch = 16

means 16 images are processed in one batch.

## Image Size

The input images are resized to:

    224 × 224

before being passed to the model.

---

# 📈 Training and Validation Results

YOLO generates several graphs during training to help understand how the model learns.

The main graphs include:

- Training Loss
- Validation Loss
- Top-1 Accuracy
- Top-5 Accuracy

These graphs help determine whether the model is learning correctly and whether it generalizes to unseen images.

---

# 📉 Training Loss

Training loss represents the error made by the model on the training dataset.

As training progresses, the model attempts to reduce this error.

    Training Loss ↓
          ↓
    Model learns training patterns

A decreasing training loss generally indicates that the model is learning from the training data.

However, training loss alone does not prove that the model will perform well on unseen images.

---

# 📉 Validation Loss

Validation loss represents the error calculated on the validation dataset.

The validation dataset is separate from the training images and helps measure how well the model generalizes.

    Validation Loss ↓
            ↓
    Better generalization

If training loss continues to decrease while validation loss starts increasing, it can indicate overfitting.

---

# 📊 Top-1 Accuracy

Top-1 accuracy checks whether the model's highest-confidence prediction is the correct class.

Example:

    Bianca          0.70
    Manager Choice  0.27
    Hawaiian        0.02
    Pepporoni       0.01

If the actual image is Bianca, then the Top-1 prediction is Bianca and the prediction is correct.

For this project, Top-1 accuracy is the most meaningful accuracy metric because the task contains four classes.

---

# 📊 Top-5 Accuracy

Top-5 accuracy checks whether the correct class appears within the model's five highest-ranked predictions.

It does NOT mean that the model was trained five times.

This project has only four classes:

    Bianca
    Hawaiian
    Manager Choice
    Pepporoni

Therefore, the correct class will always be within the top four predictions.

As a result, Top-5 accuracy is 100% and is not particularly useful for evaluating this four-class problem.

For this project:

**Top-1 Accuracy is more meaningful than Top-5 Accuracy.**

---

# 📊 Confusion Matrix

The confusion matrix provides a class-by-class view of the model's predictions.

It compares:

    Actual Class
         vs
    Predicted Class

The rows represent the actual classes and the columns represent the predicted classes.

## Diagonal Values

The diagonal represents correct predictions.

Example:

    Actual Bianca → Predicted Bianca

This means the model correctly classified the image as Bianca.

## Off-Diagonal Values

The off-diagonal cells represent incorrect predictions.

Example:

    Actual Hawaiian → Predicted Pepporoni

This means the model confused Hawaiian pizza with Pepporoni pizza.

The confusion matrix is useful for identifying which classes are easy or difficult for the model to distinguish.

---

# 📊 Normalized Confusion Matrix

The normalized confusion matrix represents the confusion matrix values as proportions rather than raw image counts.

Examples:

    1.00 = 100%
    0.95 = 95%
    0.05 = 5%

A value of 1.00 on the diagonal means that 100% of the images belonging to that actual class were correctly classified.

A value such as 0.10 in an off-diagonal cell means that approximately 10% of images were classified as another class.

Normalization makes it easier to compare performance between classes.

---

# 🧪 Model Evaluation

The trained model is evaluated using the test dataset.

The test dataset contains:

    42 images

distributed across the four classes:

    Bianca          12
    Pepporoni       10
    Hawaiian        10
    Manager Choice  10

The test images were not used for training.

This allows the test set to provide a more realistic estimate of model performance on unseen images.

---

# 🔍 New Image Prediction

After training and evaluation, the trained model was tested on additional images that were not part of the original training dataset.

Nine additional images were used with different image-quality conditions:

- Bright image
- Blurred image
- Low-contrast image
- Noisy image
- Good-quality image
- Glare
- Low-resolution image
- Combined degradation
- Occlusion

The prediction command used was:

    yolo classify predict model="runs\classify\runs\classify\food_classifier\weights\best.pt" source="new_images" imgsz=224

---

# 🔎 Example Prediction

For the image:

    image_07_good.jpg

the model produced:

    Bianca          0.70
    Manager Choice  0.27
    Hawaiian        0.02
    Pepporoni       0.01

The highest-confidence prediction was:

    Bianca

with a confidence of approximately:

    70%

---

# 🧠 Understanding Confidence Scores

The confidence score represents how strongly the model favors a particular class for an input image.

For example:

    Bianca          0.70
    Manager Choice  0.27
    Hawaiian        0.02
    Pepporoni       0.01

The model's highest score is Bianca.

Therefore:

    Predicted Class = Bianca
    Confidence = 70%

The other values show the model's relative scores for the remaining classes.

A high confidence score does not automatically guarantee that the prediction is correct.

---

# ⚠️ Important Note About New Image Testing

The nine additional images were used as an independent inference and robustness check.

They should not be treated as an official accuracy evaluation because verified ground-truth labels were not assigned to these images.

Therefore:

    New Image Prediction ≠ Accuracy Evaluation

The formal evaluation should be based on the labeled test dataset and evaluation metrics.

---

# 🧩 Model Prediction Flow

The trained model processes an input image through the following general workflow:

    Input Image
         ↓
    Image Preprocessing
         ↓
    YOLO11n Classification Model
         ↓
    Feature Extraction
         ↓
    Classification Head
         ↓
    Class Scores
         ↓
    Highest-Confidence Class
         ↓
    Final Prediction

---

# 📁 Project Structure

    CVAT_Image_Annotation_Classification/
    │
    ├── dataset/                    # Ignored by Git
    │   ├── train/
    │   ├── val/
    │   └── test/
    │
    ├── new_images/                 # Ignored by Git
    │
    ├── runs/                       # Ignored by Git
    │
    ├── split_dataset.py            # Dataset splitting script
    ├── requirements.txt            # Python dependencies
    ├── README.md                   # Project documentation
    ├── .gitignore                  # Git ignore configuration
    │
    └── yolo11n-cls.pt              # Ignored pretrained model

---

# 🔐 GitHub and Large Files

The following files and folders are intentionally excluded from GitHub:

    dataset/
    new_images/
    runs/
    *.pt
    *.pth
    *.onnx
    .venv/
    *.zip

This keeps the repository lightweight and prevents large datasets, generated training results, virtual environments, and model files from unnecessarily increasing repository size.

---

# 🧪 Technologies Used

- Python
- Ultralytics YOLO11
- YOLO11n Classification
- PyTorch
- OpenCV
- Pillow
- NumPy
- Matplotlib
- PyYAML
- Git
- GitHub
- Visual Studio Code

---

# 📚 Key Concepts Learned

This project covers:

- Image Classification
- Dataset Organization
- Train/Validation/Test Splitting
- Deep Learning
- Transfer Learning
- YOLO11 Classification
- YOLO11 Architecture
- Model Training
- Epochs
- Batch Size
- Learning Rate
- Image Size
- Training Loss
- Validation Loss
- Top-1 Accuracy
- Top-5 Accuracy
- Confusion Matrix
- Normalized Confusion Matrix
- Model Inference
- Confidence Scores
- Model Evaluation
- Git
- GitHub
- Virtual Environments

---

# 🎯 Project Outcome

The project demonstrates an end-to-end pizza image classification workflow using YOLO11n.

The complete workflow covers:

    Dataset Preparation
            ↓
    Dataset Splitting
            ↓
    Model Selection
            ↓
    YOLO11n Training
            ↓
    Validation
            ↓
    Testing
            ↓
    Performance Evaluation
            ↓
    Confusion Matrix Analysis
            ↓
    New Image Inference

The trained classifier learned visual patterns from the four pizza categories and produced class predictions with confidence scores for new images.

---

# 💡 Key Observations

The project demonstrates several important observations:

1. The model learns class-specific visual patterns during training.
2. Training and validation metrics help monitor the learning process.
3. The confusion matrix helps identify class-level classification errors.
4. Top-1 accuracy is more meaningful than Top-5 accuracy for a four-class problem.
5. Confidence scores indicate the model's preference among the available classes.
6. Image-quality changes such as blur, noise, glare, low contrast, and occlusion can affect predictions.
7. Independent images can be used to observe model behavior, but they should not be used to claim accuracy without verified ground-truth labels.

---

---

# 🔬 YOLO11n vs YOLO26n Model Comparison

To evaluate the effectiveness of the selected YOLO classification model, a comparative experiment was performed using **YOLO11n-cls** and **YOLO26n-cls**.

Both models were trained and evaluated under the same experimental conditions to ensure a fair comparison.

## Experimental Setup

| Parameter | Value |
|---|---|
| Task | Pizza Image Classification |
| Dataset | 404 images |
| Number of Classes | 4 |
| Classes | Bianca, Hawaiian, Manager Choice, Pepporoni |
| Training Images | 282 |
| Validation Images | 80 |
| Test Images | 42 |
| Image Size | 224 × 224 |
| Epochs | 20 |
| Batch Size | 16 |
| Device | CPU |
| Framework | Ultralytics YOLO |
| Python | 3.12 |

## Models Compared

### YOLO11n-cls

The original project model used for pizza classification.

### YOLO26n-cls

A newer YOLO classification model evaluated under the same conditions as YOLO11n-cls.

The purpose of the experiment was to determine whether the newer model provides better classification performance on the same pizza dataset.

---

# 📊 Comparison Results

The models were evaluated using the independent **42-image test set**.

| Metric | YOLO11n-cls | YOLO26n-cls |
|---|---:|---:|
| Test Top-1 Accuracy | **92.86%** | **92.86%** |
| Test Top-5 Accuracy | **100%** | **100%** |
| Training Time | **801.85 s** | **835.57 s** |
| Parameters | **1,531,148** | **1,531,148** |

## Top-1 Accuracy

**Top-1 accuracy** measures whether the model's highest-confidence prediction is the correct class.

Both models achieved:

**92.86% Top-1 Accuracy**

With 42 test images, this corresponds to:

- Correct predictions: 39
- Incorrect predictions: 3

Therefore:

```text
39 / 42 × 100 = 92.86%

# 🔮 Future Improvements

Possible improvements include:

1. Increase the size of the dataset.
2. Collect more images for each pizza category.
3. Improve class balance.
4. Add more pizza categories.
5. Collect images under different lighting conditions.
6. Apply systematic data augmentation.
7. Compare YOLO11n with larger YOLO classification models.
8. Perform hyperparameter tuning.
9. Experiment with different learning rates and batch sizes.
10. Evaluate the model using a larger labeled external test set.
11. Improve robustness to blur, glare, noise, occlusion, and low-resolution images.
12. Deploy the trained classifier as a web or mobile application.

---

# 👥 Contributors

**Sunkara Harshitha**

**P. Charvitha**

---

# 📌 GitHub Repository

https://github.com/Sunkara-Harshitha09/CVAT_Image_Annotation_Classification

---

