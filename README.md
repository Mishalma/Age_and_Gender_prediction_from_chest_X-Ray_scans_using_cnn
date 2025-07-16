Age and Gender Prediction from Chest X-ray Scans using CNN
This project leverages Convolutional Neural Networks (CNNs) to predict gender (classification) and age (regression) from chest X-ray images. The model uses grayscale images, preprocessed and normalized, and is trained on a dataset containing age and gender labels for X-ray scans.

📂 Dataset
Source: KaggleHub - Felipe Kitamura's SPR X-ray dataset

Structure:

train_age.csv: Contains filenames and corresponding ages.

train_gender.csv: Contains filenames and gender labels.

train/: Directory containing X-ray image files.

🔧 Preprocessing Steps
Grayscale Conversion

Image Inversion: Convert image to its negative (255 - pixel).

Resizing: Images resized to 128x128.

Normalization: Pixel values scaled to the range [0, 1].

Data Structuring: Images, age, and gender compiled into a DataFrame.

🧠 Model Architecture
The same custom CNN architecture is used for both tasks with modifications in the final layer.

🔹 Gender Prediction (Binary Classification)
Input: (128, 128, 1)

Convolutional Blocks: Several Conv2D + MaxPool + BatchNorm + ReLU layers

Output: Sigmoid layer

Loss: Binary Crossentropy

Optimizer: Adam

🔹 Age Prediction (Regression)
Output layer modified to Dense(1) with linear activation.

Loss: Likely Mean Squared Error (though not explicitly shown)

Note: Only gender training code was executed; age training appears initialized but incomplete in the notebook.

▶️ Training
python
Copy code
model_gender.fit(
    X, y, 
    steps_per_epoch=1000, 
    validation_split=0.2, 
    epochs=5
)
Training history is plotted using pandas + matplotlib.

💾 Model Saving
Trained gender model is saved as:

bash
Copy code
model_gender.h5
📊 Evaluation (To be Added)
Model evaluation (e.g., accuracy for gender, MAE for age) is not included but can be added using:

python
Copy code
model.evaluate(X_test, y_test)
📝 How to Run
Clone the repo.

Install dependencies:

bash
Copy code
pip install numpy pandas opencv-python pillow matplotlib tensorflow tqdm
Place the dataset in the specified folder.

Run the notebook or convert it to script for terminal usage.

🚀 Future Improvements
Train and evaluate the age prediction model.

Add Grad-CAM visualization for interpretability.

Use transfer learning with pre-trained CNNs like ResNet or DenseNet.

Apply data augmentation for robustness.

📜 License
This project is under the MIT License.
