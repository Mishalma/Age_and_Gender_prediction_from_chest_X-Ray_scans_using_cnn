# Age and Gender Prediction from Chest X-Ray Scans

A machine learning web application that predicts age and gender from chest X-ray images using Convolutional Neural Networks (CNN). The application provides a user-friendly web interface for uploading X-ray images and receiving instant predictions.

## Features

- **Age Prediction**: Estimates the age of patients from chest X-ray scans
- **Gender Classification**: Determines whether the patient is male or female
- **Web Interface**: Clean, responsive web UI with drag-and-drop file upload
- **Real-time Processing**: Instant predictions after image upload
- **Image Preprocessing**: Automatic image normalization and resizing

## Technology Stack

- **Backend**: Flask (Python web framework)
- **Machine Learning**: TensorFlow/Keras
- **Image Processing**: OpenCV, PIL
- **Frontend**: HTML5, CSS3, JavaScript with GSAP animations
- **Styling**: Tailwind CSS

## Project Structure

```
├── app.py                 # Flask web application
├── model.py              # ML model loading and prediction functions
├── model_age.h5          # Pre-trained age prediction model
├── model_gender.h5       # Pre-trained gender classification model
├── requirements.txt      # Python dependencies
├── static/
│   └── input.png        # Uploaded images storage
├── templates/
│   └── index.html       # Web interface template
└── Age_and_Gender_prediction_from_chest_X-Ray_scans_using_cnn.ipynb
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd lungsbased_age_gender_prediction
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # source venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install flask tensorflow opencv-python pillow numpy matplotlib
   ```

## Usage

1. **Start the application**
   ```bash
   python app.py
   ```

2. **Access the web interface**
   - Open your browser and navigate to `http://localhost:5000`

3. **Upload and analyze**
   - Click "Choose File" to select a chest X-ray image (PNG format)
   - Click "Analyze Image" to get predictions
   - View the predicted age and gender results

## Model Information

The application uses two separate CNN models:

- **Age Model** (`model_age.h5`): Regression model for age prediction
- **Gender Model** (`model_gender.h5`): Binary classification model for gender prediction

### Input Requirements
- **Image Format**: PNG files
- **Processing**: Images are automatically converted to grayscale, inverted (negative), and resized to 128x128 pixels
- **Normalization**: Pixel values are normalized to the range [0, 1]

## API Endpoints

- `GET /` - Renders the main interface
- `POST /` - Handles file upload and returns predictions

## Configuration

Update the following paths in the code to match your environment:

```python
# In app.py
UPLOAD_FOLDER = 'path/to/your/project/static'

# In model.py  
base = 'path/to/your/project'
```

## Development

The project includes a Jupyter notebook (`Age_and_Gender_prediction_from_chest_X-Ray_scans_using_cnn.ipynb`) containing the model training and development process.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).

## Disclaimer

This application is for educational and research purposes only. It should not be used for actual medical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical advice.
