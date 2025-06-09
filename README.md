# TB-Vision-X-AI
# Tuberculosis Detection Web Application

This is a full-stack web application for detecting tuberculosis from chest X-ray images using deep learning models. The application provides both DenseNet and LeNet model predictions along with GradCAM and LIME explanations for better interpretability.

## Features

- Upload chest X-ray images for TB detection
- Choose between DenseNet and LeNet models
- Get prediction results with confidence scores
- View GradCAM visualizations
- View LIME explanations
- Modern and responsive UI

## Project Structure

```
.
├── backend/
│   ├── app/
│   │   └── app.py
│   ├── models/
│   │   ├── tb_detector_model_densenet.h5
│   │   └── tb_detector_model_lenet.h5
│   └── requirements.txt
└── frontend/
    ├── public/
    │   └── index.html
    ├── src/
    │   ├── App.js
    │   └── index.js
    └── package.json
```

## Setup Instructions

### Backend Setup

1. Create a Python virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the required packages:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. Start the Flask server:
   ```bash
   python app/app.py
   ```
   The backend server will run on http://localhost:5000

### Frontend Setup

1. Install Node.js dependencies:
   ```bash
   cd frontend
   npm install
   ```

2. Start the React development server:
   ```bash
   npm start
   ```
   The frontend will run on http://localhost:3000

## Usage

1. Open your web browser and navigate to http://localhost:3000
2. Click "Upload X-Ray Image" to select a chest X-ray image
3. Choose the model you want to use (DenseNet or LeNet)
4. Click "Analyze Image" to get the prediction and explanations
5. View the results, including:
   - Diagnosis (Tuberculosis or Normal)
   - Confidence score
   - GradCAM visualization
   - LIME explanation

## Technologies Used

- Backend:
  - Flask
  - TensorFlow
  - LIME
  - OpenCV
  - NumPy
  - Pillow

- Frontend:
  - React
  - Material-UI
  - Axios
