# Grape Disease Classifier - Indian Context

This project provides a web-based application for classifying grape diseases commonly found in Indian vineyards. It uses deep learning to identify four conditions:
- Healthy grapes
- Leaf Blight (Isariopsis Leaf Spot)
- Esca (Black Measles)
- Black Rot

## Features

- Upload grape leaf images for disease classification
- Get detailed information about the detected disease
- Receive Indian-specific treatment recommendations
- Prevention tips for maintaining healthy grapevines

## Setup Instructions

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python train_model.py
```

3. Run the web application:
```bash
python app.py
```

4. Open your web browser and navigate to `http://localhost:5000`

## Usage

1. Click on the upload area or drag and drop an image of a grape leaf
2. Wait for the analysis to complete
3. View the results, including:
   - Disease name
   - Confidence level
   - Description of the disease
   - Treatment recommendations
   - Prevention tips

## Technical Details

- The model uses MobileNetV2 as the base architecture
- Image preprocessing includes resizing to 224x224 pixels
- Data augmentation is applied during training
- The web interface is built with Flask and Bootstrap

## Note

This application is specifically tailored for Indian grape cultivation conditions and provides region-specific treatment and prevention recommendations. 