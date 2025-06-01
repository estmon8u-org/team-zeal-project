# Team Zeal - Streamlit Image Classifier UI

This Streamlit application provides a user interface to interact with the deployed image classification model.

## Features

- Upload an image (JPG, PNG).
- Sends the image to the backend API (deployed on GCP Cloud Run) for prediction.
- Displays the predicted class and confidence score.
- Shows a bar chart of class probabilities.

## Running Locally

1. Ensure the backend API is deployed and accessible.
2. Update the `API_URL` in `streamlit_app.py` to point to your deployed API, or set the `PREDICTION_API_URL` environment variable.
3. Install dependencies:

```bash
pip install -r requirements.txt
```
