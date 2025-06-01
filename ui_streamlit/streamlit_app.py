# ui_streamlit/streamlit_app.py
import os

from PIL import Image
import requests
import streamlit as st

# --- Configuration ---
# IMPORTANT: Replace with YOUR deployed Cloud Run API URL
API_URL = os.getenv(
    "PREDICTION_API_URL", "https://team-zeal-api-run-1004281831193.us-west2.run.app/predict/"
)


# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Team Zeal Image Classifier",
    page_icon="🖼️",
    layout="centered",
    initial_sidebar_state="auto",
)


# --- Helper Functions ---
def call_prediction_api(image_bytes):
    """Sends image to the Cloud Run API and returns the prediction."""
    if not API_URL or "YOUR_CLOUD_RUN_API_URL_HERE" in API_URL:
        st.error(
            "API URL is not configured. Please set the PREDICTION_API_URL environment variable or update the script."
        )
        return None

    files = {"file": ("uploaded_image.jpg", image_bytes, "image/jpeg")}
    try:
        response = requests.post(API_URL, files=files, timeout=30)  # 30 second timeout
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling prediction API: {e}")
        if hasattr(e, "response") and e.response is not None:
            try:
                st.error(f"API Response Content: {e.response.json()}")
            except ValueError:  # If response is not JSON
                st.error(f"API Response Content (Non-JSON): {e.response.text}")
        return None


# --- UI Layout ---
st.title("🖼️ Team Zeal Image Classifier")
st.markdown(
    "Upload an image and our model, deployed on GCP Cloud Run, will try to classify it! "
    "This UI is built with Streamlit and can be deployed to Hugging Face Spaces."
)
st.markdown("---")

# File Uploader
uploaded_file = st.file_uploader(
    "Choose an image to classify...",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG",
)

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([2, 3])  # Adjust column widths as needed
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        with st.spinner("Sending image to the model for classification..."):
            # Get bytes from uploaded file
            image_bytes = uploaded_file.getvalue()
            prediction_data = call_prediction_api(image_bytes)

        if prediction_data:
            st.subheader("🔍 Prediction Results:")

            predicted_class = prediction_data.get("predicted_class", "N/A")
            confidence = prediction_data.get("confidence", 0.0)

            st.success(f"**Predicted Class:** {predicted_class.capitalize()}")
            st.info(f"**Confidence:** {confidence:.2%}")

            all_probabilities = prediction_data.get("all_probabilities")
            if all_probabilities:
                st.markdown("---")
                st.subheader("📊 Class Probabilities:")

                # Create a dictionary suitable for st.bar_chart (label: value)
                # And sort by probability for better visualization
                sorted_probs = {
                    k: v
                    for k, v in sorted(
                        all_probabilities.items(), key=lambda item: item[1], reverse=True
                    )
                }

                # Prepare for st.bar_chart (it expects data that can be converted to a DataFrame)
                # We'll display top N probabilities or all if less than N
                top_n = 5
                chart_data_dict = {k: sorted_probs[k] for k in list(sorted_probs)[:top_n]}

                if chart_data_dict:
                    st.bar_chart(
                        chart_data_dict, height=300
                    )  # Use_container_width is True by default
                else:
                    st.write("Could not display probability chart.")
        else:
            st.error("Failed to get a prediction from the API.")
else:
    st.info("👆 Upload an image to get started!")

# --- Footer / About ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; font-size: small;'>
        <p>
            Team Zeal MLOps Project |
            <a href="https://github.com/estmon8u/team-zeal-project" target="_blank">GitHub Repository</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
