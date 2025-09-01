import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# ----------------------------
# Load Trained Model
# ----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("DIGIT_recog.h5")

model = load_model()

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Digit Recognition", page_icon="✍️")
st.title("✍️ Handwritten Digit Recognition")
st.write("Upload a digit image or draw one below, and the model will predict it!")

# ----------------------------
# Upload Section
# ----------------------------
st.subheader("📤 Upload an Image")
uploaded_file = st.file_uploader("Upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # convert to grayscale
    st.image(image, caption="Uploaded Image", width=150)

    # Preprocess
    image = ImageOps.invert(image)   # MNIST digits are white on black
    image = image.resize((28, 28))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28)

    # Predict
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)

    st.success(f"✅ Predicted Digit: **{predicted_label}**")

# ----------------------------
# Drawing Section
# ----------------------------
# ----------------------------
# Drawing Section
# ----------------------------
st.subheader("🖌️ Draw a Digit")

# Button to clear canvas
if st.button("🧹 Clear Canvas"):
    st.session_state["canvas_key"] = st.session_state.get("canvas_key", 0) + 1
else:
    st.session_state["canvas_key"] = st.session_state.get("canvas_key", 0)

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key=f"canvas_{st.session_state['canvas_key']}",  # reset key to clear
)

if canvas_result.image_data is not None:
    img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype(np.uint8))  # grayscale
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28)

    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)

    st.success(f"✅ Predicted Digit: **{predicted_label}**")

