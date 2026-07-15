import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from tensorflow.keras.models import load_model
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog

# -----------------------------
# Load Model
# -----------------------------
model = load_model("keras_model.h5", compile=False)

# Load labels
with open("labels.txt", "r") as f:
    class_names = [line.strip().split(" ", 1)[1] for line in f.readlines()]

# -----------------------------
# Prediction Function
# -----------------------------
def predict_image():

    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )

    if not file_path:
        return

    image = Image.open(file_path).convert("RGB")

    # Display image
    display_image = image.copy()
    display_image.thumbnail((300,300))

    photo = ImageTk.PhotoImage(display_image)

    image_label.config(image=photo)
    image_label.image = photo

    # Resize for model
    image = image.resize((224,224))

    image_array = np.asarray(image)

    normalized_image = (image_array.astype(np.float32)/127.5)-1

    data = np.expand_dims(normalized_image, axis=0)

    prediction = model.predict(data, verbose=0)

    index = np.argmax(prediction)

    confidence = prediction[0][index]*100

    result_label.config(
        text=f"Prediction : {class_names[index]}\nConfidence : {confidence:.2f}%"
    )


# -----------------------------
# GUI
# -----------------------------
root = tk.Tk()

root.title("Cat vs Dog Classifier")

root.geometry("500x600")

title = tk.Label(root,
                 text="🐱 Cat vs Dog Classifier 🐶",
                 font=("Arial",18,"bold"))

title.pack(pady=15)

btn = tk.Button(root,
                text="Choose Image",
                font=("Arial",13),
                command=predict_image)

btn.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)

result_label = tk.Label(root,
                        text="Choose an image",
                        font=("Arial",15))

result_label.pack(pady=15)

root.mainloop()