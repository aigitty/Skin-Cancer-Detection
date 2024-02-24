from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load the trained model
model = load_model("model_SkinCancer.h5")

# Define the path to the downloaded image
custom_image_path = "Skin_Cancer_Custom/bgn1.jpg"

# Function to classify custom input and display the image
def classify_and_display(custom_image_path):
    # Preprocess the input image
    img = image.load_img(custom_image_path, target_size=(100, 100))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions
    predictions = model.predict(img_array)

    # Post-process predictions
    predicted_class_index = np.argmax(predictions)
    class_labels = {0: 'benign', 1: 'malignant'}
    predicted_class_label = class_labels[predicted_class_index]

    # Display the image
    img = mpimg.imread(custom_image_path)
    plt.imshow(img)
    plt.title(f"The predicted class is: {predicted_class_label}")
    plt.axis('off')
    plt.show()

# Classify and display the downloaded image
classify_and_display(custom_image_path)
