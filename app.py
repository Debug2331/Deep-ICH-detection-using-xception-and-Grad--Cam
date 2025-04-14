import os
import MySQLdb
from flask import Flask, session, url_for, redirect, render_template, request, abort, flash
 
 
from werkzeug.utils import secure_filename
import numpy as np
import joblib
import numpy as np
from flask import Flask, redirect, url_for, request, render_template
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
from database import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from pathlib import Path
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key='detection'
 
app.config['UPLOAD_FOLDER'] = 'static/uploads'
 

def calculate_white_region_percentage(img_path, white_thresh=200, white_percentage_threshold=10):
    # Load the image
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for display with matplotlib

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply threshold to detect white regions
    _, white_regions = cv2.threshold(gray_img, white_thresh, 255, cv2.THRESH_BINARY)

    # Calculate the percentage of white pixels
    white_pixels = cv2.countNonZero(white_regions)
    total_pixels = img.shape[0] * img.shape[1]
    white_percentage = (white_pixels / total_pixels) * 100

    # Determine if white space is high or low based on threshold
    white_space_category = "High" if white_percentage > white_percentage_threshold else "Low"
    
    return img_rgb, white_regions, white_percentage, white_space_category


def get_img_array(img_path, size):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)  # Change to model-specific preprocessing if needed
    return img_array 


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/registera")
def registera():
    return render_template("register.html")

@app.route("/logina")
def logina():
    return render_template("login.html")

@app.route("/predicta")
def predicta():
    return render_template("predict.html")


@app.route("/predictionoutputa")
def predictoutputa():
    return render_template("predictionoutput.html")


@app.route("/register",methods=['POST','GET'])
def signup():
    if request.method=='POST':
        username=request.form['username']
        email=request.form['email']
        password=request.form['password']
        status = user_reg(username,email,password)
        if status == 1:
            return render_template("/login.html")
        else:
            return render_template("/register.html",m1="failed")        
    

@app.route("/login",methods=['POST','GET'])
def login():
    if request.method=='POST':
        username=request.form['username']
        password=request.form['password']
        status = user_loginact(request.form['username'], request.form['password'])
        print(status)
        if status == 1:                                      
            return render_template("/menu.html", m1="sucess")
        else:
            return render_template("/login.html", m1="Login Failed")
             
app.static_folder = 'static'

    
# @app.route("/")
# def home():
#     return render_template("index.html")

@app.route('/logouta')
def logout():
    # Clear the session data
    session.clear()
    return redirect(url_for('logina'))
    
def generate_recommendation(condition):
    data = ""
    print(condition)
    if condition == "Hemorrhagic":
        data = "Brain Hemorrhagic Detected in CT Scan"
    elif condition == "NORMAL":
        data = "CT Scan Image is Normal"
      
    return data

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def apply_heatmap_on_image(img_path, heatmap, intensity=0.4, colormap=cv2.COLORMAP_JET):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    superimposed_img = cv2.addWeighted(img, 1 - intensity, heatmap, intensity, 0)
    return img, superimposed_img

def detect_white_regions(heatmap_img, white_thresh=200):
    gray_img = cv2.cvtColor(heatmap_img, cv2.COLOR_RGB2GRAY)
    _, white_regions = cv2.threshold(gray_img, white_thresh, 255, cv2.THRESH_BINARY)
    white_mask = cv2.bitwise_and(heatmap_img, heatmap_img, mask=white_regions)
    return white_mask

@app.route('/prediction1', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image file from the request
        model = load_model('ex_model.h5')
        image_file = request.files['image']
        print("Received image file:", image_file)
        filename = secure_filename(image_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(file_path)
        #image = Image.open(file_path)
        img = image.load_img(file_path, target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict using the ensemble model
        predictions = model.predict(img_array)
        predicted_class = int(predictions[0] > 0.5)  # Binary classification threshold

        # Load a pre-trained model and specify the last conv layer name
        model = tf.keras.applications.VGG16(weights="imagenet")
        last_conv_layer_name = "block5_conv3"

        # Load and preprocess the input image
        img_path = file_path
        img_size = (224, 224)
        img_array = get_img_array(img_path, img_size)

        # Generate Grad-CAM heatmap
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

        # Apply the heatmap to the original image
        original_img, gradcam_img = apply_heatmap_on_image(img_path, heatmap)

        # Detect white regions in the Grad-CAM overlay
        white_mask = detect_white_regions(gradcam_img)

        # Display the images
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(original_img)

        plt.subplot(1, 3, 2)
        plt.title("Grad-CAM")
        plt.imshow(gradcam_img)

        plt.subplot(1, 3, 3)
        plt.title("White Regions in Grad-CAM")
        plt.imshow(white_mask)

        plt.savefig('D://2024//BRAIN HEMARRAGE//intracranial Haemorrhage//static//uploads//segmented.png')

        
        # Path to the image
        img_path = file_path # Replace with your image path

        # Calculate white region percentage
        original_img, white_regions, white_percentage, white_space_category = calculate_white_region_percentage(
            img_path, white_thresh=200, white_percentage_threshold=10
        )

        # Display results
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(original_img)

        plt.subplot(1, 2, 2)
        plt.title(f"White Regions (White %: {white_percentage:.2f}% - {white_space_category})")
        plt.imshow(white_regions, cmap='gray')

        plt.savefig('D://2024//BRAIN HEMARRAGE//intracranial Haemorrhage//static//uploads//white.png')
        print(f"White Space Percentage: {white_percentage:.2f}% - {white_space_category}")

        class_names = ['Hemorrhagic', 'NORMAL']
        print(f"Predicted class: {class_names[predicted_class]}")
        imgg='white.png'
        imgs='segmented.png'
        return render_template('predictionoutput.html', prediction={'p':class_names[predicted_class],'image':os.path.join(app.config['UPLOAD_FOLDER'], filename),'data':class_names[predicted_class]})
if __name__ == "__main__":
    app.run(debug=True)

     
     