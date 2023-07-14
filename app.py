from flask import Flask, render_template, request, redirect,url_for,session 
import cv2
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from PIL import Image
import tempfile
import uuid
from scipy.interpolate import UnivariateSpline

import secrets
import shutil


app = Flask(__name__)
app.secret_key = os.urandom(24)



# Function to display image in a pop-up window using OpenCV
def display(image, title):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(title)
    plt.show()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'button1' in request.form:
            # Button 1 is clicked, redirect to URL1
            return redirect(url_for('cartoonifyy'))
        elif 'button2' in request.form:
            # Button 2 is clicked, redirect to URL2
            return redirect(url_for('effects'))
    return render_template('form.html')

# Function to upload image from device for cartoonifying
@app.route('/cartoonifyy', methods=['GET', 'POST'])
def cartoonifyy():
    if request.method == 'POST':
        f = request.files['file']
        img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_UNCHANGED)

        if img is None:
            print("Could not find any image, choose an appropriate file.")
        else:
            cartoonify(img)
            
        return redirect('/cartoonifyy')
    return render_template('index.html')

# Function for converting image to pencil sketch (both colored and grey)
def pencil_sketch(img1):
    sk_gray, sk_color = cv2.pencilSketch(img1, sigma_s=60, sigma_r=0.07, shade_factor=0.1)
    sketches = [sk_gray, sk_color]
    fig, axes = plt.subplots(1, 2, figsize=(8, 8), subplot_kw={'xticks': [], 'yticks': []}, gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(sketches[i], cmap='gray')
    plt.show()

#Funtion for Apllying Sepia Filter to image (Adding a little red-brownish color)
def sepia(img):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    img_sepia = cv2.filter2D(img, -1, kernel)
    display(img_sepia,"Sepia Filtered Image")

#Funtion for converting image to Grey image
def greyscale(img):
    greyImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    display(greyImg,"Gray Image")

#Funtion for Brigtening image
def bright(img):
    img_bright = cv2.convertScaleAbs(img, beta=50)
    display(img_bright,"Brightened Image")

#Funtion for Sharpen image
def sharpen(img):
    kernel = np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]])
    img_sharpen = cv2.filter2D(img, -1, kernel)
    display(img_sharpen,"Sharpen Image")

#Funtion for converting image to HDR image
def HDR(img):
    hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    display(hdr,"HDR Image")

#Funtion for Inverting image
def invert(img):
    inv = cv2.bitwise_not(img)
    display(inv,"Inverted Image")

def LookupTable(x, y):
    spline = UnivariateSpline(x, y)
    return spline(range(256))

#Funtion for converting image to Warm image
def Winter(img):
    increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue_channel, green_channel,red_channel  = cv2.split(img)
    red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
    sum= cv2.merge((blue_channel, green_channel, red_channel ))
    display(sum,"Warm Image")

#Funtion for converting image to Cold image
def Summer(img):
    increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue_channel, green_channel,red_channel = cv2.split(img)
    red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
    win= cv2.merge((blue_channel, green_channel, red_channel))
    display(win,"Cold Image")


# Function for cartoonifying an Image
def cartoonify(Original):
        if Original.shape[2] == 3:
        # Convert the color space from BGR to RGB
            Original = cv2.cvtColor(Original, cv2.COLOR_BGR2RGB)
        # Convert the color space from BGR to RGB
        Original = cv2.cvtColor(Original, cv2.COLOR_BGR2RGB)
        # Display original image
        display(Original, "Original Image")

        # Convert the color space from RGB to grayscale
        Grayed = cv2.cvtColor(Original, cv2.COLOR_RGB2GRAY)
        # Display grayscale image
        display(Grayed, "Gray Image")
        
        # Apply median blur to reduce noise
        Blurred = cv2.medianBlur(Grayed, 5)
        # Display blurred image
        display(Blurred, "Median Blurred Image")

        # Create edge mask using adaptive thresholding
        line_size = 7
        blur_value = 7
        LightEdged = cv2.adaptiveThreshold(Blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size,
                                           blur_value)
        # Display edge masked image
        display(LightEdged, "Light Edge Masked Image")

        # Apply bilateral filter to reduce noise while preserving edges
        NoiseFree = cv2.bilateralFilter(Original, 15, 75, 75)
        # Display noise-free image
        display(NoiseFree, "Mask Image")

        # Erode and dilate the image to enhance features
        kernel = np.ones((1, 1), np.uint8)
        Eroded = cv2.erode(NoiseFree, kernel, iterations=3)
        Dilated = cv2.dilate(Eroded, kernel, iterations=3)
        # Display eroded and dilated image
        display(Dilated, "Eroded & Dilated Image")

        # Apply K-Means Clustering to reduce the number of colors in the image
        k = number_of_colors = 5
        temp = np.float32(Dilated).reshape(-1, 3)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        compactness, label, center = cv2.kmeans(temp, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        final_img = center[label.flatten()]
        final_img = final_img.reshape(Original.shape)
        display(final_img,"K-Means")

        # Apply edge mask to the final image
        Final = cv2.bitwise_and(final_img, final_img, mask=LightEdged)
        display(Final, "FINAL")


UPLOAD_FOLDER = 'uploads/temp'

# Function to create the temporary folder
def create_temp_folder():
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

# Function to save the uploaded image in the temporary folder
def save_uploaded_image(file):
    create_temp_folder()
    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    return filepath


@app.route('/effects', methods=['GET', 'POST'])
def effects():
    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        filepath = save_uploaded_image(file)
        session['filepath'] = filepath
        return redirect(url_for('apply_effect'))
    return render_template('effects.html')


@app.route('/apply_effect', methods=['GET', 'POST'])
def apply_effect():
    
    if request.method == 'POST':
        filepath = session.get('filepath')
        effect = request.form.get('effect')
        img = cv2.imread(filepath)
        if effect == 'HDR':
            HDR(img)
        if effect == 'Invert':
            invert(img)
        if effect=='Pencil_Sketch':
            pencil_sketch(img)
        if effect=='Sepia':
            sepia(img)
        if effect=='Grayscale':
            greyscale(img)
        if effect=='Bright':
            bright(img)
        if effect=='Sharpen':
            sharpen(img)
        if effect=='Warm':
            Winter(img)
        if effect=='Cool':
            Summer(img)

        return redirect(url_for('apply_effect'))

    return render_template('apply_effect.html')

if __name__ == '__main__':
    app.run(debug=True)
