#Importing the required modules
import tkinter as tk
from tkinter import *
import easygui
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image,ImageTk
from scipy.interpolate import UnivariateSpline

#Making the GUI main window 
top = tk.Tk()
top.geometry('400x400')
top.title('Cartoonify Your Image!')
top.configure(background='blue')
label = Label(top, background='#CDCDCD', font=('calibri', 20, 'bold'))

#Setting the background of the main window
load=Image.open("background.jpg")
render=ImageTk.PhotoImage(load)
bgt=Label(top,image=render)
bgt.place(x=0,y=0)

#Function to display image
def display(Image,title):
    plt.imshow(Image,cmap="gray")
    plt.axis("off")
    plt.title(title)
    plt.show()

#Function to upload image from device for cartoonifying
def upload():
    ImagePath = easygui.fileopenbox()
    img=cv2.imread(ImagePath)
    if img is None:
        print("Could not find any image, choose appropriate file.")
    cartoonify(img, ImagePath)

#Function to open camera and take image for applying effects
def camera():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")
    img_counter = 0
    while img_counter==0:
        ret, img = cam.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow("test", img)
        k = cv2.waitKey(1)
        if k%256 == 27:
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            img_counter += 1
    cam.release()
    cv2.destroyAllWindows()
    ImagePath = "background.jpg"
    cartoonify(img, ImagePath)

#Function to open camera and take image for applying effects
def cam2():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")
    img_counter = 0
    while img_counter == 0:
        ret, img = cam.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow("test", img)
        k = cv2.waitKey(1)
        if k%256 == 27:
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            img_counter += 1
    cam.release()
    cv2.destroyAllWindows()
    ImagePath = "background.jpg"
    tryEffects(img)

#Function to upload image from device for more effects
def up2():
    ImagePath = easygui.fileopenbox()
    img=cv2.imread(ImagePath)
    if img is None:
        print("Could not find any image, choose appropriate file.")
    tryEffects(img)

#Funtion for saving an image with the provided path
def save(Image, ImagePath):
    newName="Cartoonified_Image"
    path1 = os.path.dirname(ImagePath)
    extension=os.path.splitext(ImagePath)[1]
    path = os.path.join(path1, newName+extension)
    cv2.imwrite(path, cv2.cvtColor(Image, cv2.COLOR_RGB2BGR))
    I= "Image saved by name " + newName +" at "+ path
    tk.messagebox.showinfo(title=None, message=I)

#Funtion for converting image to pencil scketch (both colored and grey)
def pencil_sketch(img1):
    sk_gray, sk_color = cv2.pencilSketch(img1, sigma_s=70, sigma_r=0.04, shade_factor=0.2)
    sketches=[sk_gray, sk_color]
    fig, axes = plt.subplots(1,2, figsize=(8,8), subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(sketches[i], cmap='gray')
    plt.show()

#Funtion for Apllying Darker Filter to image
def dark(img):
    img_dark = cv2.convertScaleAbs(img, beta=-50)
    display(img_dark,"Darker Image")

#Funtion for converting image to Gray image
def greyscale(img):
    greyImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    display(greyImg,"Gray Image")

#Funtion for Brigtening image
def bright(img):
    img_bright = cv2.convertScaleAbs(img, beta=50)
    display(img_bright,"Brighter Image")

#Funtion for Sharpen image
def sharpen(img):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
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

#Making the Effect list GUI window
def tryEffects(Orig):

    #Getting the orignal image (maintaining the color space)
    Orig = cv2.cvtColor(Orig, cv2.COLOR_BGR2RGB)

    #Making the GUI window 
    td = tk.Tk()
    td.geometry('400x400')
    td.title('Effects')
    td.configure(background='SlateGray4')
    label3 = Label(td, background='#CDCDCD', font=('calibri', 20, 'bold'))

    #Button for Pencil Sketch
    PencilSketch = Button(td, text="Pencil Sketch", command=lambda: pencil_sketch(Orig), padx=10, pady=5)
    PencilSketch.configure(background="#374256", foreground="wheat", font=('calibri', 10, 'bold'))
    PencilSketch.pack(side=TOP, pady=5)

    #Button for Dark Filter
    Dark = Button(td, text="Darker Image", command=lambda: dark(Orig), padx=10, pady=5)
    Dark.configure(background="#374256", foreground="wheat", font=('calibri', 10, 'bold'))
    Dark.pack(side=TOP, pady=5)

    #Button for Brighten Image
    BrightI = Button(td, text="Brighten Image", command=lambda: bright(Orig), padx=10, pady=5)
    BrightI.configure(background="#374256", foreground="wheat", font=('calibri', 10, 'bold'))
    BrightI.pack(side=TOP, pady=5)

    #Button for Sharpen Image
    SharpenI = Button(td, text="Sharpen Image", command=lambda: sharpen(Orig), padx=10, pady=5)
    SharpenI.configure(background="#374256", foreground="wheat", font=('calibri', 10, 'bold'))
    SharpenI.pack(side=TOP, pady=5)

    #Button for Grey Image
    GrayI = Button(td, text="GreyScaled Image", command=lambda: greyscale(Orig), padx=10, pady=5)
    GrayI.configure(background="#374256", foreground="wheat", font=('calibri', 10, 'bold'))
    GrayI.pack(side=TOP, pady=5)

    #Button for HDR Image
    hdrI = Button(td, text="HDR Image", command=lambda: HDR(Orig), padx=10, pady=5)
    hdrI.configure(background="#374256", foreground="wheat", font=('calibri', 10, 'bold'))
    hdrI.pack(side=TOP, pady=5)

    #Button for Inverted Image
    InvertI = Button(td, text="Inverted Image", command=lambda: invert(Orig), padx=10, pady=5)
    InvertI.configure(background="#374256", foreground="wheat", font=('calibri', 10, 'bold'))
    InvertI.pack(side=TOP, pady=5)

    #Button for Warm Image
    Warm = Button(td, text="Warm Image", command=lambda: Summer(Orig), padx=10, pady=5)
    Warm.configure(background="#374256", foreground="wheat", font=('calibri', 10, 'bold'))
    Warm.pack(side=TOP, pady=5)

    #Button for Cold Image
    Cold = Button(td, text="Cold Image", command=lambda: Winter(Orig), padx=10, pady=5)
    Cold.configure(background="#374256", foreground="wheat", font=('calibri', 10, 'bold'))
    Cold.pack(side=TOP, pady=5)
    
    td.mainloop()

#Making the second GUI window (For more effects)
def effects():

    #Making the Second window 
    down = tk.Tk()
    down.geometry('200x300')
    down.title('Try More Effects!')
    down.configure(background='LightSkyBlue3')
    label = Label(down, background='#CDCDCD', font=('calibri', 20, 'bold'))

    #Making the Camera button in the second window
    camera = Button(down, text="Open Camera", command=cam2, padx=10, pady=5)
    camera.configure(background="#374256", foreground="wheat", font=('calibri', 10, 'bold'))
    camera.pack(side=TOP, pady=50)

    #Making the upload button in the second window
    upload = Button(down, text="Choose Image", command=up2, padx=10, pady=5)
    upload.configure(background="#374256", foreground="wheat", font=('calibri', 10, 'bold'))
    upload.pack(side=TOP, pady=50)

    down.mainloop()

#Making the Camera button in the GUI main window
camera = Button(top, text="Camera", command=camera, padx=10, pady=5)
camera.configure(background="#374256", foreground="wheat", font=('calibri', 10, 'bold'))
camera.pack(side=TOP, pady=40)

#Making the Upload button in the GUI main window
upload = Button(top, text="Upload Image", command=upload, padx=10, pady=5)
upload.configure(background="#374256", foreground="wheat", font=('calibri', 10, 'bold'))
upload.pack(side=TOP, pady=40)

#Making the Try More Effects button in the GUI main window
tryEffectsButton = Button(top, text="Effects", command=effects, padx=10, pady=5)
tryEffectsButton.configure(background="#374256", foreground="wheat", font=('calibri', 10, 'bold'))
tryEffectsButton.pack(side=TOP, pady=40)

#Funtion for cartoonifying an Image
def cartoonify(Orignal,ImagePath):

    #To maintain the color space of the orignal image
    Orignal=cv2.cvtColor(Orignal,cv2.COLOR_BGR2RGB)
    #Displaying Orignal Image
    display(Orignal,"Original Image")    

    #Converting the color space from RGB to Grayscale
    Grayed=cv2.cvtColor(Orignal,cv2.COLOR_RGB2GRAY)
    #Displaying Gray Image
    display(Grayed,"Gray Image")

    #Applying median blur to image
    Blurred=cv2.medianBlur(Grayed,5)
    #Displaying Blurred Image
    display(Blurred,"Median Blurred Image")

    #Creating edge mask
    line_size = 15
    blur_value = 10
    LightEdged = cv2.adaptiveThreshold(Blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
    #Displaying Edge Masked Image
    display(LightEdged,"Light Edge Masked Image")

    #Applying bilateral filter to remove noise as required
    NoiseFree=cv2.bilateralFilter(Orignal, 15, 80, 80)
    #Displaying Noise Free Image
    display(NoiseFree,"Noisefree")


    #Implementing K-Means Clustering (For number of colors in the image)
    k = number_of_colors = 5
    temp=np.float32(NoiseFree).reshape(-1,3)
    criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,20,1.0)
    compactness,label,center=cv2.kmeans(temp,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center=np.uint8(center)
    final_img=center[label.flatten()]
    final_img=final_img.reshape(Orignal.shape)
    display(final_img,"k-means")

    #Final Cartoon Image
    Final=cv2.bitwise_and(final_img,final_img,mask= LightEdged)
    display(Final,"FINAL")

    #For all transition Plot
    images=[Orignal, Grayed, Blurred, LightEdged, NoiseFree, final_img, Final]
    fig, axes = plt.subplots(4,2, figsize=(8,8), subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        if i<len(images):
            ax.imshow(images[i], cmap='gray')
        else:
            ax.axis('off')
    plt.show()

    #Save Button
    saveButton=Button(top,text="Save Cartoonified image",command=lambda: save(Final, ImagePath),padx=30,pady=5)
    saveButton.configure(background='#374256', foreground='wheat',font=('calibri',10,'bold'))
    saveButton.pack(side=TOP,pady=40)
 
#Main function to build the GUI window
top.mainloop()







