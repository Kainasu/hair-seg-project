import argparse
import numpy as np
import sys
import matplotlib.pyplot as plt
import time
import cv2

from keras.models import load_model
import h5py
from tkinter import *
from tkinter import filedialog
from tkinter import PhotoImage

def predict(image, model, height=128, width=128):
    im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    model = load_model(model)
    """Preprocess the input image before prediction"""
    im = im / 255
    im = cv2.resize(im, (height, width))
    im = im.reshape((1,) + im.shape)
    pred = model.predict(im)   
    mask = pred.reshape((height, width))
    return mask


def predict_and_plot(img_path, model, color, mask_path=None):
    ncols = 2

    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))

    if color is not None:
        ncols += 1
        col_color = 3        

    if mask_path is not None:

        mask = cv2.imread(mask_path)
        mask = cv2.resize(mask, (128, 128))
        ncols += 1
        col_mask = 3 if ncols < 4 else 4

    pred = predict(img, model)
    
    treshold = 0.7
    pred_mask = ((pred > treshold) * 255.)
    
    fig, axes = plt.subplots(nrows=1, ncols=ncols)    
    plt.subplot(1,ncols,1)    
    plt.imshow(img)
    plt.title("Original")
    plt.subplot(1,ncols,2)
    plt.imshow(pred_mask)
    plt.title("pred")

    if color is not None:                
        plt.subplot(1,ncols,col_color)
        colored_img = change_color(img, pred, color) 
        plt.imshow(cv2.cvtColor(colored_img, cv2.COLOR_BGR2RGB))
        plt.title("Colored photo")

    if mask_path is not None:
        plt.subplot(1,ncols, col_mask)
        plt.imshow(mask)
        plt.title("GT")
        # score = model.evaluate(img, mask[np.newaxis,:,:])
        # print("accuacy : ", score[1])
    
    plt.show()
    

def change_color(image, model, color, height=128, width=128):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = predict(image, model, height, width)
    thresh = 0.7
    """Create 3 copies of the mask, one for each color channel"""
    blue_mask = mask.copy()    
    blue_mask[mask > thresh] = color[0]
    blue_mask[mask <= thresh] = 0
    
    green_mask = mask.copy()    
    green_mask[mask > thresh] = color[1]
    green_mask[mask <= thresh] = 0

    red_mask = mask.copy()
    red_mask[mask > thresh] = color[2]
    red_mask[mask <= thresh] = 0

    blue_mask = cv2.resize(blue_mask, (image.shape[1], image.shape[0]))
    green_mask = cv2.resize(green_mask, (image.shape[1], image.shape[0]))
    red_mask = cv2.resize(red_mask, (image.shape[1], image.shape[0]))

    """Create an rgb mask to superimpose on the image"""
    mask_n = np.zeros_like(image)
    mask_n[:, :, 0] = blue_mask
    mask_n[:, :, 1] = green_mask
    mask_n[:, :, 2] = red_mask
    alpha = 0.85
    beta = (1.0 - alpha)
    out = cv2.addWeighted(image, alpha, mask_n, beta, 0.0)    
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return out


from tkinter import *
from tkinter import filedialog, colorchooser
import customtkinter as ctk
from PIL import ImageTk, Image


class HairColorApp:
    def __init__(self, master):
        self.master = master
        master.title("Segmentation App")

        # Create widgets
        self.load_button = ctk.CTkButton(root, text="Load Image", width=400, height=100)
        self.color_button = ctk.CTkButton(root, text="Choose Color", width=400, height=100)
        self.change_color_button = ctk.CTkButton(root, text="Change Color", width=400, height=100)

        # Set widget layout
        self.load_button.grid(row=0, column=0, padx=100, pady=100)
        self.color_button.grid(row=0, column=1, padx=100, pady=100)
        self.change_color_button.grid(row=0, column=1, padx=100, pady=100)

        self.color_button.place(relx=0.5, rely=0.5, anchor=CENTER)
        self.change_color_button.place(relx=0.8, rely=0.5, anchor=CENTER)
        self.load_button.place(relx=0.2, rely=0.5, anchor=CENTER)

        # Set widget callbacks
        self.load_button.configure(command=self.load_image)
        self.color_button.configure(command=self.pick_color)
        self.change_color_button.configure(command=self.change_color)

        self.model = "model.h5"


    def load_image(self):
        self.filepath = filedialog.askopenfilename()
        self.img = cv2.imread(self.filepath)

    def pick_color(self):
        self.color = colorchooser.askcolor()[0]

    def change_color(self):
        self.colored_img = change_color(self.img, self.model, self.color)
        cv2.imwrite("colored_img.jpg", self.colored_img)
        self.colored_img = ImageTk.PhotoImage(Image.open("colored_img.jpg"))
        self.colored_img_label = Label(image=self.colored_img)
        self.colored_img_label.image = self.colored_img
        self.colored_img_label.place(relx=0.5, rely=0.15, anchor=CENTER)


root = ctk.CTk()
my_gui = HairColorApp(root)



root.mainloop()
