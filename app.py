import customtkinter as ctk
import cv2
from tkinter import filedialog, colorchooser
from segmentation import change_color
from PIL import ImageTk, Image
from webcam import process_frame_color
from keras.models import load_model



ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

root = ctk.CTk()
root.geometry("500x350")
model = "mobile_unet.h5"

def real_time():
    color = colorchooser.askcolor()[0]
    print(color)
    cap = cv2.VideoCapture(0)
    mod = load_model(model)
    while True:
        ret, cam_frame = cap.read()
        cam_frame = process_frame_color(cam_frame, mod, color)
        cv2.imshow("frame", cam_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def switch_to_real_time():
    frame.pack_forget()
    real_time_frame.pack(pady=20, padx=60, fill="both", expand=True)
    change_color_btn_rt.pack(pady=12, padx=10)
    


def image_btn():
    frame.pack_forget()
    image_frame.pack(pady=20, padx=60, fill="both", expand=True)
    import_img_btn.pack(pady=12, padx=10)
    change_color_btn.pack(pady=12, padx=10)

def load_image():
    global img
    global filepath
    filepath = filedialog.askopenfilename()
    img = cv2.imread(filepath)

def pick_color():
    color = colorchooser.askcolor()[0]
    colored_img = change_color(img, color)
    cv2.imwrite("colored_img.jpg", colored_img)
    colored_img = ImageTk.PhotoImage(Image.open("colored_img.jpg"))
    label = ctk.CTkLabel(master=image_frame, text="",image=colored_img)
    label.image = colored_img
    image = ImageTk.PhotoImage(Image.open(filepath))
    original_label = ctk.CTkLabel(master=image_frame, text="",image=image)
    original_label.image = img
    original_label.pack(pady=12, padx=10)
    label.pack(pady=12, padx=10)



    



frame = ctk.CTkFrame(master=root)
frame.pack(pady=20, padx=60, fill="both", expand=True)

label = ctk.CTkLabel(master=frame, text="Hair Segmentation App", font=("Roboto", 24))
label.pack(pady=12, padx=10)

real_time_btn = ctk.CTkButton(master=frame, text="Real Time Segmentation", command=switch_to_real_time)
real_time_btn.pack(pady=12, padx=10)
image_btn = ctk.CTkButton(master=frame, text="Image Segmentation", command=image_btn)
image_btn.pack(pady=12, padx=10)

image_frame = ctk.CTkFrame(master=root)
#image_frame.pack(pady=20, padx=60, fill="both", expand=True)

label = ctk.CTkLabel(master=image_frame, text="Hair Segmentation App", font=("Roboto", 24))
label.pack(pady=12, padx=10)

import_img_btn = ctk.CTkButton(master=image_frame, text="Import Image", command=load_image)
#import_img_btn.pack(pady=12, padx=10)

change_color_btn = ctk.CTkButton(master=image_frame, text="Choose Color", command=pick_color)
#change_color_btn.pack(pady=12, padx=10)

real_time_frame = ctk.CTkFrame(master=root)
#real_time_frame.pack(pady=20, padx=60, fill="both", expand=True)

realtime_label = ctk.CTkLabel(master=real_time_frame, text="Hair Segmentation App", font=("Roboto", 24))
realtime_label.pack(pady=12, padx=10)

change_color_btn_rt = ctk.CTkButton(master=real_time_frame, text="Choose Color", command=real_time)
#change_color_btn_rt.pack(pady=12, padx=10)

root.mainloop()