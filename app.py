import customtkinter as ctk
import cv2
import os
from tkinter import filedialog, colorchooser
from coloration import change_color
from PIL import ImageTk, Image
from webcam import process_frame_color
from keras.models import load_model

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Hair Segmentation App")
        self.geometry("700x450")
        self.model = "model.h5"
        self.img = None
        self.filepath = None
        self.color = None
        self.colored_img = None
        self.cap = None
        self.mod = None
        self.ret = None
        self.cam_frame = None
        self.label = None
        self.w = 0
        self.h = 0

        self.frame = ctk.CTkFrame(master=self)
        self.frame.pack(pady=20, padx=60, fill="both", expand=True)


        self.real_time_btn = ctk.CTkButton(master=self.frame, text="Real Time", command=self.switch_to_real_time)
        self.real_time_btn.pack(pady=12, padx=10)

        self.image_btn = ctk.CTkButton(master=self.frame, text="Image", command=self.switch_to_image)
        self.image_btn.pack(pady=12, padx=10)

        self.video_btn = ctk.CTkButton(master=self.frame, text="Video", command=self.switch_to_video)
        self.video_btn.pack(pady=12, padx=10)



    def real_time(self):
        self.cap = cv2.VideoCapture(0)
        self.mod = load_model(self.model)
        while True:
            self.ret, self.cam_frame = self.cap.read()
            self.cam_frame = process_frame_color(self.cam_frame, self.mod, self.color)
            cv2.imshow("frame", self.cam_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

    def pick_color_rt(self):
        self.color = colorchooser.askcolor()[0]
        self.real_time()

    def switch_to_real_time(self):
        self.real_time_frame = ctk.CTkFrame(master=self)
        self.frame.pack_forget()
        self.real_time_frame.pack(pady=20, padx=60, fill="both", expand=True)
        self.change_color_btn_rt = ctk.CTkButton(master=self.real_time_frame, text="Change Color", command=self.pick_color_rt)
        self.change_color_btn_rt.pack(pady=12, padx=10)
        self.home_btn = ctk.CTkButton(master=self.real_time_frame, text="Home", command=self.switch_to_home_rt)
        self.home_btn.pack(pady=12, padx=10)

    def switch_to_home_rt(self):
        self.frame = ctk.CTkFrame(master=self)
        self.real_time_frame.pack_forget()
        self.frame.pack(pady=20, padx=60, fill="both", expand=True)
        self.real_time_btn = ctk.CTkButton(master=self.frame, text="Real Time", command=self.switch_to_real_time)
        self.real_time_btn.pack(pady=12, padx=10)

        self.image_btn = ctk.CTkButton(master=self.frame, text="Image", command=self.switch_to_image)
        self.image_btn.pack(pady=12, padx=10)

        self.video_btn = ctk.CTkButton(master=self.frame, text="Video", command=self.switch_to_video)
        self.video_btn.pack(pady=12, padx=10)
        

    def switch_to_home(self):
        self.frame = ctk.CTkFrame(master=self)
        self.image_frame.pack_forget()
        self.frame.pack(pady=20, padx=60, fill="both", expand=True)
        self.real_time_btn = ctk.CTkButton(master=self.frame, text="Real Time", command=self.switch_to_real_time)
        self.real_time_btn.pack(pady=12, padx=10)

        self.image_btn = ctk.CTkButton(master=self.frame, text="Image", command=self.switch_to_image)
        self.image_btn.pack(pady=12, padx=10)

        self.video_btn = ctk.CTkButton(master=self.frame, text="Video", command=self.switch_to_video)
        self.video_btn.pack(pady=12, padx=10)


    def switch_to_home_video(self):
        self.frame = ctk.CTkFrame(master=self)
        self.video_frame.pack_forget()
        self.frame.pack(pady=20, padx=60, fill="both", expand=True)
        self.real_time_btn = ctk.CTkButton(master=self.frame, text="Real Time", command=self.switch_to_real_time)
        self.real_time_btn.pack(pady=12, padx=10)
        self.image_btn = ctk.CTkButton(master=self.frame, text="Image", command=self.switch_to_image)
        self.image_btn.pack(pady=12, padx=10)

        self.video_btn = ctk.CTkButton(master=self.frame, text="Video", command=self.switch_to_video)
        self.video_btn.pack(pady=12, padx=10)

    def switch_to_image(self):
        self.image_frame = ctk.CTkFrame(master=self)
        self.frame.pack_forget()
        self.image_frame.pack(pady=20, padx=60, fill="both", expand=True)
        self.import_img_btn = ctk.CTkButton(master=self.image_frame, text="Import Image", command=self.load_image)
        self.change_color_btn = ctk.CTkButton(master=self.image_frame, text="Change Color", command=self.pick_color)
        # home button
        self.home_btn = ctk.CTkButton(master=self.image_frame, text="Home", command=self.switch_to_home)
        self.save_img_btn = ctk.CTkButton(master=self.image_frame, text="Save Image", command=self.save_image)
        self.import_img_btn.pack(pady=12, padx=10)
        self.change_color_btn.pack(pady=12, padx=10)
        # Save image button
        self.save_img_btn.pack(pady=12, padx=10)
        self.home_btn.pack(pady=12, padx=10)

    def pick_color_video(self):
        self.color = colorchooser.askcolor()[0]
        self.mod = load_model(self.model)
        print(self.w, self.h)
        while True:
            self.ret, self.cam_frame = self.cap.read()
            self.cam_frame = process_frame_color(self.cam_frame, self.mod, self.color)
            if self.cam_frame is None:
                break
            if self.w == 0 or self.h == 0:
                self.w = self.cam_frame.shape[1]
                self.h = self.cam_frame.shape[0]
                self.out_vid = cv2.VideoWriter("colored_vid.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (self.w, self.h))
            self.out_vid.write(self.cam_frame)
            cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("frame", 600, 600)
            cv2.startWindowThread()
            cv2.imshow("frame", self.cam_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.out_vid.release()
        self.cap.release()
        cv2.destroyAllWindows()
        for i in range(1, 5):
            cv2.waitKey(1)
        


    def save_video(self):
        self.filepath = filedialog.asksaveasfilename()
        os.rename("colored_vid.mp4", self.filepath)

    def switch_to_video(self):
        self.video_frame = ctk.CTkFrame(master=self)
        self.frame.pack_forget()
        self.video_frame.pack(pady=20, padx=60, fill="both", expand=True)
        self.import_vid_btn = ctk.CTkButton(master=self.video_frame, text="Import Video", command=self.load_video)
        self.change_color_btn_vid = ctk.CTkButton(master=self.video_frame, text="Change Color", command=self.pick_color_video)
        self.import_vid_btn.pack(pady=12, padx=10)
        self.change_color_btn_vid.pack(pady=12, padx=10)
        # home button video
        self.home_btn_vid = ctk.CTkButton(master=self.video_frame, text="Home", command=self.switch_to_home_video)
        # Save video button
        self.save_vid_btn = ctk.CTkButton(master=self.video_frame, text="Save Video", command=self.save_video)
        self.save_vid_btn.pack(pady=12, padx=10)
        self.home_btn_vid.pack(pady=12, padx=10)




    def load_video(self):
        self.filepath = filedialog.askopenfilename()
        self.cap = cv2.VideoCapture(self.filepath)

    

    def save_image(self):
        self.filepath_save = filedialog.asksaveasfilename(defaultextension=".jpg")
        # rename colored_img.jpg to self.filepath_save
        os.rename("colored_img.jpg", self.filepath_save)

    def load_image(self):
        if self.label != None:
            self.label.place_forget()
            self.original_label.place_forget()
        self.filepath = filedialog.askopenfilename()
        self.img = cv2.imread(self.filepath)
        image = ImageTk.PhotoImage(Image.open(self.filepath))
        self.w = image.width()
        self.h = image.height()
        self.original_label = ctk.CTkLabel(master=self.image_frame, text="", image=image)
        self.original_label.place(x=10, y=20)
        
    
    def pick_color(self):
        if self.label != None:
            self.label.place_forget()
        self.color = colorchooser.askcolor()[0]
        self.colored_img = change_color(self.img, self.color)
        cv2.imwrite("colored_img.jpg", self.colored_img)
        self.colored_img = ImageTk.PhotoImage(Image.open("colored_img.jpg"))
        self.label = ctk.CTkLabel(master=self.image_frame, text="", image=self.colored_img)
        self.w_frame = self.image_frame.winfo_width()
        self.label.place(x=self.w_frame - self.w -10 , y=20)
        

if __name__ == "__main__":
    app = App()
    app.mainloop()
    #delete the colored image if it exists
    if os.path.exists("colored_img.jpg"):
        os.remove("colored_img.jpg")
    if os.path.exists("colored_vid.mp4"):
        os.remove("colored_vid.mp4")

