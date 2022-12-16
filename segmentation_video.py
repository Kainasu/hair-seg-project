import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt


def apply_model_on_video(model, video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (frame_width,frame_height))
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.resize(frame, (128, 128))
            frame = np.expand_dims(frame, axis=0)
            frame = frame / 255.
            frame = model.predict(frame)
            treshold = 0.7
            pred_mask = ((frame > treshold) * 255.)
            frame = pred_mask[0] # 128x128x1
            frame = frame.astype(np.uint8)
            frame = np.squeeze(frame, axis=2) # 128x128
            frame= cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) # 128x128x3
            frame = cv2.resize(frame, (frame_width, frame_height)) # 1280x720x3
            out.write(frame)
            
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    model = load_model('model.h5')
    apply_model_on_video(model, 'video_2.webm', 'output.mp4')

