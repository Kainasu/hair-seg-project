import argparse
import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from coloration import change_color

def apply_model_on_video(model, video_path, output_path, color):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 15, (frame_width,frame_height))
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame_1 = cv2.resize(frame, (128, 128))
            frame_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2RGB)
            
            mask = np.expand_dims(frame_1, axis=0)
            mask = mask / 255.
            mask = model.predict(mask)
            
            treshold = 0.7
            pred_mask = ((mask > treshold) * 255.)
            mask = pred_mask[0]
            mask = mask.astype(np.uint8)
            mask = np.squeeze(mask, axis=2)
            
            result=change_color(frame_1, mask, color)
            
            result= cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            result = cv2.resize(result, (frame_width, frame_height))
            out.write(result)
            
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', action='store', required=True)    
    parser.add_argument('--color', dest='color', action='store', required=True)
    parser.add_argument('--input', dest='input', action='store', required=True)
    parser.add_argument('--output', dest='output', action='store', required=True)
    args = parser.parse_args()
    model = load_model(args.model)
    apply_model_on_video(model, args.input, args.output, args.color)