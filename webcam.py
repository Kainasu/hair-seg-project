import argparse
import cv2
from keras.models import load_model
import numpy as np
from coloration import change_color

def process_frame(frame, model):
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
    return frame 
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', action='store', required=True)    
    args = parser.parse_args()
    model = load_model(args.model)

    vid = cv2.VideoCapture(0)
    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))
    if vid.isOpened() is False:
        raise Exception("webcam not found")
    
    
    while(True):

        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        
        frame = process_frame(frame, model)
        
        # Display the resulting frame
        cv2.imshow('frame', frame)
        
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
