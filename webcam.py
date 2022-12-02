import argparse
import cv2
from keras.models import load_model
import numpy as np

def process_frame(frame, model, image_size=(128,128,3)):
    original_size = frame.shape
    frame = cv2.resize(frame, image_size)
    model = load_model(model)
    mask = model(frame)
    frame = color(frame, mask)
    return cv2.resize(frame, original_size)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', action='store', required=True)    
    args = parser.parse_args()
    model = load_model(args.model)

    vid = cv2.VideoCapture(0)
    if vid.isOpened() is False:
        raise Exception("webcam not found")
    
    
    while(True):

        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        
        frame = process_frame(frame)
        
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
