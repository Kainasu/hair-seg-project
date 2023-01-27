import argparse
import cv2
from keras.models import load_model
import numpy as np
from coloration import change_color, hex_to_rgb
    
def process_frame_color(frame, model, color):
    _, height, width, _ = model.layers[0].input_shape[0]
    # if the frame is not None 
    if frame is None:
        return None
    else:
        frame_1 = cv2.resize(frame, (width, height))

        mask = np.expand_dims(frame_1, axis=0)
        mask = mask / 255.
        mask = model.predict(mask)
        
        mask = np.squeeze(mask)
        
        result=change_color(frame_1, color, mask)
        
        result = cv2.resize(result, (frame.shape[1]*2, frame.shape[0]*2))
        return result 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', action='store', required=True)    
    parser.add_argument('--color', dest='color', action='store', required=True)
    args = parser.parse_args()
    model = load_model(args.model)

    vid = cv2.VideoCapture(0)
    if vid.isOpened() is False:
        raise Exception("webcam not found")
    
    
    while(True):

        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        
        frame = process_frame_color(frame, model,hex_to_rgb(args.color))
        
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
