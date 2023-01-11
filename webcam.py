import argparse
import cv2
from keras.models import load_model
import numpy as np
from coloration import change_color

# superpose the mask on the frame and return the result 
def process_frame(frame, model):

    frame_1 = cv2.resize(frame, (128, 128))
    mask = np.expand_dims(frame_1, axis=0)
    mask = mask / 255.
    mask = model.predict(mask)
    treshold = 0.7
    pred_mask = ((mask > treshold) * 255.)
    mask = pred_mask[0] # 128x128x1
    mask = mask.astype(np.uint8)
    mask = np.squeeze(mask, axis=2) # 128x128
    
    # create image in which the pixels that are zero in the mask are retained from the original image, and the pixels that are nonzero in the mask are set to zero.
    # invert the mask 
    mask = cv2.bitwise_not(mask)
    # Split the image into its three color channels
    b,g,r = cv2.split(frame_1)

    # Create a mask for each color channel
    mask_b = cv2.bitwise_and(b, b, mask=mask)
    mask_g = cv2.bitwise_and(g, g, mask=mask)
    mask_r = cv2.bitwise_and(r, r, mask=mask)

    # Combine the masks
    mask = cv2.bitwise_or(mask_b, mask_g)
    mask = cv2.bitwise_or(mask, mask_r)

    # Overlay the mask on the original image to highlight the hair
    result = cv2.bitwise_and(frame_1, frame_1, mask=mask)
    result = cv2.resize(result, (frame.shape[1]*2, frame.shape[0]*2))
    return result
    
def process_frame_color(frame, model):
    # ne marche pas à revoir
    frame_1 = cv2.resize(frame, (128, 128))
    mask = np.expand_dims(frame_1, axis=0)
    mask = mask / 255.
    mask = model.predict(mask)
    result=change_color(frame_1, mask, '#ff0000')
    result=cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    #resize the result to the original frame size
    result = cv2.resize(result, (frame.shape[1]*2, frame.shape[0]*2))
    return result 

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
