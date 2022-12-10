import sys
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    
    if len(sys.argv) == 1:
        print("Give history file as argument")
    else :
        history=np.load(sys.argv[1],allow_pickle='TRUE').item()

        h = history
        plt.plot(h.history['acc'], label='acc')
        plt.plot(h.history['val_acc'], label='val_acc')
        plt.title('Model accuracy')
        plt.legend()
        plt.show()
        
        plt.plot(h.history['loss'], label='loss')
        plt.plot(h.history['val_loss'], label='val_loss')
        plt.title('Model Loss')
        plt.legend()
        plt.show()
    
        plt.plot(h.history['binary_io_u'], label='binary_io_u')
        plt.plot(h.history['val_binary_io_u'], label='val_binary_io_u')
        plt.title('Model Loss')
        plt.legend()
        plt.show()