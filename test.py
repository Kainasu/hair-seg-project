from generator import create_training_generators, create_testing_generator
import os
import numpy as np
import matplotlib.pyplot as plt

dataset = 'Figaro1k'
test_generator, test_steps = create_testing_generator(dataset=dataset)

cols = ['Original', 'GT', 'pred', 'tresholded pred']
fig, axes = plt.subplots(15, 4, figsize=(10, 20))
for ax, col in zip(axes[0], cols):
    ax.set_title(col)
i = 0
t = 0
for generator in test_generator: 
    #Original
    axes[i, 0].imshow(generator[0])
    axes[i, 0].axis('off')
    #GT
    axes[i, 1].imshow(generator[1][0])
    axes[i, 1].axis('off')
    #Pred
    axes[i, 2].imshow(generator[0][0])
    axes[i, 2].axis('off')
    #Tresholded pred
    axes[i, 3].imshow(generator[1][0])
    axes[i, 3].axis('off')
    i += 1
    if i % 15 == 0:        
        plt.savefig(f'{cols[t]}.png')
        t += 1
        i = 0
        if t % 2 == 0:
            break        
        




"""
    plt.subplot(20,4,1+4*i)
    plt.imshow(generator[0][0])
    plt.axis('off')
    plt.subplot(20,4,2+4*i)
    plt.imshow(generator[1][0])
    plt.axis('off')
    plt.subplot(20,4,3+4*i)
    plt.imshow(generator[0][0])
    plt.axis('off')
    plt.subplot(20,4,4+4*i)
    plt.imshow(generator[1][0])
    plt.axis('off')
    i += 1
    if i % 20 == 0:
        break

plt.show()
"""