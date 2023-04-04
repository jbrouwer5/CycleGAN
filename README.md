# CycleGAN
Uses two GANs to perform unpaired image set to image set translation

One GAN is trained to take images from setA as input as produce images that fit the distribution of setB and vice versa for the other. 

Loss is calculated by 
- comparing output of each generator to actual images from the other set
- seeing if the discriminators were correct in their guesses
- also feeding the output of each generator to the other generator and calculating loss for those in order to create a "closed loop" effect where the output should be the same as the original. 

