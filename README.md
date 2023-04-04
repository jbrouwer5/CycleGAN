# CycleGAN
Uses two GANs to perform unpaired image set to image set translation

The GANs are both DCGANs which means that the generators and discriminators are built with convolutional neural networks. 

One GAN is trained to take images from setA as input as produce images that fit the distribution of setB and vice versa for the other. 

Loss is calculated by 
- comparing output of each generator to actual images from the other set
- seeing if the discriminators were correct in their guesses
- also feeding the output of each generator to the other generator and calculating loss for those in order to create a "closed loop" effect where the output should be the same as the original. 

If given a dataset with paintings by Monet and a dataset with photographs, the output should be like this. 

<img width="222" alt="image" src="https://user-images.githubusercontent.com/63489213/229849451-56cae4c9-ce5d-4f48-8bf7-8dabf8e48a0b.png">
