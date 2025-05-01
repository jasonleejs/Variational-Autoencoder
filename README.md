# ConvVAE

The module defined in this repository provides a general implementation of a Variational Autoencoder that can be used to process and generate 2 dimensional images. 

Using this general module will involve defining the following model settings:
- input dimensions
- latent dimensions
- convolutional parameters:
  - number of cnn layers
  - stride
  - padding
- Activation function (during Convolution and output)
- Reconstruction Loss
- 