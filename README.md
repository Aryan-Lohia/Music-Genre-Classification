# Music-Genre-Classification
Music Genre Classification using various deep learning techniques such as ANNs, CNNs and RNN-LSTM.

# Dataset Used

<h1>GTZAN Dataset - Music Genre Classification</h1>

<h2>About Dataset</h2>
<h3>Context</h3>
Music. Experts have been trying for a long time to understand sound and what differenciates one song from another. How to visualize sound. What makes a tone different from another.
This data hopefully can give the opportunity to do just that.

<h3>Content</h3>
genres original - A collection of 10 genres with 100 audio files each, all having a length of 30 seconds (the famous GTZAN dataset, the MNIST of sounds)\
images original - A visual representation for each audio file. One way to classify data is through neural networks. Because NNs (like CNN, what we will be using today) usually take in some sort of image representation, the audio files were converted to Mel Spectrograms to make this possible.\
2 CSV files - Containing features of the audio files. One file has for each song (30 seconds long) a mean and variance computed over multiple features that can be extracted from an audio file. The other file has the same structure, but the songs were split before into 3 seconds audio files (this way increasing 10 times the amount of data we fuel into our classification models). With data, more is always better.\

<h3>Acknowledgements</h3>
The GTZAN dataset is the most-used public dataset for evaluation in machine listening research for music genre recognition (MGR). The files were collected in 2000-2001 from a variety of sources including personal CDs, radio, microphone recordings, in order to represent a variety of recording conditions 
(<a href="http://marsyas.info/downloads/datasets.html">Original Dataset</a>)

# Techniques Used

<h1>ANN</h1>
Artificial Neural Network (ANN), is a group of multiple perceptrons or neurons at each layer. ANN is also known as a Feed-Forward Neural network because inputs are processed only in the forward direction.

Model: "Dense"
_________________________________________________________________
 Layer (type)          |      Output Shape       |       Param #   
=================================================================
 flatten (Flatten)     |      (None, 1690)        |      0         
                                                                 
 dense (Dense)         |      (None, 512)         |      865792    
                                                                 
 dropout (Dropout)     |      (None, 512)         |      0         
                                                                 
 dense_1 (Dense)       |      (None, 256)         |      131328    
                                                                 
 dropout_1 (Dropout)   |      (None, 256)         |      0         
                                                                 
 dense_2 (Dense)       |      (None, 64)          |      16448     
                                                                 
 dropout_2 (Dropout)   |      (None, 64)          |      0         
                                                                 
 dense_3 (Dense)       |      (None, 10)          |      650       
                                                                 
_________________________________________________________________
Total params: 1014218 (3.87 MB)\
Trainable params: 1014218 (3.87 MB)\
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________

<h1>CNN</h1>
Convolutional neural networks (CNN) are one of the most popular models used today. This neural network computational model uses a variation of multilayer perceptrons and contains one or more convolutional layers that can be either entirely connected or pooled. These convolutional layers create feature maps that record a region of image which is ultimately broken into rectangles and sent out for nonlinear processing.

Model: "CNN + Dense"
_________________________________________________________________
 Layer (type)            |    Output Shape          |    Param #   
=================================================================
 conv2d (Conv2D)         |    (None, 128, 11, 32)   |    320       
                                                                 
 max_pooling2d           |    (None, 64, 6, 32)     |    0         
 (MaxPooling2D)                                                              
                                                                 
 batch_normalization     |    (None, 64, 6, 32)     |    128       
 (Batch Normalization)                                                  
                                                                 
 conv2d_1 (Conv2D)       |    (None, 62, 4, 32)     |    9248      
                                                                 
 max_pooling2d_1         |    (None, 31, 2, 32)     |    0         
 (MaxPooling2D)                                                            
                                                                 
 batch_normalization_1   |    (None, 31, 2, 32)     |    128       
 (Batch Normalization)                                                
                                                                 
 conv2d_2 (Conv2D)       |    (None, 30, 1, 32)     |    4128      
                                                                 
 max_pooling2d_2         |    (None, 15, 1, 32)     |    0         
 (MaxPooling2D)                                                            
                                                                 
 batch_normalization_2   |    (None, 15, 1, 32)     |    128       
 (Batch Normalization)                                                
                                                                 
 flatten (Flatten)       |    (None, 480)           |    0         
                                                                 
 dense (Dense)           |    (None, 64)            |    30784     
                                                                 
 dropout (Dropout)       |    (None, 64)            |    0         
                                                                 
 dense_1 (Dense)         |    (None, 10)            |    650       
                                                                 
_________________________________________________________________
Total params: 45514 (177.79 KB)\
Trainable params: 45322 (177.04 KB)\
Non-trainable params: 192 (768.00 Byte)
_________________________________________________________________

<h1>RNN-LSTM</h1>
Recurrent neural networks (RNN) are more complex. They save the output of processing nodes and feed the result back into the model (they did not pass the information in one direction only). This is how the model is said to learn to predict the outcome of a layer. Each node in the RNN model acts as a memory cell, continuing the computation and implementation of operations. If the network’s prediction is incorrect, then the system self-learns and continues working towards the correct prediction during backpropagation.

Model: "RNN-LSTM"
_________________________________________________________________

 Layer (type)       |         Output Shape       |       Param #   
=================================================================
 lstm (LSTM)        |         (None, 130, 64)    |       19968     
                                                                 
 lstm_1 (LSTM)      |         (None, 64)         |       33024     
                                                                 
 dense (Dense)      |         (None, 64)         |       4160      
                                                                 
 dropout (Dropout)  |         (None, 64)         |       0         
                                                                 
 dense_1 (Dense)    |         (None, 10)         |       650       
                                                                 
_________________________________________________________________
Total params: 57802 (225.79 KB)\
Trainable params: 57802 (225.79 KB)\
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________

