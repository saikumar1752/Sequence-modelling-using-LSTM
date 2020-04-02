# Sequence-modelling-using-LSTM
Sequence modelling is used to predict the next word/letter in the sequence. LSTM(Long Short Term Memory) an recurrent neural network is used for sequence modelling. In the present context LSTM is used to generate Linux source code by generating one character at a time based on the previous sequence of characters.

Run pip3 -r requirements.py to download the necessary libraries

Run setup.py to download the linux source code from GitHub and unzip the files and preprocess the code
For this experiment only the file with extension .c are considered to train the model.
Run train.py to train the model and save the model parameters

Run generate.py with the start sequence and along with the number of characters to generate as arguments.
A notebook version of this code is also available and can be run on Google colab.
# Note
The code is configured to run only on a gpu.
