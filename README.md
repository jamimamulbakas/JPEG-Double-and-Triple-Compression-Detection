# JPEG-Double-and-Triple-Compression-Detection

1.Pre-Processing
--> execute pre_proc.m for single, double and triple compressed images to save the respective d1.mat , d2.mat and d3.mat files in the same folder
--> execute generate_data.m to assign corresponding class labels, which generates t_data.mat containing the training data and l_data.mat containing the label data

2.Training the CNN
--> execute cnn_train_keras.py with t_data.mat and l_data.mat
--> checkpoint folder will be saved.

3.Testing CNN
--> execute pre_proc.m for forged test image to get test.mat file.
--> execute cnn_test_keras.py with test.mat.
--> Probability matrix is saved as pMtest.mat

4.Postprocessing
--> execute Postprocessing.m with pMtest.mat, generate res.mat
--> execute accuracy.m with res.mat for accuracy, F1score and success rate


Please cite following paper if you have used this code

@article{bakas2020double,
  title={Double and triple compression-based forgery detection in JPEG images using deep convolutional neural network},
  author={Bakas, Jamimamul and Ramachandra, Sumana and Naskar, Ruchira},
  journal={Journal of Electronic Imaging},
  volume={29},
  number={2},
  pages={023006},
  year={2020},
  publisher={International Society for Optics and Photonics}
}
