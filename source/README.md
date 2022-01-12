# How the code works

In the file API-ESN_training.ipynb, we train the API-ESN to reconstruct the hidden variables in the Lorenz system. By changing the variable 'idx' and the function 'Lorenz_rec', different chocies of observed and hidden states are studied. Automatic differentiation and forward euler can be chosen by selecting the variable FE=False/True, respectively. Training is carried out using stochastic grdient descent using a Tensorflow 2.x graph defined using @tf.function.

The details for the Echo State Network reservoir and the automatic differentiation of the reservoir with respect to time are implemented in ESN_bias.ipynb and ESN_bias_drdt.ipynb, respectively. Both are implemented in Tensorflow 2.x in the form of cells for Recurrent Neural Networks.

Once all the different cases, which consists of different choices of observed states and size of the reservoir, are trained, post processing is carried out in Post_Process.ipynb, where the plots from the paper are reproduced.

More explanations regarding the code can be found throughout the scripts as comments.


