# Context-Dependent Gating

This code accompanies our paper:
> Alleviating catastrophic forgetting using context-dependent gating and synaptic stabilization  
> Nicolas Y. Masse, Gregory D. Grant, David J. Freedman  
> https://arxiv.org/abs/1802.01569  

The recurrent network model can be found in the repository:
> https://github.com/gdgrant/Context-Dependent-Gating-RNN

Dependencies:  
> Python 3  
> TensorFlow 1+  

In the paper, the model is tested on the following datasets:  
**MNIST Dataset**
> https://github.com/mrgloom/MNIST-dataset-in-different-formats
>
> The dataset folder for MNIST is extracted and placed in './mnist/', so accessing the data from stimulus.py will be, for example, './mnist/data/original/train-images-idx3-ubyte'

**CIFAR Dataset**
> https://www.cs.toronto.edu/~kriz/cifar.html
>
> The dataset folders for CIFAR-10 and CIFAR-100 are extracted and placed separately in './cifar/', so accessing the data from stimulus.py will be, for example, './cifar/cifar-10-python/data_batch_1' or './cifar/cifar-100-python/test'

**ImageNet Dataset**
>  http://image-net.org/
> 
> The dataset files for ImageNet are extracted and placed into './ImageNet', so accessing the data from stimulus.py will be, for example, './ImageNet/train_data_batch_1'
