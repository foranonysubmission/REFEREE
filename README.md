# REFEREE
Open source code for paper *On Structural Explanation of Biases in Graph Neural Networks*.

## Environment
Experiments are carried out on a Tesla V100 GPU with Cuda 11.0.

Library details can be found in requirements.txt.

Notice: Cuda is enabled for default settings.

## Usage
We have three datasets for experiments. To choose the dataset, change the variable named prog_args.dataset in train.py. Then run the following command to train the GNN to be explained:
```
python train.py
```

To explain the trained GNN, first specify the dataset by changing the variable named prog_args.dataset in explainer_main.py. Then run the following command to explain the GNN:
```
python explainer_main.py
```


## Log example for training
```
python train.py 
```
```
epoch:  0 ; loss:0.7054  ; train_acc:  0.32625 ; test_acc:  0.27 ; epoch time:  0.01
epoch:  200 ; loss:0.5617  ; train_acc:  0.71 ; test_acc:  0.69 ; epoch time:  0.01
epoch:  400 ; loss:0.5182  ; train_acc:  0.745 ; test_acc:  0.69 ; epoch time:  0.01
epoch:  600 ; loss:0.5119  ; train_acc:  0.7525 ; test_acc:  0.7 ; epoch time:  0.01
epoch:  800 ; loss:0.5019  ; train_acc:  0.76 ; test_acc:  0.71 ; epoch time:  0.01
Optimization Finished!
```


## Log example for explaining:

```
python explainer_main.py
```

```
loading model
ckpt\german_base_h20_o20.pth.tar
=> loading checkpoint 'ckpt\german_base_h20_o20.pth.tar'
Loaded model from ckpt
input dim:  27 ; num classes:  2
Method:  base
node label:  0
neigh graph idx:  507 504
Node predicted label:  0
epoch:  0 ; loss: 37.0874 ; WD loss: 0.9769 ; KL loss: 0.0000 ; mask density: 0.6796 
epoch:  40 ; loss: 3.6809 ; WD loss: 0.8165 ; KL loss: 0.5373 ; mask density: 0.0604  
epoch:  80 ; loss: 1.2953 ; WD loss: 0.7718 ; KL loss: 0.4999 ; mask density: 0.0178  
epoch:  120 ; loss: 1.0574 ; WD loss: 0.7972 ; KL loss: 0.3701 ; mask density: 0.0103 
epoch:  160 ; loss: 0.9637 ; WD loss: 0.7861 ; KL loss: 0.2819 ; mask density: 0.0070  
finished training in  13.76
```
