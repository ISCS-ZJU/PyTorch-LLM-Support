# PyTorch-LLM-Support

## Applying the asynchronous tensor swapping to the PyTorch framework.


ZJU Large Model Support, based on IBM PyTorch Large Model Support ([LMS](https://github.com/IBM/pytorch-large-model-support)), enables the successful training of deep learning models that may otherwise exhaust GPU memory and result in "out-of-memory" errors. Our designs manage this oversubscription of GPU memory by temporarily transferring tensors to host memory when they are not required.


## Motivation

IBM PyTorch Large Model Support (LMS) currently only supports synchronous tensor swapping. DNN researchers still require a framework to support low-level asynchronous editable data offloading to support high-performance Large Model Training.



## Building PyTorch from the source

```
git clone https://github.com/pytorch/pytorch  
git clone https://github.com/Nayaco/mahoshojos-large-model-support mlms 
cd pytorch    
git checkout v1.12.0    
git am ../mlms/patches/pytorch_v1.12.0_large_model_support.patch   
```

Patch installation is completed according to the official Pytorch warehouse documentation compiled software, Pytorch official warehouse address: https://github.com/pytorch/pytorch/tree/v1.12.0

Torchvision compiling: https://github.com/pytorch/vision

## Examples

We show the examples in ([DenseNet with CIFAR10](https://github.com/Nayaco/mahoshojos-large-model-support/tree/main/examples))

## Authors

Wenjie Zhang (Master Student from Zhejiang University) : [Git Home page](https://github.com/Nayaco/mahoshojos-large-model-support/tree/main)

Editor & co-author: Ping Chen (PhD Student from Zhejiang University)
