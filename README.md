# RGDA
Domain Adaption Via Relation-Guided Dilated Attention

## Prerequisites:

* Python3
* PyTorch == 1.5.0 (with suitable CUDA and CuDNN version)
* Numpy
* argparse
* PIL
* tqdm

## Contribution:

The contributions of this paper are four-fold:  
1. Domain Relationships Attention is proposed to reveal the relationships between source domain and target domain by computing their interaction to capture transferable features.
2. Multiple Feature-Fusion is presented to allow each neuron adaptively adjusting its receptive field size based on multiple scales of input information to acquire transferable features.   
3. RGDA can be easily implemented by most deep learning libraries. Extensive experiments on public datasets demonstrate that RGDA is comparable to state-of-the-art methods. 
