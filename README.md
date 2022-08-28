# Neuro myKakuritsu Research Code with PyTorch

## Idea

You may heard of [Dropout](https://jmlr.org/papers/v15/srivastava14a.html) which is A Simple Way to Prevent Neural Networks from Overfitting.

While, the author thought the neuro cells died in brain is useful, because it make neuros have ability to random cooperation and can prevent overfitting during learning.

We think currently computers can only simulate fewer neuros, the random death of neuros also cause serious memory lossing, makes convergence harder. The key reason that cause overfit is dataset's problem, enhance the dataset is the best way, but improving dataset is hard work. 

To increasing the Neuro Network's performance with limited dataset, We can increase the single neuro level Divergent ability, with myKakuritsu Activation.

Kakuritsu means probability in Japanese, instead of killing neuro cell, We let each synapse activation with a probability. This will make hidden layer's data Generalization during training but less memory lossing, maybe prevent overfitting better.

Still Need more Experment to prove this guess.

## Usage

python3 Imagenet\_train.py \[args\] \[Dataset\_Dir\]

positional arguments:
  DIR                   path to dataset (default: imagenet)

optional arguments:
  -h, --help            show this help message and exit
  -a ARCH, --arch ARCH  model architecture: Kakuritsu and Dropout, with ResNet152
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 64), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --wd W, --weight-decay W
                        weight decay (default: 1e-4)
  -p N, --print-freq N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --pretrained          use pre-trained model
  --world-size WORLD\_SIZE
                        number of nodes for distributed training
  --rank RANK           node rank for distributed training
  --dist-url DIST\_URL   url used to set up distributed training
  --dist-backend DIST\_BACKEND
                        distributed backend
  --seed SEED           seed for initializing training.
  --gpu GPU             GPU id to use.
  --multiprocessing-distributed
                        Use multi-processing distributed training to launch N processes per node, which has N GPUs. This is the fastest way to use PyTorch for either single
                        node or multi node data parallel training
  --dummy               use fake data to benchmark

## Credit

SuperHacker UEFI (Shizhuo Zhang)

Cookie (Yue Fang)

Research supported by Automation School, BISTU; Microsoft The Practice Space (ai-edu)
