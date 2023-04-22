# Convolutional-Neural-Network
Designed deep network for scene recognition using miniplaces data set

To Train: python train_miniplaces.py --resume ./outputs/checkpoint.pth.tar

To Evaluate: python eval_miniplaces.py --load ./outputs/model_best.pth.tar

To run model with different configurations: 
--epochs: number of total epochs to run
--lr: initial learning rate
--batch-size: number of images within a mini-batch
--resume: path to latest checkpoint (default: none)
