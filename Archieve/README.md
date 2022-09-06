# Archieve

This folder contains archieved experiments.

You can copy the py files and pth file into project's root folder(Replace) to test.

## ResNet152\_K

This folder contains pretrained ResNet152 CNN part and myKakuritsu based FC 2 Layers, each layer has 1000 neural cells.

The pretrained ResNet152 CNN part is static(Not trainable), the FC 2 layers can be trained.

Set learning rate to 0.01, Kakuritsu rate(p) 0.5, training on ILSVRC2012, best pth saved at epoch 3.

Keep p = 0.5 in Validation:

Acc@1 Avg= 67.090 Acc@5 Avg= 87.362

Switch p = 1.0 in Validation:

Acc@1 Avg= 75.338 Acc@5 Avg= 92.260

## ResNet152\_D

This folder contains ResNet152 CNN part and Dropout based FC 2 Layers, each layer has 1000 neural cells.

The pretrained ResNet152 CNN part is static(Not trainable), the FC 2 layers can be trained.

Set learning rate to 0.01, Dropout rate(p) 0.5, training on ILSVRC2012, best pth saved at epoch 5.

Keep p = 0.5 in Validation:

Acc@1 Avg= 35.752 Acc@5 Avg= 43.654

Switch p = 1.0 in Validation:

Acc@1 Avg= 72.248 Acc@5 Avg= 89.688


