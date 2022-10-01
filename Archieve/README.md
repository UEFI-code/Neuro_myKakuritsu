# Archieve

This folder contains archieved experiments.

However because of No LFS uploaded, so you need to found pth files in my [Google Drive](https://drive.google.com/drive/folders/1J2_FkFKFnkagXT4x3rEZagRy-eK4HX8w?usp=sharing)

You can copy the py files and pth file into project's root folder(Replace) to test.

## ResNet152\_K

This folder contains pretrained ResNet152 CNN part and myKakuritsu based FC 2 Layers, each layer has 1000 neural cells.

The pretrained ResNet152 CNN part is static(Not trainable), the FC 2 layers can be trained.

Set learning rate to 0.01, Kakuritsu rate(p) 0.5, trained 20 epochs on ILSVRC2012, best pth saved at epoch 10.

Keep p = 0.5 in Validation:

Acc@1 Avg= 67.110 Acc@5 Avg= 87.186

Switch p = 1.0 in Validation:

Acc@1 Avg= 75.718 Acc@5 Avg= 92.240

## ResNet152\_D

This folder contains ResNet152 CNN part and Dropout based FC 2 Layers, each layer has 1000 neural cells.

The pretrained ResNet152 CNN part is static(Not trainable), the FC 2 layers can be trained.

Set learning rate to 0.01, Dropout rate(p) 0.5, trained 20 epochs on ILSVRC2012, best pth saved at epoch 16.

Keep p = 0.5 in Validation:

Acc@1 Avg= 36.366 Acc@5 Avg= 44.176

Switch p = 1.0 in Validation:

Acc@1 Avg= 72.996 Acc@5 Avg= 90.210

Replace Dropout to Kakuritsu with p = 0.5 in Validation:

Acc@1 Avg= 61.772 Acc@5 Avg= 81.744

## ResNet152\_W

This folder contains ResNet152 CNN part and FC 2 Layers, each layer has 1000 neural cells.

The pretrained ResNet152 CNN part is static(Not trainable), the FC 2 layers can be trained.

Set learning rate to 0.01, without Dropout or Kakurtistu, trained 20 epochs on ILSVRC2012, best pth saved at epoch 18.

Keep without Dropout or Kakurtistu in Validation:

Acc@1 Avg= 76.124 Acc@5 Avg= 93.134

Apply Kakurtisu with p = 0.5 in Validation:

Acc@1 Avg= 59.630 Acc@5 Avg= 85.538

Apply Dropout with p = 0.5 in Validation:

Acc@1 Avg= 38.678 Acc@5 Avg= 46.282


