# AMD

# Hadasa
Send cluster bid
Do we have shared memory for the files?


# ideas:
Add labels to unlabeled data by checking if the first and the last are labeled the same
Check similarity between different oct scans in order to label them?

# Questions
Data:
How do I use all the oct slices?
How do I use less than 8/16 slices? start simple
How do I use parital labeled data? only last, 
Should we predict improvment and worsening?
Augmentations - resize and crop (224, 448), rotating in small angle, cut mix 
simclr augmentation

Model:
Model training - change the training code? or traing the model to different task? rot net/image inpaiting/Puzzle
Should we use single MLP or multiple MLPs for the multi-output classification?
Should we change the final linear layer to more complex layer?

# Next things 
How to handle mislabeled data?

# TODO
Extract data info
write model
write training function - 
    build trainer - model, optimizer, train_loader, val_loader,
    train, train epoch, eval epoch

# Data errors
AIA 03775 labeled several times


# Problematic data proccessing
AIA 03332 OD 03.12.2017


download pretrained model
configs 
timesformer -> models -> vit