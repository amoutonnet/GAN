# Progressive WGAN with gradient penalty

For this personal project I implement a progressive growing WGAN, the one described in the paper from NVidia : https://research.nvidia.com/publication/2017-10_Progressive-Growing-of
The tensorflow 2.0 version is working but I am still working on the "equalized learning rate" feature.
The tensorflow 1.14 version is not working yet, it lacks the management of the introduction of new blocks inside the training (Just have to make alpha shift from 0 to 1 for the growing phases). Plus, the management of the "equalized learning rate" feature still has some problem as it makes the weights go to 0, ruining the training.
