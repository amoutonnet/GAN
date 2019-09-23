# Progressive WGAN with gradient penalty

For this personal project I implement a progressive growing WGAN, the one described in the paper from NVidia : https://arxiv.org/abs/1710.10196 <br/>
The tensorflow 2.0 version is working but I am still working on the "equalized learning rate" feature. <br/>
The tensorflow 1.14 version is not working yet, it lacks the management of the introduction of new blocks inside the training (Just have to make alpha shift from 0 to 1 for the growing phases). Plus, the management of the "equalized learning rate" feature still has some problem as it makes the weights go to 0, ruining the training.<br/><br/>
Results so far with the 2.0 version on mnist dataset:<br/>
1st phase: We generate 7x7 images<br/>
![Alt Text](https://media.giphy.com/media/iIH75iSqCaTl7GnGdG/giphy.gif)
2nd phase: We make the transition from 7x7 to 14x14 images<br/>
![Alt Text](https://media.giphy.com/media/W6Rw766evn0ZX5iDt5/giphy.gif)
3rd phase: We generate 14x14 images<br/>
![Alt Text](https://media.giphy.com/media/hS97cWnzxyUrGIru9g/giphy.gif)
4th phase: We make the transition from 14x14 to 28x28 images<br/>
![Alt Text](https://media.giphy.com/media/h58qiVXQngOHgqIdAi/giphy.gif)
5th phase: We generate 28x28 images<br/>
![Alt Text](https://media.giphy.com/media/ZEHPAhSndJGcPHMlP0/giphy.gif)
