# Evaluating surgical skills from kinematic data using convolutional neural networks
This is the companion repository for [our paper](https://link.springer.com/chapter/10.1007/978-3-030-00937-3_25) also available on [ArXiv](https://arxiv.org/abs/1806.02750) titled "Evaluating surgical skills from kinematic data using convolutional neural networks".
This paper has been accepted at the [International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI) 2018](https://www.miccai2018.org).

## Architecture
![architecture fcn](https://github.com/hfawaz/miccai18/blob/master/archi.png)

## The source code
The software is developed using Python 3.6, it takes as input the kinematic data and predicts the subject's skill level. We trained the model on an NVIDIA GPU GTX 1080 (this is only necessary to speed up the calculations). You will need the [JIGSAWS dataset](https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/) to re-run the experiments of the paper. 
The source code is composed of one python file with the code needed to re-run the experiments. The hyper-parameters published in the paper are present in the source code. The empty folders (in the repository) are necessary so the code could run without the "folder not found error". The content of JIGSAWS.zip (once downloaded) should be placed in the empty folder "JIGSAWS".
If you are looking for more information on how to classify time series with deep learning, also have a look to our paper ["Deep learning for time series classification: a review"](https://arxiv.org/abs/1809.04356). The code of the models compared in this review is available [here](https://github.com/hfawaz/dl-4-tsc).

## Prerequisites
* [Keras](https://keras.io/) 2.12.2
* [Tensorflow](https://www.tensorflow.org/) 1.4.1
* [Numpy](http://www.numpy.org/) 1.13.3
* [Scikit-learn](http://scikit-learn.org/stable/) 0.19.1
* [Pandas](https://pandas.pydata.org/) 0.21.1
* [Matplotlib](https://matplotlib.org/) 2.0.0
* [Imageio](https://imageio.github.io/) 2.2.0

We used the versions listed above, but you should be able to run the code with much recent versions. Different versions of the above-listed packages can be found in the archives of the packages' corresponding websites.

## Visualizing the movements' contribution to the classification
The proposed method uses the Class Activation Map to localize the regions of the surgical task and their corresponding contribution to a certain classification.

A video illustrating the method can be seen [here](https://www.youtube.com/watch?v=1tjyWKREt2U). 

The functions that are used to visualize the trajectories are present in the source code.

Figure 2 in the paper, illustrates the trajectory of subject (Novice) H's movements for the left master manipulator.

The colors are obtained using the Class Activation Map values of a given class and then projected on the (x,y,z) coordinates of the master left hand.

You can find [here](https://germain-forestier.info/src/miccai2018/img/) the visualizations for all the subjects and all their trials for the Suturing task for each one of the five cross-validation folds.
Frame             |  Trajectory explained
:-------------------------:|:-------------------------:
![snap](https://github.com/hfawaz/miccai18/blob/master/snapshot-7.png)  |  ![cam](https://github.com/hfawaz/miccai18/blob/master/cam-7.png)

## Reference

If you re-use this work, please cite:

```
@InProceedings{IsmailFawaz2018evaluating,
  Title                    = {Evaluating surgical skills from kinematic data using convolutional neural networks},
  Author                   = {Ismail Fawaz, Hassan and Forestier, Germain and Weber, Jonathan and Idoumghar, Lhassane and Muller, Pierre-Alain},
  booktitle                = {International Conference On Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  Year                     = {2018}
}
```

## Ipynb version  of "Evaluating surgical skills from kinematic data using convolutional neural networks"

This section is added by @Hosseinhashemiir. 

This modified project of @https://github.com/hfawaz/miccai18
cause I don't have a good hardware (GPU) for running this code by directly, I used "Google's Colaboratory".

you don't need any setup for the run, just refer this Repo to Colab website http://colab.research.google.com and find the miccai18.ipynb.

*notice that processing need around 2 hours.*
