# Object Segmentation from Unlabeled Video Frames based on Deep Learning Methods

## Contents
-   [Collaborators](#collaborators)
-   [Introduction](#introduction)
-   [Dataset](#dataset)

## Collaborators
-   Jerry Peng
-   Joseph Edell
-   Aman Singhal

## Introduction
In this research project, we proposed three different ways to segment object from video frames based on deep learning methods.

## Dataset
The dataset has the following structure:
-   13,000 unlabeled videos with 22 frames each,
-   1,000 labeled training videos with 22 frames each,
-   1,000 labeled validation videos with 22 frames each.

### Video
The dataset features synthetic videos with simple 3D shapes that interact with each other according to basic physics principles. Objects in videos have three shapes (cube, sphere, and cylinder), two materials (metal and rubber), and eight colors (gray, red, blue, green, brown, cyan, purple, and yellow). In each video, there is no identical objects, such that each combination of the three attributes uniquely identifies one object.

For unlabeled, training and validation set, we have all 22 frames for each video. For hidden set we only have the first 11 frames of each video.

### Label
For training set and validation set, we have the full semantic segmentation mask for each frame.

### Task
The task on hidden set is to use the first 11 frames to generate the semantic segmentation mask of the last frame (the 22nd frame). The performance is evaluated by calculate the IOU between the ground truth and the generated mask.