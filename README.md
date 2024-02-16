# Human-Body-Pose-Estimation-Missing-Joints-Reconstruction-through-Interpolations-Comparison

## Overview
In this work, we focus on Human Pose Estimation (HPE), a field dedicated to determining the pose of a human body by estimating the 2D or 3D spatial position of its joints. HPE finds widespread use in sports contexts for understanding movements. Specifically, we apply HPE to classify Calisthenics isometric elements. For this purpose, we utilize OpenPose as the chosen pose estimator, configured specifically for 2D pose estimation.

The discipline of Calisthenics comprises various branches, including endurance, strength, and skills. In recent years, each of these branches has experienced significant growth in competitive arenas. Notably, the skills branch holds a particularly influential position on the international stage due to the demanding strength requirements of the performed poses. Calisthenics encompasses a wide range of skills and their variations. However, for this specific project, a carefully selected subset of skills has been chosen.

Estimating Missing Joint Spatial Information Through Interpolation
This work offers a comprehensive overview of estimating missing joint spatial information through six interpolation methods. Missing values can be due to various factors such as lighting conditions and background influences. Each interpolation will be thoroughly examined and utilized to estimate the missing values.

## Project Goals and Applications
The primary task is to recognize and classify Calisthenics skills from video footage. This work’s applications include monitoring athletes’ skill execution during competitions and serving as a training tool to simulate a judge’s role. To accomplish the project’s ultimate goal, several sequential steps must be completed to successfully address various subtasks.

The process begins with the selection of skills to recognize and the creation of a video dataset. Subsequently, a pre-trained pose estimator is utilized to extract spatial information about the athletes’ joints in the videos. Following this, a dataset is constructed from the frame-level body pose tracking results. A subset of this dataset is then used as reference data to evaluate the performance of different interpolation methods. The original keypoint dataset is filled with various interpolations, and all these datasets are utilized to train and test a multiclass classifier designed to detect poses.

## Interpolation Techniques and Evaluation
The effectiveness of different interpolation techniques on these sequences was evaluated, comparing them to ground truth values. Additionally, a Multilayer Perceptron was developed and its performance was assessed by training and testing it on different interpolated datasets.

## Results and Insights
Analysis of the results revealed a discrepancy between the performance of the interpolations compared to ground truth data and the testing results of the classifier. Interestingly, Linear Interpolation, despite its simplicity, demonstrated the best results in the initial phase, along with Pchip interpolation. In the testing phase of the multiclass classifier, the Inverse Distance Weight showed the second-best results after the dataset with no interpolations.

## Motivation and Contribution
We provided insight into these outcomes, emphasizing that the presence of missing values enhances the robustness of the network and helps prevent overfitting. This project serves as a side extension of the paper titled "Calisthenics Skills Temporal Segmentation," which is expected to be presented at VISAPP24.
