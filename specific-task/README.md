# Specific Task 2g

## 1. Overview of SSL Methods

* **SimCLR:** Utilizes a contrastive approach (NT-Xent loss) to maximize agreement between augmented views of the same image and minimize agreement with other images, using a ResNet-15 encoder and MLP projection head.
* **Barlow Twins:** Employs a redundancy reduction loss, forcing the cross-correlation matrix between augmented views' features (from ResNet-15 + projector) towards the identity matrix, promoting invariance and feature decorrelation without negative samples.

## 2. Implementation Summary

* **Dataset:** Unlabeled 8-channel HDF5 images (`Dataset_Specific_Unlabelled.h5`) for pre-training; Labeled 8-channel HDF5 images (`Dataset_Specific_labelled_full_only_for_2i.h5`) for fine-tuning (80/20 split).
* **Backbone:** ResNet-15 (adapted from ResNet-18 for 8-channel input).
* **Augmentations:** Random Horizontal/Vertical Flips, Random Cropping (112x112), Normalization.
* **Pre-training:** 10 epochs using Adam optimizer.
* **Fine-tuning:** Pre-trained ResNet-15 backbone (projector removed) with separate linear heads (`nn.Linear(512, 2)`) for classification (binary) and regression (`m`, `pT`). Trained with combined CrossEntropy + MSE loss and differential learning rates (backbone: `1e-5`, heads: `1e-4`).
* **Baseline:** VGG-11 (adapted for 8 channels) trained from scratch with a large MLP head (`Linear(512*3*3 -> 4096) -> ... -> Linear(4096 -> num_outputs)`).

## 3. Results Summary

The fine-tuning results showed distinct performance patterns. For classification, the Barlow Twins pre-trained model achieved 83.35% test accuracy, while the SimCLR pre-trained model achieved 64.65% accuracy and the scratch VGG baseline achieved 53%. However, for the regression task evaluated by test MSE, the scratch VGG baseline performed better, achieving MSE around 5900-6100, while the fine-tuned SimCLR model yielded an MSE of approximately 135k and the fine-tuned Barlow Twins model resulted in an MSE around 145k. The reason for scratch vgg performing better is stated below

## 4. Discussion & Analysis

The results show clear benefits of SSL pre-training for the classification task, with Barlow Twins and SimCLR significantly outperforming the scratch VGG baseline. This suggests SSL effectively learned transferable discriminative features from the unlabeled data.

For the regression task, the scratch VGG baseline yielded the lowest MSE, highlighting the impact of task-specific head architecture. The VGG's substantially larger MLP regression head  (`Linear(512*3*3 -> 4096) -> ... -> Linear(4096 -> num_outputs)`) likely offered greater capacity for mapping features to the continuous `m` and `pT` targets compared to the simpler linear head (`nn.Linear(512, 2)`) used with the fine-tuned ResNets.


