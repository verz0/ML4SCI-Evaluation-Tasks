# ML4SCI-Evaluation-Tasks
This repository contains the required notebooks and models for the tasks for application to GSoC 2025 for the CMS and End-to-End Deep Learning Projects @ ML4SCI Umbrella Organization:
1. **Common Task:**  
   Electron vs. Photon Classification using a ResNet-15 architecture.
2. **Specific Task 2g:**
   Next generation vision transformers for end to end mass regression and classification.
   Self-Supervised Pretraining using SimCLR and Barlow Twins on the provided 8-channel datasets, followed by fine-tuning for downstream tasks (classification and regression).


## Repository Structure

- **Common Task**  
  - **ResNet-15 Electron vs. Photon Classification:**  
    Notebook to train a ResNet-15 model for classifying electron events versus photon events. The resultant model is also added for your reference/evaluation

- **Specific Task**  
  - **SimCLR Implementation:**  
    A complete notebook that implements SimCLR for self-supervised pretraining on unlabelled data using a ResNet-15 backbone. The pretrained encoder is then fine-tuned for downstream tasks.
  - **Barlow Twins Implementation:**  
    A complete notebook that implements the Barlow Twins method, also using a ResNet-15 backbone for pretraining. After pretraining, the encoder is fine-tuned for the downstream tasks.
  - **Comparison with Scratch Models:**  
    For both the SimCLR and Barlow Twins pipelines, there is a comparison with model trained from scratch (a custom VGG-11 built from scratch) on the labelled dataset for both classification and regression.

PS: The models were developed in a kaggle environment. (16gb RAM + T4 GPU)
