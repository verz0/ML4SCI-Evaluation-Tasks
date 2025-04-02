# Common task

## 1. Objective

The goal of this task was to train a ResNet-15 model to perform binary classification on 2-channel image representations of particle showers detected in the electromagnetic (ECAL) and hadronic (HCAL) calorimeters, distinguishing between electron and photon signatures.

## 2. Dataset

* **Source:** Provided HDF5 files containing separate datasets for single electrons and single photons (`SingleElectronPt50_IMGCROPS_n249k_RHv1.hdf5`, `SinglePhotonPt50_IMGCROPS_n249k_RHv1.hdf5`).
* **Format:** Each sample is represented as a 32x32 pixel image with 2 channels, corresponding to energy deposits in the ECAL and HCAL layers.
* **Labels:** Binary labels were used (e.g., 0 for electron, 1 for photon).
* **Preprocessing:** Electron and photon datasets were combined. The data was split into training (80%) and testing (20%) sets. Images were transposed to the PyTorch format (`[N, C, H, W]`, i.e., `[N, 2, 32, 32]`) and converted to float tensors.

## 3. Implementation Details

* **Model Architecture:** A custom ResNet-15 architecture was implemented in PyTorch.
    * Inspired by the standard ResNet design, using `BasicBlock` modules (two 3x3 convolutions with BatchNorm and ReLU).
    * The initial `nn.Conv2d` layer was modified to accept `in_channels=2`.
    * The network consisted of an initial convolution/BatchNorm/ReLU layer followed by three stages of residual blocks (`[2, 2, 2]` blocks per stage), with downsampling (stride=2) occurring at the start of stages 2 and 3.
    * An `nn.AdaptiveAvgPool2d((1, 1))` layer followed the residual blocks.
    * A final `nn.Linear` layer mapped the pooled features to the 2 output classes.
* **Data Handling:** Standard PyTorch `Dataset` and `DataLoader` classes were used for efficient batching and loading during training and testing.
* **Training:**
    * **Loss Function:** `nn.CrossEntropyLoss` was used for the binary classification task.
    * **Optimizer:** `torch.optim.Adam` with a learning rate of 0.001.
    * **Epochs:** The model was trained for 20 epochs.

## 4. Results

The trained ResNet-15 model was evaluated on the held-out test set:

* **Test Accuracy:** **74.06%**
* **Test AUC (Area Under ROC Curve):** **0.8099**

Training progress was monitored by plotting loss and accuracy curves for both training and testing sets over the epochs. An ROC curve was also generated to visualize the classifier's performance across different thresholds.

## 5. Conclusion

This task successfully demonstrated the implementation and training of a ResNet-15 model for classifying electron versus photon showers based on 2-channel calorimeter images. The achieved accuracy and AUC provide a baseline performance measure for this classification task using a standard CNN architecture on this dataset. The workflow established proficiency in handling the HDF5 data format, implementing custom CNN architectures in PyTorch, and performing standard training and evaluation procedures.
