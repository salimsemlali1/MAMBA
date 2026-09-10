# Autism-Spectrum-Disorder-Detection-Using-MAMBA-State-Space-Models
#  ABIDE fMRI — Mamba vs LSTM vs Transformer

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?style=for-the-badge&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Neuroimaging-fMRI-purple?style=for-the-badge" alt="fMRI">
  <img src="https://img.shields.io/badge/Dataset-ABIDE--I-green?style=for-the-badge" alt="ABIDE-I">
  <img src="https://img.shields.io/badge/Task-ASD%20Classification-orange?style=for-the-badge" alt="ASD Classification">
</p>

<p align="center">
  <b>Temporal Modeling of Brain Connectomes for Autism Spectrum Disorder Classification</b>
</p>

---

##  Overview

This project investigates **deep learning approaches for Autism Spectrum Disorder (ASD) classification** using resting-state functional MRI (**rs-fMRI**) data from the **ABIDE-I dataset**.

The main objective is to model the temporal dynamics of brain activity and evaluate whether **State Space Models (Mamba)** can provide competitive performance compared with more traditional sequential architectures such as **LSTM**, as well as Transformer-based approaches.

The complete pipeline covers:

*  ABIDE-I fMRI data acquisition
*  ROI extraction and preprocessing
*  Functional connectivity estimation
*  Temporal sequence modeling
*  Mamba architecture implemented in PyTorch
*  5-fold stratified cross-validation
*  Anti-overfitting techniques
*  Accuracy and ROC-AUC evaluation
*  Mamba vs LSTM comparison
*  Training and performance visualization

---

##  Research Objective

The project addresses a binary classification problem:

> **Given resting-state fMRI data, can temporal deep learning models distinguish individuals with ASD from typically developing controls?**

The two target classes are:

| Label       | Description                  |
| ----------- | ---------------------------- |
| **ASD**     | Autism Spectrum Disorder     |
| **Control** | Typically Developing Control |

The dataset contains **653 subjects**, including **307 ASD subjects** and **346 control subjects**.

---

##  Dataset

The project uses **ABIDE-I (Autism Brain Imaging Data Exchange I)** through the Nilearn interface.

### Dataset configuration

```text
Dataset       : ABIDE-I
Subjects      : 653
Pipeline      : CPAC
Atlas         : CC200
Band-pass     : Enabled
Global Signal : Regression enabled
Quality check : Enabled
```

The data are obtained using Nilearn's `fetch_abide_pcp` interface, with ROI time series extracted using the **CC200 atlas**.

---

##  Processing Pipeline

```text
                 ABIDE-I Dataset
                        │
                        ▼
              ┌──────────────────┐
              │   fMRI Preprocess │
              │      CPAC + GSR   │
              └─────────┬────────┘
                        │
                        ▼
                ROI Time Series
                     CC200
                        │
                        ▼
              Functional Connectivity
                        │
                        ▼
              Temporal Representation
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
       Mamba          LSTM       Transformer
          │             │             │
          └─────────────┼─────────────┘
                        ▼
                ASD Classification
                        │
                        ▼
             Accuracy / ROC-AUC
```

---

##  Model Architecture

### Mamba

The main model is a custom **Mamba-based architecture designed for fMRI temporal sequences**.

The architecture projects ROI sequences into a latent representation and processes them through stacked Mamba layers. Temporal representations are aggregated using both **mean pooling** and **max pooling** before classification.

The implementation also supports the integration of functional connectivity features into the final representation.

### LSTM

An LSTM-based model is implemented as a classical recurrent baseline using the same general training and evaluation framework.

This allows a controlled comparison between:

* State Space Models
* Recurrent Neural Networks
* Transformer-based architectures

---

##  Anti-Overfitting Strategy

Because neuroimaging datasets can be relatively small compared with model complexity, several regularization techniques are incorporated.

### Training techniques

* **Mixup augmentation**
* **Label smoothing**
* **Gradient clipping**
* **Dropout**
* **Drop Path**
* **Weight decay**
* **Early stopping**
* **Learning-rate scheduling**
* **Class-weighted loss**

For example, the Mamba configuration uses:

```python
d_model = 32
d_state = 16
n_layers = 2

dropout = 0.5
drop_path_rate = 0.2
weight_decay = 5e-2
label_smoothing = 0.15
mixup_alpha = 0.3

learning_rate = 5e-5
batch_size = 32
epochs = 100
patience = 20
folds = 5
```

These settings are explicitly defined in the notebook's cross-validation training configuration.

---

##  Cross-Validation

The experiments use **5-fold stratified cross-validation** to obtain a more robust estimate of model performance.

For each fold:

1. Split the subjects into training and validation sets.
2. Standardize connectivity features using the training data.
3. Train the model.
4. Evaluate on the validation fold.
5. Apply early stopping based on validation loss.
6. Store the best validation AUC.
7. Aggregate the results across folds.

The implementation uses `StratifiedKFold` with `shuffle=True` and `random_state=42`.

---

##  Results

### Mamba vs LSTM

The obtained cross-validation results are:

|   Fold   |  Mamba AUC |  LSTM AUC  | Δ Mamba − LSTM |
| :------: | :--------: | :--------: | :------------: |
|     1    |   0.8205   |   0.7856   |     +0.0348    |
|     2    |   0.8518   |   0.8691   |     −0.0173    |
|     3    |   0.7979   |   0.7534   |     +0.0445    |
|     4    |   0.7589   |   0.7403   |     +0.0185    |
|     5    |   0.7550   |   0.7463   |     +0.0088    |
| **Mean** | **0.7968** | **0.7789** |   **+0.0179**  |
|  **Std** | **0.0368** | **0.0477** |        —       |

The notebook reports an average Mamba AUC of approximately **0.797 ± 0.037**, compared with **0.779 ± 0.048** for LSTM.

### Key observation

> **Mamba achieves a higher mean ROC-AUC than LSTM across the five folds, with an improvement of approximately 1.79 percentage points.**

The results also show that LSTM performs better than Mamba on Fold 2, demonstrating that model performance varies across validation splits.

---

##  Model Complexity

The implemented models have the following number of trainable parameters:

| Model     | Parameters |
| --------- | ---------: |
| **Mamba** |     29,090 |
| **LSTM**  |     21,506 |

Mamba therefore uses a moderately larger parameter budget while achieving a higher mean AUC in this experiment.

---

##  Evaluation Metrics

The project evaluates the models using:

* **Accuracy**
* **ROC-AUC**
* **Validation Loss**
* **Training Loss**
* **Training Accuracy**
* **Validation Accuracy**
* **Fold-wise performance**

ROC-AUC is used as an important metric because it provides a threshold-independent measure of the model's ability to distinguish ASD from Control subjects.

---

##  Visualizations

The notebook includes visualization of:

* Training and validation loss
* Training and validation accuracy
* Validation ROC-AUC
* Fold-wise performance
* Model comparison
* Mamba vs LSTM performance

These visualizations make it possible to analyze both model convergence and generalization behavior.

---

##  Technologies

The project is implemented using:

| Technology       | Purpose                                |
| ---------------- | -------------------------------------- |
| **Python**       | Main programming language              |
| **PyTorch**      | Deep learning and model implementation |
| **Nilearn**      | Neuroimaging processing                |
| **NiBabel**      | Neuroimaging data handling             |
| **Scikit-learn** | Cross-validation and evaluation        |
| **NumPy**        | Numerical computation                  |
| **Pandas**       | Data manipulation                      |
| **Matplotlib**   | Visualization                          |
| **Seaborn**      | Statistical visualization              |
| **tqdm**         | Progress monitoring                    |
| **Einops**       | Tensor manipulation                    |

The notebook explicitly installs and imports the main scientific, neuroimaging and deep-learning dependencies.

---

##  Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/ABIDE-Mamba.git
cd ABIDE-Mamba
```

Install the dependencies:

```bash
pip install nilearn nibabel scikit-learn matplotlib seaborn tqdm einops torch torchvision
```

---

##  Usage

The main experiment is contained in:

```text
ABIDE_Mamba_vs_LSTM_vs_Transformer.ipynb
```

Open the notebook with Jupyter:

```bash
jupyter notebook
```

or use **Google Colab**.

Then execute the notebook sequentially:

```text
1. Install dependencies
        ↓
2. Download ABIDE-I
        ↓
3. Explore the dataset
        ↓
4. Extract ROI time series
        ↓
5. Build connectivity representations
        ↓
6. Define Mamba
        ↓
7. Train with cross-validation
        ↓
8. Evaluate Mamba
        ↓
9. Train baseline models
        ↓
10. Compare results
```

---

##  Repository Structure

```text
ABIDE-Mamba/
│
├──  ABIDE_Mamba_vs_LSTM_vs_Transformer.ipynb
│
├──  abide_data/
│   └── ABIDE-I data
│
├──  results/
│   ├── figures/
│   └── metrics/
│
├──  models/
│   └── trained models
│
├──  README.md
└──  requirements.txt
```

---

##  Research Perspective

This project explores the potential of **State Space Models for neuroimaging-based classification**.

Compared with recurrent architectures, Mamba provides an alternative mechanism for modeling long sequential dependencies while maintaining a relatively compact architecture.

For the present experiment, Mamba achieved a mean AUC of **0.7968**, compared with **0.7789** for LSTM.

Future experiments could investigate:

* Larger Mamba configurations
* Different brain atlases
* Multi-site harmonization
* Site-aware cross-validation
* Transformer architectures
* Graph Neural Networks
* Hybrid Mamba + GNN models
* Explainable AI for neuroimaging
* External validation on independent cohorts

---

##  Author

**Salim Semlali**

Master's-level research project focused on:

> **Deep Learning • Neuroimaging • State Space Models • Brain Connectivity • Autism Spectrum Disorder**

**Supervisor:** Prof. Ahmed Drissi El Maliani

---

##  Acknowledgments

This project uses the **ABIDE-I dataset** and relies on the open-source neuroimaging ecosystem provided by **Nilearn**, together with the PyTorch deep learning framework.

---

##  License

This project is intended for **research and educational purposes**.

Please refer to the original ABIDE dataset terms and the licenses of the external libraries used in this project before redistributing data or derivative resources.

---

<p align="center">
  <b>  Exploring Brain Connectivity with Modern Deep Learning</b>
  <br>
  <sub>ABIDE-I • fMRI • Mamba • LSTM • Transformer • PyTorch</sub>
</p>
