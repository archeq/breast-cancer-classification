# Breast Cancer Classification

A neural network classifier built with **PyTorch** for the [Wisconsin Breast Cancer dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html). The model distinguishes between **benign** and **malignant** tumours using 30 numeric features computed from digitised images of fine-needle aspirates.




The data is split into **training (80 %)**, **validation (10 %)**, and **test (10 %)** sets with `StandardScaler` normalisation fitted on the training split only (no data leakage).

## Model Architecture

A simple fully-connected feed-forward network:

| Layer | Details |
|---|---|
| Input | 30 features |
| Hidden | 16 neurons, ReLU activation |
| Output | 2 classes (CrossEntropyLoss) |

**Optimiser:** Adam (lr = 0.01)  
**Epochs:** 100

## Getting Started

### Prerequisites

- Python 3.9+

### Installation

```bash
git clone https://github.com/<your-username>/breast-cancer-classification.git
cd breast-cancer-classification
pip install -r requirements.txt
```

### Running the Notebook

Open and run all cells in `breast_cancer_classification.ipynb` using Jupyter Notebook / JupyterLab or any compatible IDE (e.g. PyCharm, VS Code).

```bash
jupyter notebook breast_cancer_classification.ipynb
```

## Results

The notebook evaluates the model on a held-out test set and reports:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**
- **Full Classification Report** (per-class precision / recall / F1)

## Visualisations

| Plot | Description |
|---|---|
| **Treemap** | Shows the train / validation / test split sizes |
| **Learning Curve** | Training loss, validation loss, and validation accuracy over epochs |

## Key Dependencies

| Package | Purpose |
|---|---|
| `torch` | Neural network framework |
| `scikit-learn` | Dataset, preprocessing, metrics |
| `matplotlib` | Plotting |
| `squarify` | Treemap visualisation |
| `pandas` / `numpy` | Data handling |

See `requirements.txt` for the full pinned dependency list.


