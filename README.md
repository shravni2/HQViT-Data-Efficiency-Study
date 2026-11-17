# Data-Efficient Image Classification Using a Hybrid Quantum-Classical Vision Transformer

A research project investigating the use of quantum circuits in Vision Transformer (ViT) architectures for image classification, with particular emphasis on low-dimensional feature representations and the effects of quantum feature transformations on classification performance.

## Overview

This project explores a hybrid quantum-classical approach to image classification using the CIFAR-10 dataset.

The central idea is to investigate whether a parameterized quantum circuit can transform a compressed visual representation into useful features for downstream classification.

The study compares three architectures:

1. **Classical Vision Transformer (ViT):** Uses a 128-dimensional visual representation for classification.
2. **Classical Bottleneck:** Compresses the ViT representation from 128 dimensions to 4 dimensions using a classical projection.
3. **Hybrid Quantum Vision Transformer (HQViT):** Uses the same four-dimensional projection, followed by a parameterized quantum circuit before classification.

A standalone CNN baseline is also being added to broaden the comparison.

The objective is not to assume that quantum integration improves performance, but to experimentally investigate its benefits, limitations, and optimization behaviour.

## Research Objectives

- Investigate quantum feature transformations in Vision Transformer architectures.
- Study the effects of dimensionality reduction on image classification.
- Compare quantum and classical transformations under a shared experimental setup.
- Examine gradient propagation through parameterized quantum circuits.
- Analyse class-wise performance and classification errors.
- Evaluate whether quantum feature processing preserves discriminative information.

## Dataset

**CIFAR-10**

The dataset contains 60,000 colour images across 10 classes.

| Dataset partition | Images |
|---|---:|
| Training | 45,000 |
| Validation | 5,000 |
| Testing | 10,000 |
| Total | 60,000 |

Image dimensions: 32 × 32 × 3

Number of classes: 10

Classes:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

The test set is kept separate from model training and validation-based checkpoint selection.

## Model Architectures

### 1. Classical Vision Transformer

The classical ViT serves as the full-dimensional transformer baseline.

Architecture:

- Convolutional patch embedding
- 128-dimensional embedding representation
- Four transformer encoder layers
- Four attention heads
- CLS token representation
- Layer normalization
- Ten-class classification head

The 128-dimensional CLS representation is directly used for classification.

### 2. Classical Bottleneck

This model investigates the effect of reducing the ViT representation before classification.

Architecture:

- ViT feature extractor
- 128-dimensional CLS representation
- Layer normalization
- Linear projection: 128 → 4
- Classical classifier: 4 → 32 → 10

This architecture provides a classical reference for evaluating the quantum transformation.

### 3. Hybrid Quantum-Classical Vision Transformer

HQViT extends the classical bottleneck architecture with a parameterized quantum circuit.

Architecture:

- ViT feature extractor
- 128-dimensional CLS representation
- Layer normalization
- Linear projection: 128 → 4
- Four-qubit parameterized quantum circuit
- Quantum expectation-value measurements
- Classification head

The quantum circuit uses trainable rotation operations and entanglement through CNOT gates. Pauli-Z expectation values provide the quantum output features.

The resulting four-dimensional representation is passed to the classifier.

### 4. CNN Baseline

A standalone convolutional neural network is being added as an additional classical baseline.

The planned architecture uses:

- Three convolutional blocks
- Batch normalization
- ReLU activations
- Max pooling
- Adaptive global average pooling
- Fully connected classification layers

Its training and evaluation results will be added after the experiment is completed.

## Experimental Configuration

The matched three-model experiment uses the following configuration:

| Parameter | Configuration |
|---|---|
| Dataset | CIFAR-10 |
| Training samples | 45,000 |
| Validation samples | 5,000 |
| Test samples | 10,000 |
| Batch size | 32 |
| Training epochs | 10 |
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Evaluation metric | Accuracy, Precision, Recall, Macro F1 |

The models use shared initial ViT backbone weights. The Classical Bottleneck and HQViT also share their initial projection and classifier weights.

Best checkpoints are selected using validation accuracy.

**Reproducibility note:** The Classical Bottleneck training was interrupted after its first epoch and resumed from its saved weights with a newly initialized optimizer. Its optimizer state was not restored. This recovery detail should be considered when interpreting comparisons.

## Experimental Results

### Test-set comparison

The following results were obtained using the held-out CIFAR-10 test set.

| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Classical ViT | **64.16%** | 0.6453 | 0.6416 | **0.6367** |
| Classical Bottleneck | 60.47% | 0.6042 | 0.6047 | 0.5991 |
| HQViT | 54.38% | 0.5340 | 0.5438 | 0.5276 |

### Initial observations

- Classical ViT achieved the highest test accuracy.
- Reducing the ViT representation to four dimensions resulted in a performance reduction.
- HQViT performed below the equivalent classical bottleneck.
- The current quantum feature transformation did not demonstrate a classification improvement over its classical counterpart.

These findings suggest that the current low-dimensional quantum transformation does not preserve sufficient discriminative information to outperform the classical bottleneck under the evaluated configuration.

They do not establish that quantum machine learning is inherently inferior to classical methods.

## Class-wise Analysis

The per-class evaluation revealed substantial differences between the models.

One notable observation was the low cat-class recall of HQViT.

| Metric | Classical ViT | Classical Bottleneck | HQViT |
|---|---:|---:|---:|
| Cat recall | 36.2% | 40.1% | 10.7% |
| Cat F1-score | 0.3998 | 0.4111 | 0.1584 |

HQViT also exhibited confusion between visually similar categories, particularly:

- Cat and dog
- Deer and horse
- Bird and deer

These observations motivate further investigation into representation compression and class-discriminative feature preservation.

## Quantum Gradient Diagnostics

Gradient-flow diagnostics were conducted on the saved HQViT checkpoint using a training batch.

| Component | Total gradient norm |
|---|---:|
| ViT backbone | 21.1875 |
| Projection | 2.1829 |
| Quantum circuit | 0.2147 |
| Classifier | 0.7754 |

All 58 trainable parameter tensors received nonzero gradients during this diagnostic pass.

This confirms that the quantum circuit received a learning signal for the examined batch. It does not, by itself, establish the absence of barren plateaus or prove that the learned quantum representation is useful.

## Repository Structure

```text
HQViT-Data-Efficiency-Study/
│
├── HQViT_Clean_Restart.ipynb
│
├── models_clean/
│   ├── final_hqvit/
│   │   └── rebuilt_hqvit_final.pt
│   │
│   ├── section108/
│   │   ├── classical_bottleneck_initial.pt
│   │   ├── classical_vit_initial.pt
│   │   └── hqvit_initial.pt
│   │
│   ├── section109/
│   │   ├── Classical_ViT_best.pt
│   │   └── Classical_Bottleneck_best.pt
│   │
│   └── section110/
│       ├── Classical_Bottleneck_best.pt
│       ├── Classical_Bottleneck_latest.pt
│       ├── HQViT_best.pt
│       └── HQViT_latest.pt
│
├── results/
│   ├── section109/
│   ├── section110/
│   ├── final_test_evaluation/
│   └── research_analysis/
│
├── README.md
└── .gitignore
```

The CNN checkpoint and its result files will be added under `models_clean/cnn_baseline/` and `results/cnn_baseline/` after training.

## Technologies Used

- Python
- PyTorch
- Vision Transformers
- PennyLane
- Quantum Neural Networks
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook
- GitHub Codespaces

## Running the Project

1. Clone the repository:

```bash
git clone https://github.com/shravni2/HQViT-Data-Efficiency-Study.git
cd HQViT-Data-Efficiency-Study
```

2. Create and activate a Python environment.

3. Install the required dependencies:

```bash
pip install torch torchvision pennylane numpy pandas matplotlib scikit-learn jupyter
```

4. Open the notebook:

```bash
jupyter notebook HQViT_Clean_Restart.ipynb
```

5. Execute the notebook setup and dataset preparation cells in order.

6. Run the desired model training and evaluation sections.

**Note:** The notebook is the primary experimental implementation. Exact package versions and hardware requirements should be recorded alongside future reproducibility releases.

## Current Limitations

- The current comparison is based on a single completed training run per model.
- The four-dimensional bottleneck may discard useful visual information.
- The quantum circuit has only been investigated in the current four-qubit configuration.
- Gradient diagnostics were conducted on a single batch.
- The Classical Bottleneck optimizer was reinitialized after an interrupted training session.
- The CNN baseline and its evaluation are pending.
- Multiple independent runs are required to establish statistical reliability.

## Future Work

1. Complete the CNN baseline and evaluate it on the same test set.
2. Investigate alternative quantum feature encoding strategies.
3. Study different quantum circuit depths and entanglement structures.
4. Analyse quantum feature distributions and parameter sensitivity.
5. Investigate class-wise representation collapse.
6. Conduct multiple independent training runs.
7. Evaluate alternative bottleneck dimensions.
8. Investigate hybrid architectures that preserve more classical information while incorporating quantum features.

## Research Direction

This project investigates the practical limitations of integrating parameterized quantum circuits into transformer-based image classification systems.

Rather than assuming a quantum advantage, the study focuses on controlled comparisons, feature compression, optimization behaviour, and empirical evaluation.

The broader objective is to understand when quantum feature transformations can be useful in hybrid machine learning architectures and what architectural constraints may prevent them from improving classical models.

---

**Project:** Data-Efficient Image Classification Using a Hybrid Quantum-Classical Vision Transformer  
**Author:** Shravni Kale  
**Repository:** [HQViT-Data-Efficiency-Study](https://github.com/shravni2/HQViT-Data-Efficiency-Study)
