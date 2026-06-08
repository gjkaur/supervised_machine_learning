# Supervised Machine Learning — Course Materials

Beginner-friendly Jupyter notebooks based on **Step 1 through Step 4** of *Mastering Machine Learning with Python in Six Steps* (2nd ed.) by Manohar Swamynathan.

Each chapter has a **detailed README** that explains every topic from the book in plain language — use it as a study guide before or while working through the notebook.

---

## Course Roadmap

| Step | Chapter | Notebook | Detailed guide |
|------|---------|----------|----------------|
| 1 | Getting Started in Python 3 | [`Chapter_1_Getting_Started_in_Python3.ipynb`](Chapter_1_Getting_Started_in_Python3.ipynb) | [**README — Chapter 1**](README_Chapter_1.md) |
| 2 | Introduction to Machine Learning | [`Chapter_2_Introduction_to_Machine_Learning.ipynb`](Chapter_2_Introduction_to_Machine_Learning.ipynb) | [**README — Chapter 2**](README_Chapter_2.md) |
| 3 | Fundamentals of Machine Learning | [`Chapter_3_Fundamentals_of_Machine_Learning.ipynb`](Chapter_3_Fundamentals_of_Machine_Learning.ipynb) | [**README — Chapter 3**](README_Chapter_3.md) |
| 4 | Model Diagnosis and Tuning | [`Chapter_4_Model_Diagnosis_and_Tuning.ipynb`](Chapter_4_Model_Diagnosis_and_Tuning.ipynb) | [**README — Chapter 4**](README_Chapter_4.md) |

**Suggested order:** Complete chapters 1 → 2 → 3 → 4 sequentially. Each chapter builds on the previous one.

---

## What Each Chapter Covers

### [Chapter 1 — Python 3 Basics](README_Chapter_1.md)

Python setup (Anaconda), syntax, data types (list, tuple, set, dict), operators, control flow, functions, modules, file I/O, exception handling. **53 code listings (1-1 to 1-53).**

### [Chapter 2 — Introduction to ML](README_Chapter_2.md)

What ML is, AI history, Turing test, ANI/AGI/ASI, supervised/unsupervised/reinforcement learning, KDD/CRISP-DM/SEMMA frameworks, NumPy, pandas, matplotlib, scikit-learn overview. **49 code listings (2-1 to 2-49).**

### [Chapter 3 — ML Fundamentals](README_Chapter_3.md)

Measurement scales, feature engineering, EDA, linear/polynomial/multivariate regression, regularization, logistic regression, decision trees, SVM, kNN, ARIMA time series, k-means, hierarchical clustering, PCA. **56 code listings (3-1 to 3-56).**

### [Chapter 4 — Model Diagnosis & Tuning](README_Chapter_4.md)

Optimal probability cutoff, Type I/II errors, imbalanced data (RUS/ROS/SMOTE), bias–variance, k-fold & stratified CV, bagging, Random Forest, Extra Trees, AdaBoost, gradient boosting, XGBoost, voting, stacking, GridSearch, RandomSearch, Bayesian optimization, wavelet denoising. **25 code listings (4-1 to 4-25).**

---

## Data Files

CSV files for notebook exercises live in the [`Data/`](Data/) folder:

| File | Chapter |
|------|---------|
| `Grade_Set_1.csv`, `Grade_Set_2.csv`, `Grade_Set_1_Classification.csv` | 3 |
| `Housing_Modified.csv`, `LR_NonLinear.csv`, `TS.csv` | 3 |
| `Diabetes.csv`, `digit.csv`, `Temperature.csv` | 4 |

Iris and other datasets are loaded from scikit-learn where noted in the notebooks.

---

## Getting Started

1. Install [Anaconda](https://www.anaconda.com/) (Python 3) or ensure Python 3.8+ with Jupyter.
2. Clone this repository and open a notebook in Jupyter Lab, Jupyter Notebook, or VS Code.
3. Read the chapter README first, then run cells **top to bottom**.
4. Optional packages for Chapter 4: `imbalanced-learn`, `xgboost`, `bayesian-optimization`, `PyWavelets`.

```bash
pip install numpy pandas matplotlib scikit-learn statsmodels jupyter
pip install imbalanced-learn xgboost bayesian-optimization PyWavelets
```

---

## Repository Structure

```
Supervised_Machine_Learning/
├── README.md                          ← You are here (course overview)
├── README_Chapter_1.md                ← Detailed Chapter 1 study guide
├── README_Chapter_2.md                ← Detailed Chapter 2 study guide
├── README_Chapter_3.md                ← Detailed Chapter 3 study guide
├── README_Chapter_4.md                ← Detailed Chapter 4 study guide
├── Chapter_1_Getting_Started_in_Python3.ipynb
├── Chapter_2_Introduction_to_Machine_Learning.ipynb
├── Chapter_3_Fundamentals_of_Machine_Learning.ipynb
├── Chapter_4_Model_Diagnosis_and_Tuning.ipynb
└── Data/                              ← Exercise CSV files
```

---

## Book Reference

Swamynathan, M. (2019). *Mastering Machine Learning with Python in Six Steps* (2nd ed.). Apress.  
ISBN: 978-1-4842-4946-8

Steps 5 (Text Mining & Recommender Systems) and 6 (Deep & Reinforcement Learning) are covered in the book but not yet in this repository.

---

## License & Use

Course materials for CIMT College — Supervised Machine Learning. Notebooks include modern Python 3 / scikit-learn updates for runnable code alongside the book’s original listings.
