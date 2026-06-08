# Chapter 4: Model Diagnosis and Tuning

**Notebook:** [`Chapter_4_Model_Diagnosis_and_Tuning.ipynb`](Chapter_4_Model_Diagnosis_and_Tuning.ipynb)  
**Source:** *Mastering Machine Learning with Python in Six Steps* (2nd ed.) — Manohar Swamynathan, Step 4  
**Code listings:** 4-1 through 4-25 (25 examples)  
**Prerequisite:** [Chapter 3 — Fundamentals of ML](README_Chapter_3.md)

---

## Who This Chapter Is For

You can train basic models from Chapter 3. This chapter teaches **how to diagnose problems**, **handle imbalanced data**, **validate reliably**, **combine models**, **tune hyperparameters**, and **denoise sensor data** — the skills that separate toy notebooks from production-ready ML.

---

## What You Will Learn

- Choose the optimal probability cutoff (not always 0.5)
- Understand Type I vs Type II errors and business cost
- Handle imbalanced datasets with RUS, ROS, and SMOTE
- Apply k-fold and stratified cross-validation
- Build ensemble models: bagging, Random Forest, Extra Trees, boosting, XGBoost
- Combine models with voting and stacking
- Tune hyperparameters with GridSearch, RandomSearch, and Bayesian optimization
- Denoise IoT time-series data with wavelet transforms

---

## Primary Dataset

Most examples use the **Pima Indians Diabetes** dataset (`Data/Diabetes.csv`) from the UCI repository:

- **768 patients**, **8 medical features** (glucose, BMI, age, etc.)  
- **Target:** binary — tested positive for diabetes (1) or not (0)  
- Naturally **imbalanced** (~65% negative, ~35% positive) — ideal for cutoff and resampling lessons  

Additional data:

| File | Purpose |
|------|---------|
| `Data/digit.csv` | Handwritten digit features for ensemble / tuning demos |
| `Data/Temperature.csv` | IoT sensor readings for wavelet denoising |

---

## Topics Covered (Complete Chapter Outline)

### Optimal Probability Cutoff Point

Logistic regression outputs a **probability** (0 to 1). Default rule: classify as positive if probability ≥ **0.5**.

That default is not always best — especially when classes are imbalanced or errors have different costs.

#### Listing 4-1 — Load Data and Check Class Distribution

Inspect how many positive vs negative cases exist. Fractions should sum to 1.

#### Listing 4-2 — Build Logistic Regression and Evaluate

Train a baseline model; report accuracy and confusion matrix at threshold 0.5.

#### Listing 4-3 — Find Optimal Cutoff Point

Plot **TPR (true positive rate)** and **1 − FPR** vs threshold. The **crossing point** suggests a better cutoff than 0.5 for balanced sensitivity/specificity.

#### Listing 4-4 — Reusable Optimal Cutoff Function

Encapsulate the search into a function you can reuse on any binary classifier with `predict_proba()`.

**Beginner insight:** Accuracy alone can hide poor performance on the minority class. Adjusting the cutoff trades off false positives vs false negatives.

---

### Which Error Is Costly? (Figure 4-1)

| Error | Name | Meaning | Example (diabetes) |
|-------|------|---------|---------------------|
| **Type I** | False Positive (FP) | Predict positive, actually negative | Healthy patient flagged as diabetic |
| **Type II** | False Negative (FN) | Predict negative, actually positive | Diabetic patient missed |

**Business context decides priority:**

- Cancer screening → minimize **FN** (don’t miss disease) even if FP rises  
- Spam filter → may tolerate some FN but minimize **FP** (don’t lose real emails)  

Choose metrics and cutoff based on **cost of mistakes**, not accuracy alone.

---

### Rare Event or Imbalanced Dataset (Figure 4-2)

When one class is much rarer (fraud, disease, failure), models may achieve high accuracy by always predicting the majority class — useless in practice.

#### Listing 4-5 — Create and Visualize Imbalanced Data

Use `make_classification()` to simulate extreme imbalance (~10% positive). Apply three **resampling** techniques:

| Technique | Abbreviation | What it does |
|-----------|--------------|--------------|
| **Random Under-Sampling** | RUS | Remove majority-class rows until balanced |
| **Random Over-Sampling** | ROS | Duplicate minority-class rows |
| **SMOTE** | Synthetic Minority Over-sampling | Create synthetic minority examples between neighbors |

**Caution:** RUS discards data; ROS can overfit duplicates; SMOTE needs the `imbalanced-learn` package.

#### Listing 4-6 — Compare Models After Resampling

Train logistic regression on original vs resampled data; compare recall, precision, and confusion matrices.

#### Which Resampling Technique Is the Best?

No universal winner — depends on dataset size, imbalance ratio, and which error is costlier. Always validate on **held-out real data**, not the resampled training set alone.

---

### Bias and Variance (Figure 4-3)

Two sources of prediction error:

| Concept | Symptom | Cause | Fixes |
|---------|---------|-------|-------|
| **Bias** | Underfitting | Model too simple | More features, complex model, lower regularization |
| **Variance** | Overfitting | Model too complex | More data, regularization, simpler model, ensembles |

**Bias–variance trade-off:** reducing one often increases the other. Goal = balance that generalizes to new data.

---

### Cross-Validation

A single train/test split can be **lucky or unlucky**. Cross-validation gives a more stable performance estimate.

#### K-Fold Cross-Validation (Listing 4-7, Figure 4-4)

Split data into **k** folds. For each fold:

1. Train on k−1 folds  
2. Test on the held-out fold  
3. Record metric  

Report **mean and std** of scores across folds.

#### Stratified K-Fold Cross-Validation (Listing 4-8)

Like k-fold but **preserves class proportions** in each fold — critical for imbalanced classification.

#### Plot ROC by Fold (Listing 4-9)

Overlay ROC curves from each fold to visualize stability of ranking performance.

**Beginner insight:** Cross-validation helps **compare models** and **detect overfitting**; it is not a substitute for a final untouched test set before deployment.

---

### Ensemble Methods — Overview

Combine multiple models for better, more stable predictions. Two main families:

| Family | Idea | Examples |
|--------|------|----------|
| **Bagging** | Train many models on bootstrap samples, average votes | BaggingClassifier, Random Forest, Extra Trees |
| **Boosting** | Train models sequentially, each correcting previous errors | AdaBoost, Gradient Boosting, XGBoost |

---

### Bagging (Bootstrap Aggregation) (Figures 4-5, 4-6)

1. Draw **N bootstrap samples** (random rows with replacement)  
2. Train a model on each sample  
3. **Aggregate** predictions (vote for classification, average for regression)  

Reduces **variance** — especially helps unstable models like single decision trees.

#### Listing 4-10 — Decision Tree vs Bagging

Compare standalone tree vs `BaggingClassifier` — bagging usually improves test accuracy.

#### Listing 4-11 — Feature Importance Function

Visualize which features the tree ensemble uses most (`feature_importances_`).

#### Listing 4-12 — Random Forest Classifier

Random Forest = bagging + **random feature subset** at each split. Strong default for tabular data.

#### Listing 4-13 — Extremely Randomized Trees (ExtraTree)

Extra randomness in split thresholds — often faster, sometimes better generalization.

#### Listing 4-14 — Plot Decision Boundaries

Project data to 2 PCA components and visualize how each ensemble separates classes.

#### Bagging — Essential Tuning Parameters

| Parameter | Effect |
|-----------|--------|
| `n_estimators` | Number of trees |
| `max_features` | Features considered per split |
| `max_depth` | Tree depth limit |
| `n_jobs` | Parallel cores (-1 = all) |

---

### Boosting (Figures 4-7 through 4-12)

Sequential ensemble: each new model focuses on **misclassified** points from earlier rounds.

#### Example Illustration for AdaBoost

Walk through iterations 1–3 on a toy 10-point dataset (Figures 4-8 – 4-11): misclassified points get higher weight; the final model is a weighted vote of weak learners.

#### Listing 4-15 — Decision Tree vs AdaBoost

AdaBoost often lifts test accuracy ~9%+ over a single shallow tree on diabetes data.

#### Gradient Boosting (Listing 4-16)

Fits each new tree to the **residual errors** (gradient) of the combined model — foundation for modern tabular ML.

#### Boosting — Essential Tuning Parameters

| Parameter | Effect |
|-----------|--------|
| `n_estimators` | Number of boosting stages |
| `learning_rate` | Shrink each tree’s contribution |
| `max_depth` | Complexity of each base learner |

Too many trees + high learning rate → overfitting.

#### XGBoost — eXtreme Gradient Boosting (Listings 4-17 – 4-18)

Optimized gradient boosting with regularization, parallel training, and missing-value handling.

- Listing 4-17 — sklearn-compatible `XGBClassifier` wrapper  
- Listing 4-18 — native `xgboost` API with `DMatrix`  

Key advantages: speed, regularization term, handles sparse data, early stopping.

---

### Ensemble Voting — Machine Learning’s Biggest Heroes United (Figure 4-13)

Train **diverse** models (logistic regression, random forest, SVM, kNN, decision tree, AdaBoost, gradient boosting) and combine their predictions.

#### Listing 4-19 — Individual Model Comparison

Cross-validated accuracy for each stand-alone model.

#### Hard Voting vs Soft Voting (Listing 4-20, Table 4-1)

| Method | Rule |
|--------|------|
| **Hard voting** | Majority class vote |
| **Soft voting** | Average predicted probabilities, then argmax |

Soft voting often works better when base models output calibrated probabilities. Requires `predict_proba` support.

---

### Stacking (Figure 4-14)

**Stacked generalization (Wolpert, 1992):**

1. Train **Level-0** models on training data  
2. Use their predictions as **new features**  
3. Train a **Level-1 meta-learner** (e.g., logistic regression) on those predictions  

Listing 4-21 implements level-2 stacking with multiple base classifiers and a meta-model.

**Caution:** Stacking can overfit if meta-model is trained on the same predictions it saw during training — use cross-validated stacking (as in sklearn / mlxtend) in production.

---

### Hyperparameter Tuning

Models have **parameters learned from data** (coefficients) and **hyperparameters set by you** (tree depth, learning rate, number of estimators). Tuning = finding hyperparameter values that maximize validation performance.

#### GridSearch (Listing 4-22)

Exhaustively try every combination from a parameter grid.

**Pros:** thorough within the grid  
**Cons:** exponential cost — 5 params × 5 values × 5-fold CV = thousands of fits; misses optima between grid points  

Use `GridSearchCV` with a cross-validation strategy.

#### RandomSearch (Listing 4-23)

Sample random combinations from parameter distributions.

Often finds **nearly optimal** settings in far less time — recommended when the search space is large.

#### Bayesian Optimization (Listing 4-24)

Build a probabilistic model of “parameter → score” and intelligently pick the next points to try (`bayesian-optimization` package).

Most efficient for **expensive** models or large spaces — learns from prior evaluations.

| Method | Best when |
|--------|-----------|
| GridSearch | Small grid, need reproducibility |
| RandomSearch | Medium/large space, limited budget |
| Bayesian | Expensive training, many continuous params |

---

### Noise Reduction for Time-Series IoT Data

Industrial sensors (temperature, vibration, pressure) produce noisy signals. Clean signals improve anomaly detection and forecasting.

#### Wavelet Transform (Listing 4-25)

**Wavelets** analyze signal at multiple time–frequency scales. Process:

1. Decompose signal into approximation + detail coefficients  
2. Threshold or shrink noise coefficients  
3. Reconstruct denoised signal  

Uses `PyWavelets` (`pywt`) on `Data/Temperature.csv`.

**Beginner insight:** Moving averages smooth but blur sharp changes; wavelets preserve edges while removing noise — useful for IoT edge devices.

---

## Code Listings Summary

| Listings | Topic |
|----------|-------|
| 4-1 – 4-4 | Optimal probability cutoff |
| 4-5 – 4-6 | Imbalanced data: RUS, ROS, SMOTE |
| 4-7 – 4-9 | K-fold and stratified CV, ROC by fold |
| 4-10 – 4-14 | Bagging, feature importance, RF, ExtraTrees, boundaries |
| 4-15 – 4-18 | AdaBoost, gradient boosting, XGBoost |
| 4-19 – 4-21 | Individual models, voting, stacking |
| 4-22 – 4-24 | GridSearch, RandomSearch, Bayesian optimization |
| 4-25 | Wavelet denoising for IoT data |

---

## Optional Python Packages

| Package | Used for |
|---------|----------|
| `imbalanced-learn` | SMOTE, RUS, ROS (`fit_resample`) |
| `xgboost` | Gradient boosting listings |
| `bayesian-optimization` | Listing 4-24 |
| `PyWavelets` | Listing 4-25 |
| `mlxtend` | Stacking (notebook falls back to sklearn if unavailable) |

Install as needed:

```bash
pip install imbalanced-learn xgboost bayesian-optimization PyWavelets mlxtend
```

---

## How to Use the Notebook

1. Run **Shared Setup** — imports and `DATA_DIR`.  
2. Work through in order: cutoff → imbalance → CV → ensembles → tuning → wavelets.  
3. For each listing, read the markdown **“Look for:”** line — it tells you what output matters.  
4. Compare **train vs test vs CV** scores every time — consistency beats a single lucky number.  
5. When tuning, always hold out a **final test set** never seen during GridSearch.

---

## Key Takeaways

- **Default 0.5 cutoff** is not always optimal; align threshold with business error costs.  
- **Imbalanced data** requires resampling and/or metrics beyond accuracy (recall, AUC, F1).  
- **Cross-validation** gives reliable model comparison; **stratified** folds protect class balance.  
- **Ensembles** (bagging, boosting, voting, stacking) almost always beat a single weak model.  
- **Hyperparameter tuning** is systematic search — start with RandomSearch, refine with GridSearch or Bayesian methods.  
- **Wavelets** clean noisy sensor data for downstream ML on IoT streams.  

---

## What Comes Next (Book Chapter 5)

The next step in the book is **Text Mining and Recommender Systems** — NLP preprocessing, TF-IDF, sentiment analysis, and collaborative filtering. That chapter is not yet included in this repository.

---

## Course Navigation

| Previous | Next |
|----------|------|
| [Chapter 3 README](README_Chapter_3.md) | [Main course README](README.md) |
