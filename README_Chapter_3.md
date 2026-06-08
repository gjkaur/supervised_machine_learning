# Chapter 3: Fundamentals of Machine Learning

**Notebook:** [`Chapter_3_Fundamentals_of_Machine_Learning.ipynb`](Chapter_3_Fundamentals_of_Machine_Learning.ipynb)  
**Source:** *Mastering Machine Learning with Python in Six Steps* (2nd ed.) — Manohar Swamynathan, Step 3  
**Code listings:** 3-1 through 3-56 (56 examples)  
**Prerequisite:** [Chapter 2 — Introduction to ML](README_Chapter_2.md)

---

## Who This Chapter Is For

You understand Python, NumPy, pandas, and basic ML concepts. This chapter walks through the **full ML workflow**: prepare data, explore it, build models, diagnose them, and evaluate results — covering regression, classification, time series, clustering, and dimensionality reduction.

---

## What You Will Learn

- How ML views data types and measurement scales
- Feature engineering: missing values, categoricals, scaling, new features
- Exploratory data analysis (EDA) with univariate and multivariate methods
- Linear and polynomial regression, multivariate models, regularization
- Logistic regression and classification metrics (confusion matrix, ROC-AUC)
- Decision trees, SVM, k-nearest neighbors
- ARIMA time-series forecasting
- k-means and hierarchical clustering, elbow/silhouette methods
- Principal Component Analysis (PCA)

---

## Topics Covered (Complete Chapter Outline)

### Machine Learning Perspective of Data

Before modeling, understand what your columns represent:

- **Features (X)** — inputs the model uses  
- **Target (y)** — what you predict (supervised learning)  
- **Observations (rows)** — individual records  

Figure 3-1 shows the logical flow: raw data → clean → engineer features → split train/test → train → evaluate → deploy.

---

### Scales of Measurement

| Scale | Meaning | Example | Valid operations |
|-------|---------|---------|------------------|
| **Nominal** | Categories, no order | Gender, zip code, species | Count, mode |
| **Ordinal** | Ordered categories | Rating (low/med/high), education level | Rank, median |
| **Interval** | Numeric, equal gaps, no true zero | Temperature °C | Mean, std dev |
| **Ratio** | Numeric with true zero | Age, income, weight | All math ops |

Choosing the wrong encoding (e.g., treating ordinal as nominal) hurts model quality.

---

### Feature Engineering

Transform raw data into a form algorithms can use.

#### Dealing with Missing Data

Common strategies:

- **Drop** rows or columns with too many missing values  
- **Impute** with mean, median, mode, or model-based fill  
- pandas: `.fillna()`, `.dropna()`, `.isna()`  

#### Handling Categorical Data

ML algorithms expect numbers. Options:

1. **Dummy (one-hot) encoding** — one binary column per category (Listing 3-1)  
2. **Label encoding** — map categories to integers 0, 1, 2… (Listing 3-2)  

Use one-hot for nominal variables; label encoding only when order matters or for tree-based models.

#### Normalizing Data (Listing 3-3)

Put features on comparable scales:

- **Standardization (Z-score):** mean = 0, std = 1 — `(x - mean) / std`  
- **Min–max scaling:** squeeze to [0, 1]  

Required for distance-based models (SVM, kNN) and gradient-based optimization. Remove extreme outliers before scaling.

#### Feature Construction or Generation

Create new columns from existing ones (ratios, bins, date parts, interactions). Requires domain knowledge. Figure 3-2 summarizes common summarization methods (count, sum, mean, min, max by group).

---

### Exploratory Data Analysis (EDA)

Understand data **before** modeling. The chapter uses the **Iris dataset** (150 flowers, 4 measurements, 3 species).

#### Univariate Analysis (Listing 3-4)

One variable at a time: `.describe()`, histograms, value counts for categoricals.

#### Pandas DataFrame Visualization (Listing 3-5)

Quick plots: `df.hist()`, boxplots, bar charts of class frequencies.

#### Multivariate Analysis (Listing 3-6)

Relationships between two or more variables — scatter plots colored by species.

#### Correlation Matrix (Listing 3-7)

Numeric correlations (`df.corr()`) — values near ±1 indicate strong linear relationships.

#### Pair Plot (Listing 3-8)

Grid of scatter plots for all feature pairs — reveals clusters and separability.

#### EDA Findings (Iris)

- Petal length/width strongly separate species  
- Setosa is linearly separable; Versicolor vs Virginica overlap slightly  
- These insights guide algorithm choice  

---

### Supervised Learning — Regression

Predict a **continuous** target (“how much?”, “how many?”).

Examples: credit score, house price, test grade, sales forecast.

#### Students' Score vs. Hours Studied (Listing 3-9)

Simple dataset linking study hours to test grades — strong positive correlation (~0.99).

#### Linear Regression Model Components (Figure 3-5)

- **Intercept (c)** — baseline prediction when X = 0  
- **Slope (m)** — change in y per unit change in X  
- Equation: **y = mX + c**  

#### Linear Regression (Listing 3-10)

Fit with `LinearRegression` from scikit-learn; compare manual formula vs `.predict()`.

#### How Good Is Your Model?

| Metric | What it measures |
|--------|------------------|
| **R² (R-squared)** | Fraction of variance explained (0–1, higher better) |
| **MAE** (Mean Absolute Error) | Average absolute prediction error |
| **RMSE** (Root Mean Squared Error) | Penalizes large errors more than MAE |

Listings 3-11 and 3-12 compute these metrics; Listing 3-12 shows how **outliers** inflate error and distort R².

#### Correlation and Causation

High correlation does **not** prove cause. Ice cream sales and drowning both rise in summer — correlated but not causal. Always combine statistics with **domain expertise**.

#### Fitting a Slope

Least squares finds the line that minimizes sum of squared residuals.

#### Polynomial Regression

When the relationship is curved, add powers of X:  
Y = m₁X + m₂X² + c (quadratic), cubic, etc.

- Listings 3-13 – 3-16: manual polynomial fit, R² by degree, `PolynomialFeatures` in sklearn  
- **Risk:** high degree → overfitting  

---

### Multivariate Regression

Multiple independent variables predict one target (e.g., house price from lot size, bedrooms, basement, etc.).

#### Multicollinearity and VIF (Listing 3-17)

When predictors are highly correlated with each other, coefficients become unstable. **VIF (Variance Inflation Factor)** > 10 (or 5) suggests removing or combining variables.

#### Remove Multicollinearity (Listing 3-18)

Iteratively drop high-VIF features until the set is acceptable.

#### Build Multivariate Model (Listing 3-19)

Train/test split; report MAE and RMSE on train and test sets.

#### Interpreting OLS Regression Results

statsmodels `OLS` summary includes:

- **R² / Adjusted R²** — model fit (adjusted penalizes extra variables)  
- **F-statistic / Prob(F)** — overall model significance  
- **Coefficients / p-values** — which features matter (p ≤ 0.05 often “significant”)  
- **Durbin–Watson** — autocorrelation in residuals (time series concern)  

#### Regression Diagnostics

Check assumptions before trusting coefficients:

| Check | Listing | What to look for |
|-------|---------|------------------|
| **Outliers / leverage** | 3-20, 3-21 | Points far from the rest influence the line |
| **Homoscedasticity** | 3-22 | Residual spread should be constant |
| **Linearity** | 3-23 | Residuals vs fitted should be random scatter |
| **Normality** | implied | Residuals roughly bell-shaped |

#### Overfitting and Underfitting (Figure 3-8)

- **Underfitting** — model too simple (high bias)  
- **Overfitting** — model memorizes training data (high variance)  
- **Right fit** — generalizes to new data  

#### Regularization (Listing 3-24)

Penalize large coefficients to reduce overfitting:

- **Ridge (L2)** — shrinks coefficients toward zero  
- **LASSO (L1)** — can zero out useless features (feature selection)  
- **Elastic Net** — combines L1 and L2  

#### Nonlinear Regression (Listing 3-25)

When the relationship is not polynomial-linear, use transformations or specialized fits (e.g., `curve_fit`).

---

### Supervised Learning — Classification

Predict a **discrete class** (“yes/no”, “cat/dog/bird”).

Examples: loan default, disease diagnosis, species identification.

#### Logistic Regression (Listings 3-26 – 3-28)

Despite the name, used for **classification**. Applies the **sigmoid (logit)** function to squash linear output to a probability between 0 and 1.

- Listing 3-27 plots the sigmoid  
- Listing 3-28 trains with scikit-learn; outputs probabilities and class labels  

#### Evaluating Classification Performance (Listings 3-29 – 3-30)

**Confusion matrix:**

|  | Predicted No | Predicted Yes |
|--|--------------|---------------|
| **Actual No** | TN | FP (Type I) |
| **Actual Yes** | FN (Type II) | TP |

Key metrics:

| Metric | Formula / meaning |
|--------|-------------------|
| **Accuracy** | (TP + TN) / total |
| **Sensitivity (Recall)** | TP / (TP + FN) — catch positives |
| **Specificity** | TN / (TN + FP) |
| **Precision** | TP / (TP + FP) |
| **FPR** | FP / (FP + TN) |

**ROC curve** — plot TPR vs FPR at all thresholds; **AUC** summarizes overall ranking ability (1.0 = perfect, 0.5 = random).

#### Fitting Line / Complexity (Listing 3-31)

Visualize decision boundary complexity for classification.

#### Stochastic Gradient Descent

Iterative algorithm to update weights for large datasets — foundation for many scalable classifiers.

#### Regularization — Underfitting vs Overfitting (Listing 3-32)

Logistic regression parameter **C** (inverse regularization strength): small C = more regularization = simpler boundary.

#### Multiclass Logistic Regression (Listings 3-33 – 3-36)

Full pipeline on Iris:

1. Load data (3-33)  
2. Normalize (3-34)  
3. Train/test split (3-35)  
4. Train, confusion matrix, classification report (3-36)  

#### Generalized Linear Models (GLM) (Listing 3-37)

statsmodels GLM framework — extends linear models to different distributions (Gaussian, Binomial, Poisson, etc.) via a **link function**.

#### Supervised Learning Process Flow (Figure 3-13)

End-to-end: define problem → collect data → EDA → feature engineering → split → train → validate → tune → test → deploy → monitor.

---

### Decision Trees (Listing 3-38)

Split data recursively on feature thresholds that best separate classes.

- **Pros:** interpretable, handles non-linear boundaries, no scaling needed  
- **Cons:** prone to overfitting without limits  
- **Key parameters:** `max_depth`, `min_samples_split`, `min_samples_leaf`  

Figure 3-14 shows Quinlan’s classic weather/play example.

#### How the Tree Splits and Grows

Uses impurity measures (Gini, entropy) to pick each split.

#### Conditions for Stopping Partitioning

Stop when max depth reached, too few samples in a node, or pure nodes.

---

### Support Vector Machines — SVM (Listings 3-39 – 3-40)

Find the **maximum-margin hyperplane** separating classes. Can use **kernels** (RBF, polynomial) for non-linear boundaries.

- Listing 3-39 trains SVM on Iris  
- Listing 3-40 plots decision boundaries  

**Key parameters:** `C` (regularization), `kernel`, `gamma`.

---

### k-Nearest Neighbors — kNN (Listing 3-41)

Classify a point by **majority vote** of its k closest training neighbors. Simple but effective; sensitive to feature scale (normalize first).

Figure 3-16 illustrates k = 5.

---

### Time-Series Forecasting

Predict future values from past observations (sales, temperature, stock prices).

#### Components of Time Series (Figure 3-17)

- **Trend** — long-term direction  
- **Seasonality** — repeating patterns  
- **Cyclical** — non-fixed period fluctuations  
- **Irregular / noise** — random variation  

#### Decompose Time Series (Listing 3-42)

`seasonal_decompose()` splits series into trend, seasonal, and residual parts.

#### ARIMA — Autoregressive Integrated Moving Average

Model notation **ARIMA(p, d, q)**:

- **p** — autoregressive lags  
- **d** — differencing order (make series stationary)  
- **q** — moving average terms  

#### Checking Stationarity (Listing 3-43)

**Augmented Dickey–Fuller test** — p-value < 0.05 suggests stationarity. Log transform and first differencing often help.

#### Autocorrelation Test (Listing 3-44)

ACF/PACF plots guide choice of p and q.

#### Build and Evaluate ARIMA (Listings 3-45 – 3-47)

Fit models with statsmodels; compare MAE, RMSE, Durbin–Watson across parameter choices.

#### Predicting Future Values (Listing 3-48)

`.forecast()` or `.predict()` for out-of-sample periods. Book recommends **3–4 years** of history for reliable seasonality.

---

### Unsupervised Learning Process Flow (Figure 3-18)

No labeled target — discover structure: segment customers, reduce dimensions, detect anomalies.

---

### Clustering

Group similar observations without predefined labels.

#### k-means Clustering (Listings 3-49 – 3-50)

1. Choose k cluster centers  
2. Assign points to nearest center  
3. Recompute centers as cluster means  
4. Repeat until stable  

Uses **Euclidean distance** only. Limitations: assumes spherical clusters, sensitive to outliers and scale.

Compare cluster labels to true species (when known) for accuracy assessment.

#### Finding the Value of k

**Elbow method (Listing 3-51)** — plot within-cluster sum of squares vs k; look for an “elbow”.  
**Silhouette method (Listing 3-52)** — higher average silhouette score = better-defined clusters.

#### Hierarchical Clustering (Listings 3-53 – 3-54)

Agglomerative approach: start with each point as its own cluster, merge closest pairs repeatedly.

**Linkage methods:** single, complete, average, **Ward** (minimizes variance increase — common default).

Output visualized as a **dendrogram** — cut height determines number of clusters.

---

### Principal Component Analysis — PCA (Listings 3-55 – 3-56)

Reduce many correlated features to fewer **principal components** that capture most variance.

Steps:

1. Standardize features  
2. Compute covariance matrix  
3. Find eigenvectors (directions) and eigenvalues (variance explained)  
4. Project data onto top components  

Use when you have many features, multicollinearity, or need visualization in 2D/3D. Listing 3-56 plots Iris in PC1 vs PC2 space.

---

## Data Files Used

| File | Used for |
|------|----------|
| `Data/Grade_Set_1.csv` | Linear regression (hours vs grade) |
| `Data/Grade_Set_2.csv` | Polynomial regression |
| `Data/Grade_Set_1_Classification.csv` | Logistic regression |
| `Data/Housing_Modified.csv` | Multivariate regression, VIF, diagnostics |
| `Data/LR_NonLinear.csv` | Nonlinear regression |
| `Data/TS.csv` | ARIMA time series |
| Iris (sklearn built-in) | EDA, classification, clustering, PCA |

---

## Code Listings Summary

| Listings | Topic |
|----------|-------|
| 3-1 – 3-3 | Dummy variables, label encoding, scaling |
| 3-4 – 3-8 | EDA: univariate, multivariate, correlation, pair plot |
| 3-9 – 3-16 | Linear and polynomial regression |
| 3-17 – 3-19 | VIF, multivariate regression |
| 3-20 – 3-25 | Diagnostics, regularization, nonlinear regression |
| 3-26 – 3-37 | Logistic regression, metrics, multiclass, GLM |
| 3-38 | Decision trees |
| 3-39 – 3-40 | SVM |
| 3-41 | kNN |
| 3-42 – 3-48 | Time series / ARIMA |
| 3-49 – 3-54 | k-means, elbow, silhouette, hierarchical clustering |
| 3-55 – 3-56 | PCA |

---

## How to Use the Notebook

1. Run **Shared Setup** — sets `DATA_DIR = Path('Data')` and imports libraries.  
2. Do not skip EDA — it explains *why* later models succeed or fail.  
3. For each model section, note: **What is X? What is y? What metric matters?**  
4. Compare train vs test performance — a large gap means overfitting.  
5. Optional packages: `statsmodels` for OLS/GLM/ARIMA summaries.

---

## Key Takeaways

- **Measurement scale** and **encoding** affect every downstream step.  
- **EDA** saves time — visualize before fitting.  
- **Regression** metrics (R², MAE, RMSE) differ from **classification** metrics (confusion matrix, AUC).  
- **Regularization** and **diagnostics** separate reliable models from misleading ones.  
- **Unsupervised** methods discover structure when you have no labels.  
- Chapter 4 builds on this with cutoff tuning, imbalanced data, ensembles, and hyperparameter search.

---

## Next Step

Continue to **[Chapter 4: Model Diagnosis and Tuning](README_Chapter_4.md)** — optimal thresholds, resampling, cross-validation, ensemble methods, and automated tuning.
