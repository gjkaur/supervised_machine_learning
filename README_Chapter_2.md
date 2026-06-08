# Chapter 2: Introduction to Machine Learning

**Notebook:** [`Chapter_2_Introduction_to_Machine_Learning.ipynb`](Chapter_2_Introduction_to_Machine_Learning.ipynb)  
**Source:** *Mastering Machine Learning with Python in Six Steps* (2nd ed.) — Manohar Swamynathan, Step 2  
**Code listings:** 2-1 through 2-49  
**Prerequisite:** [Chapter 1 — Python 3 basics](README_Chapter_1.md)

---

## Who This Chapter Is For

You know basic Python. This chapter explains **what machine learning is**, how it relates to AI and data science, and introduces the **three libraries** you will use in every ML project: **NumPy**, **pandas**, and **matplotlib**.

---

## What You Will Learn

- Definitions and real-world applications of machine learning
- History of AI and where ML fits in the bigger picture
- Differences between statistics, data analytics, and data science
- Three ML paradigms: supervised, unsupervised, reinforcement
- Industry frameworks for building ML systems (KDD, CRISP-DM, SEMMA)
- NumPy arrays: creation, indexing, slicing, math, broadcasting
- pandas: Series, DataFrame, I/O, statistics, merge, groupby, pivot tables
- matplotlib: line plots, bar charts, pie charts, customization
- Overview of scikit-learn and other ML core libraries

---

## Topics Covered (Complete Chapter Outline)

### What Is Machine Learning?

**Machine learning (ML)** is a subfield of AI focused on algorithms that **learn patterns from data** to make predictions or decisions without being explicitly programmed for every case.

Key definitions from the book:

- **Arthur Samuel (1959):** “Field of study that gives computers the ability to learn without being explicitly programmed.”
- **Modern view:** Statistical methods + algorithms that improve with data or reveal patterns humans would miss.

In short: ML = **algorithms + data → predictions and insights**.

#### Common ML Applications

| Domain | Example |
|--------|---------|
| **Recommendations** | YouTube video suggestions, Amazon product recommendations |
| **Spam detection** | Email filters that learn from labeled spam/ham |
| **Prospect identification** | Banks predicting which customers will convert |
| **Fraud detection** | Unusual transaction patterns |
| **Image / speech** | Face recognition, voice assistants |
| **Healthcare** | Disease risk from patient records |

---

### History and Evolution of AI

ML is a **subset of AI**. AI aims to build systems that observe, plan, optimize, act, and adapt — replicating or surpassing human intelligence in specific tasks.

#### The AI Process Loop (Figure 2-1)

1. **Observe** — identify patterns in data  
2. **Plan** — list possible solutions  
3. **Optimize** — pick the best solution  
4. **Action** — execute the chosen solution  
5. **Learn and adapt** — if results are wrong, adjust  

#### Intelligent Agents — Taxi Driver Example

An intelligent agent has:

| Component | Taxi example |
|-----------|--------------|
| **Agent type** | Robotic taxi driver |
| **Sensors (percepts)** | GPS, cameras, speedometer, obstacle sensors |
| **Actuators (actions)** | Steering, acceleration, braking, passenger interface |
| **Goals** | Safe, legal, comfortable, profitable trip |
| **Environment** | Roads, traffic, signals, pedestrians |

#### The Turing Test (1950)

Alan Turing proposed an operational test for intelligence: a human interrogator cannot reliably tell whether replies come from a machine or another human.

To pass, a system needs capabilities such as:

- **Natural language processing** — communicate in human language  
- **Knowledge representation** — store and use facts (expert systems)  
- **Automated reasoning** — draw conclusions  
- **Machine learning** — adapt from experience  
- **Computer vision** — interpret images  
- **Robotics** — act in the physical world  

#### Artificial Intelligence Evolution

AI has progressed through waves of optimism and “AI winters.” Modern success comes from **big data**, **compute power**, and **better algorithms** (especially deep learning).

#### Different Forms: ANI, AGI, ASI

| Term | Meaning |
|------|---------|
| **ANI** (Artificial Narrow Intelligence) | Excels at one task (chess, spam filter, face ID) — **today’s ML** |
| **AGI** (Artificial General Intelligence) | Human-level intelligence across all tasks — not yet achieved |
| **ASI** (Artificial Super Intelligence) | Surpasses human intelligence — theoretical |

---

### Statistics vs. Data Science vs. Data Analytics

| Field | Focus |
|-------|-------|
| **Statistics** | Mathematical theory of uncertainty, inference, hypothesis testing |
| **Data analytics** | Answer business questions with data (often descriptive) |
| **Data science** | Combines statistics, programming, and domain knowledge to build predictive systems |

#### Descriptive vs. Predictive Analytics (Figure 2-4)

- **Descriptive** — *What happened?* (reports, dashboards, summaries)  
- **Predictive** — *What will happen?* (forecasting, classification, scoring)  
- **Prescriptive** — *What should we do?* (optimization, recommendations)  

ML is central to **predictive** and **prescriptive** analytics.

---

### Machine Learning Categories (Figure 2-9)

#### 1. Supervised Learning

Learn a mapping from **labeled** inputs to outputs.

- **Regression** — predict a number (price, temperature, score)  
- **Classification** — predict a category (spam/not spam, disease yes/no)  

#### 2. Unsupervised Learning

Find structure in **unlabeled** data.

- **Clustering** — group similar records  
- **Dimensionality reduction** — compress features (PCA)  
- **Association rules** — items bought together  

#### 3. Reinforcement Learning

An **agent** learns by trial and error, receiving **rewards** or **penalties** (games, robotics, ad bidding).

---

### Frameworks for Building ML Systems

Real projects need process, not just algorithms.

#### KDD — Knowledge Discovery in Databases (Figure 2-10)

1. Selection → 2. Preprocessing → 3. Transformation → 4. Data mining → 5. Interpretation/evaluation  

#### CRISP-DM — Cross-Industry Standard Process for Data Mining (Figure 2-11)

1. **Business understanding**  
2. **Data understanding**  
3. **Data preparation**  
4. **Modeling**  
5. **Evaluation**  
6. **Deployment**  

#### SEMMA (SAS Enterprise Miner)

1. **Sample** — select relevant data  
2. **Explore** — visualize and summarize  
3. **Modify** — transform and feature engineer  
4. **Model** — train algorithms  
5. **Assess** — evaluate and iterate  

All three share: understand data → prepare → model → evaluate → deploy.

---

### Python Packages for Machine Learning

Four pillars of the Python data stack:

| Package | Role |
|---------|------|
| **NumPy** | Fast numerical arrays and math |
| **pandas** | Tabular data (spreadsheet-like) |
| **matplotlib** | Plots and charts |
| **scikit-learn** | ML algorithms and pipelines |

Also mentioned: SciPy, statsmodels, TensorFlow, Keras, PyTorch, XGBoost, NLTK, etc.

---

### NumPy

NumPy is the **core library for scientific computing**. ML libraries are built on NumPy arrays.

#### Array Basics (Listing 2-1)

Create arrays from Python lists; access elements by index.

#### Creating NumPy Arrays (Listing 2-2)

`np.array()`, `np.zeros()`, `np.ones()`, `np.arange()`, `np.linspace()`, `np.eye()`.

#### NumPy Data Types (Listing 2-3)

`int32`, `float64`, `bool_`, structured dtypes for mixed columns.

#### Field Access (Listing 2-4)

Access named fields in structured arrays.

#### Basic Slicing (Listings 2-5 through 2-10)

Slice 1D and multi-dimensional arrays: `arr[start:stop:step]`, negative indices, reshaping views.

#### Advanced Indexing (Listings 2-11 and 2-12)

- **Integer array indexing** — pick arbitrary positions  
- **Boolean array indexing** — filter by condition (`arr[arr > 0]`)  

#### Array Math (Listings 2-13 and 2-14)

Element-wise operations (`+`, `*`, `np.sin`, `np.exp`) and matrix operations (`np.dot`, `@`).

#### Sum and Transpose (Listings 2-15 and 2-16)

`np.sum()`, `axis=` parameter, `.T` or `np.transpose()`.

#### Broadcasting (Listings 2-17 through 2-20)

Apply operations between arrays of different shapes without explicit loops — essential for efficient ML code.

---

### Pandas

pandas adds **Series** (1D labeled) and **DataFrame** (2D table) structures.

#### Series (Listing 2-21)

Labeled one-dimensional array with index.

#### DataFrame (Listing 2-22)

Rows and columns with labels — like Excel or SQL tables.

#### Reading and Writing Data (Listing 2-23)

`read_csv()`, `read_excel()`, `read_table()`, `to_csv()`, `to_excel()`.

#### Basic Statistics (Listings 2-24 through 2-26)

`.describe()`, `.mean()`, covariance, correlation matrix.

#### Viewing and Filtering Data (Table 2-2)

`.head()`, `.tail()`, `.loc[]`, `.iloc[]`, boolean filtering, sorting.

#### Basic Operations (Table 2-3)

`pd.to_datetime()`, `.duplicated()`, missing value checks, `.apply()`.

#### Merge and Join (Listings 2-27 through 2-32)

- **concat / append** — stack tables  
- **merge** — SQL-style joins: inner, left, right, outer  

#### Grouping and Pivot Tables (Listings 2-33 and 2-34)

- **groupby** — split-apply-combine aggregations  
- **pivot_table** — Excel-style summary tables  

---

### Matplotlib

matplotlib creates publication-quality charts.

#### Plots on Variables (Listing 2-35)

Quick plots with global functions: `plt.plot()`, `plt.hist()`.

#### Plots Directly on DataFrames (Listing 2-36)

`df.plot()`, `df.hist()`, `df.boxplot()`.

#### Customizing Labels (Listing 2-37)

Titles, axis labels, legends, font sizes.

#### Object-Oriented API (Listing 2-38)

`fig, ax = plt.subplots()` — preferred for complex figures.

#### Line Plots (Listings 2-39 through 2-42)

Single line, multiple lines on same axis, multiple subplots, line/marker styles.

#### Bar Charts (Listings 2-43 through 2-46)

Vertical/horizontal bars, grouped bars, stacked bars.

#### Pie Chart and Grid Layout (Listings 2-47 and 2-48)

`plt.pie()`, `plt.subplot()` grid arrangements.

#### Plotting Defaults (Listing 2-49)

`plt.rcParams` — set global style, figure size, colors.

---

### Machine Learning Core Libraries (scikit-learn Summary)

**scikit-learn** provides consistent APIs for:

- Preprocessing (scaling, encoding)  
- Model training (regression, classification, clustering)  
- Model selection (train/test split, cross-validation)  
- Metrics (accuracy, RMSE, ROC-AUC)  

Pattern used throughout the course:

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model = LogisticRegression().fit(X_train, y_train)
model.score(X_test, y_test)
```

---

## Data Files Used

| File | Purpose |
|------|---------|
| `Data/iris`-style data via sklearn | pandas and matplotlib examples |
| `Data/mtcars.csv` (and variants) | merge, join, groupby exercises |

---

## Code Listings Summary

| Range | Topic |
|-------|-------|
| 2-1 – 2-4 | NumPy basics and dtypes |
| 2-5 – 2-10 | Slicing and reshaping |
| 2-11 – 2-12 | Advanced indexing |
| 2-13 – 2-16 | Math, sum, transpose |
| 2-17 – 2-20 | Broadcasting |
| 2-21 – 2-23 | pandas Series, DataFrame, I/O |
| 2-24 – 2-26 | Statistics and correlation |
| 2-27 – 2-32 | Concat, merge, joins |
| 2-33 – 2-34 | Groupby, pivot tables |
| 2-35 – 2-49 | matplotlib plotting |

---

## How to Use the Notebook

1. Complete Chapter 1 first (or confirm you know Python basics).
2. Run the **Shared Setup** cell — it imports NumPy, pandas, matplotlib and sets display options.
3. Work through NumPy → pandas → matplotlib in order; each section builds on the last.
4. When you see a plot, ask: *What pattern would help a model?*
5. Refer to **Table 2-2** and **Table 2-3** in the book/notebook as cheat sheets for pandas syntax.

---

## Key Takeaways

- ML learns from **data**; choose supervised vs unsupervised based on whether you have labels.
- Use a **structured process** (CRISP-DM) — jumping straight to algorithms leads to failure.
- **NumPy** = fast arrays; **pandas** = tables; **matplotlib** = visualization; **scikit-learn** = models.
- Mastering these four tools unlocks everything in Chapters 3 and 4.

---

## Next Step

Continue to **[Chapter 3: Fundamentals of Machine Learning](README_Chapter_3.md)** — feature engineering, regression, classification, time series, clustering, and PCA.
