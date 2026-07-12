# Applied AI & ML Capstone Project
## End-to-End Data Intelligence System

## Project Overview

This project was completed as part of the Applied AI & Machine Learning Capstone Project.

The objective of this project is to develop a complete end-to-end machine learning workflow starting from raw data preprocessing to advanced predictive modelling and finally integrating a Large Language Model (LLM) powered intelligent feature.

The project is divided into four independent but connected parts.

- Part 1 – Data Acquisition, Cleaning and Exploratory Data Analysis
- Part 2 – Supervised Machine Learning
- Part 3 – Advanced Ensemble Models, Hyperparameter Tuning and ML Pipeline
- Part 4 – LLM Powered Intelligent Feature

Each part builds upon the previous one and demonstrates practical data science techniques used in real-world analytics projects.

## Dataset

**Name:** House Prices – Advanced Regression Techniques
**Source:** Kaggle
**Rows:** 1460 · **Columns:** 81 (38 numerical, 43 categorical)

### Description
The dataset contains detailed information about residential houses, including numerical and categorical attributes describing physical characteristics, construction quality, location, size, utilities, and selling price.

**Regression target:** `SalePrice`
**Classification target:** `PriceCategory = (SalePrice > SalePrice.median()).astype(int)`

### Justification
This dataset was chosen because it satisfies all assignment requirements (more than 500 rows, more than 5 numeric columns, more than 2 categorical columns, a target usable for both regression and classification) and realistically contains missing values, skewed distributions, categorical variables, and outliers — making it suitable for demonstrating the full cleaning → EDA → modeling pipeline.

---

## Repository Structure

```text
Applied_AI_ML_Capstone/
│
├── data/
│   ├── train.csv
│   └── cleaned_data.csv
│
├── models/
│   └── best_model.pkl
│
├── notebooks/
│   ├── Part1_EDA.ipynb
│   ├── Part2_ML.ipynb
│   ├── Part3_AdvancedModels.ipynb
│   └── Part4_LLM.ipynb
│
├── README.md
└── requirements.txt
```


# Part 1 – Data Acquisition, Cleaning and Exploratory Data Analysis

## Objective

The objective of Part 1 is to convert the raw dataset into a clean and analysis-ready dataset before applying any machine learning models.

The workflow includes:

- Loading the raw dataset
- Handling missing values
- Detecting duplicate records
- Correcting data types
- Memory optimization
- Statistical analysis
- Outlier detection
- Exploratory Data Analysis (EDA)
- Correlation analysis
- Feature relationship analysis
- Saving the cleaned dataset for subsequent parts of the project

The final cleaned dataset generated in this phase is stored as:

cleaned_data.csv

This file is used directly in Part 2 and Part 3.


## 1. Data Loading

Loaded using `pd.read_csv()`. First 5 rows, `.dtypes`, and `.shape` were printed and confirmed.

- **Shape:** `(1460, 81)`
- **Numerical columns:** 38 · **Categorical columns:** 43

---

## 2. Null Value Analysis

Computed using `df.isnull().sum()` and `(df.isnull().sum() / df.shape[0]) * 100`.

**Columns exceeding 20% nulls:**

| Column | Missing Values | Missing % |
|---|---|---|
| PoolQC | 1453 | 99.52% |
| MiscFeature | 1406 | 96.30% |
| Alley | 1369 | 93.77% |
| Fence | 1179 | 80.75% |
| MasVnrType | 872 | 59.73% |
| FireplaceQu | 690 | 47.26% |

These six columns were left untouched by the median-imputation step (they are handled separately, e.g. via encoding, in Part 2).

**Example column below 20% nulls:** `LotFrontage` — 259 missing values (17.74%) — imputed with the column median.

For all other **numeric** columns with <20% nulls, missing values were filled using `df[col].fillna(df[col].median())`.

> Note: after this step, `df.isnull().sum().sum()` still returns **7481** — these remaining nulls all belong to the categorical/high-null columns above (and a few low-null categorical columns like `Electrical`, `GarageType`, `BsmtQual`, etc.), which were intentionally left for the encoding stage in Part 2 rather than filled here.

### Why Median Instead of Mean?
The dataset contains several highly skewed numerical variables (e.g. `MiscVal`, skew = 24.48; `LotArea`, skew = 12.21). The mean is pulled toward extreme values, while the median stays stable under skew. Since housing datasets naturally contain a small number of very expensive or unusual properties, median imputation gives a more representative central value than the mean.

---

## 3. Duplicate Detection and Removal

- Duplicates found (`df.duplicated().sum()`): **0**
- Rows after `df.drop_duplicates()`: **1460** (unchanged)
- Null percentage change after removal: **No change** — since zero duplicate rows existed, the missing-value percentages before and after are identical for every column.

---

## 4. Data Type Correction

- **Category conversion:** `Neighborhood` was converted to `category` dtype (a highly repetitive string column with 25 unique values repeated across the 1460 rows).
- **Numeric correction check:** All `object`-typed columns in this dataset are genuinely categorical (e.g. `MSZoning`, `Street`, `SaleType`, `SaleCondition`) — none of them represent numeric data stored incorrectly as text, so no `pd.to_numeric()` correction was required for this dataset.

**Memory usage (`df.memory_usage(deep=True).sum()`):**

| Stage | Memory Usage |
|---|---|
| Before conversion | 3,600,842 bytes (≈ 3.43 MB) |
| After conversion | 3,523,227 bytes (≈ 3.36 MB) |
| Reduction | 77,615 bytes (≈ 2.16%) |

The reduction is modest here because only one column (`Neighborhood`) was converted; converting the remaining 42 categorical columns to `category` dtype in Part 2's encoding step would reduce memory further.

---

## 5. Descriptive Statistics & Skewness

`df.describe()` was run on all numeric columns (38 columns; see notebook for the full table — e.g. `SalePrice` mean = 180,921.20, std = 79,442.50, min = 34,900, max = 755,000).

**Skewness (`df[col].skew()`) — top result:**

The column with the **highest absolute skewness** is **`MiscVal`**, with a skewness value of **24.48**.

Top 5 most skewed columns overall:

| Column | Skewness |
|---|---|
| MiscVal | 24.48 |
| PoolArea | 14.83 |
| LotArea | 12.21 |
| 3SsnPorch | 10.30 |
| LowQualFinSF | 9.01 |

### Interpretation
- **Positive skew** (long right tail, as seen in `MiscVal`): most houses have `MiscVal = 0`, but a small number of properties report very large miscellaneous-feature values, pulling the mean far above the median.
- **Negative skew** (long left tail, e.g. `YearBuilt`, skew = −0.61): most houses are relatively recent builds, with a smaller number of much older houses pulling the mean down.
- **Consequence for imputation:** filling missing values with the mean on a column like `MiscVal` would inject a distorted value (43.49) that doesn't represent the true typical value (0) — this is exactly why median imputation was used throughout Task 2.

---

## 6. Outlier Detection (IQR Method)

Applied to two numeric columns: **`SalePrice`** and **`GrLivArea`**.

| Column | Q1 | Q3 | IQR | Lower Bound | Upper Bound | Outlier Rows |
|---|---|---|---|---|---|---|
| SalePrice | 129,975.00 | 214,000.00 | 84,025.00 | 3,937.50 | 340,037.50 | 61 |
| GrLivArea | 1,129.50 | 1,776.75 | 647.25 | 158.63 | 2,747.63 | 31 |

*(For reference, the same IQR method was also run across all 38 numeric columns; the highest outlier counts were seen in `EnclosedPorch` (208), `BsmtFinSF2` (167), and `OverallCond` (125), while columns like `YearRemodAdd`, `FullBath`, `HalfBath`, `MoSold`, and `YrSold` showed 0 outliers.)*

### Interpretation & Handling Decision
Outliers were **not dropped**. High values in `SalePrice` and `GrLivArea` mostly represent genuine large/premium properties rather than data errors — removing them would throw away real signal the model needs to learn the full price range. These rows are documented and retained; in Part 2, Ridge Regression and ensemble methods (which are more robust to outliers than plain Linear Regression) are used to handle their influence, rather than capping or deleting them.

---

## 7. Visualizations

### 7.1 Line Plot
Plotted **`SalePrice`** sorted by row index.
*Description:* Prices fluctuate without any clear long-term trend across row order (row index is just the original record order, not a time axis), with visible spikes where premium properties (up to 755,000) appear among typically priced homes (median ≈ 163,000).

### 7.2 Bar Chart
Mean of **`SalePrice`** grouped by **`Neighborhood`**.
*Description:* `NoRidge` has the highest average sale price (≈ 335,295), while `MeadowV` has the lowest (≈ 98,576) — showing that neighborhood location has a strong influence on price.

### 7.3 Histogram
Histogram (20 bins) of the most skewed column, **`MiscVal`**.
*Description:* The distribution is extremely right-skewed — almost all houses have `MiscVal = 0`, with a long, sparse tail extending out to 15,500, confirming the skewness value of 24.48 measured in Section 5.

### 7.4 Scatter Plot
**`GrLivArea`** vs **`SalePrice`**.
*Interpretation:* The plot shows a clear positive relationship — as living area increases, sale price generally increases too, consistent with a Pearson correlation of **0.709** and Spearman correlation of **0.731** with SalePrice.

### 7.5 Box Plot
**`SalePrice`** split by **`OverallQual`**.
*Description:* Median sale price rises steadily with each increase in overall quality rating, and the spread (IQR) also widens at higher quality levels — higher-quality homes have both higher and more variable prices, confirming `OverallQual` as a strong predictive feature (it has the highest Pearson correlation with `SalePrice` at 0.791).

---

## 8. Correlation Heat Map

Computed with `df.corr(numeric_only=True)`, visualized with `sns.heatmap(annot=True)`.

**Highest absolute correlation pair (excluding self-pairs):** **`GarageCars`** and **`GarageArea`** (r = **0.882**).

Other strong pairs found: `GrLivArea`–`TotRmsAbvGrd` (0.825), `TotalBsmtSF`–`1stFlrSF` (0.820), `OverallQual`–`SalePrice` (0.791), `YearBuilt`–`GarageYrBlt` (0.777).

### Causation Discussion
The strong correlation between `GarageCars` and `GarageArea` does not imply one directly causes the other — a garage that holds more cars naturally needs more floor area, so this is largely a mechanical/definitional relationship rather than two independently varying features. More generally, a plausible third-variable explanation applies across several of these pairs: overall house size or quality (e.g. `OverallQual`, `GrLivArea`) likely drives multiple features simultaneously — larger, higher-quality homes tend to have bigger garages, more rooms, and larger basements all at once, so several of these correlations may reflect this shared underlying "house size/quality" factor rather than direct cause-and-effect between the two named variables.

---

## 9. Mean vs Median — Imputation Strategy Comparison

For the two highest-skewness columns identified in Section 5:

| Column | Skewness | Mean (before imputation) | Median (before imputation) | Chosen Statistic |
|---|---|---|---|---|
| MiscVal | 24.48 | 43.49 | 0.00 | Median |
| PoolArea | 14.83 | 2.76 | 0.00 | Median |

**Justification:** Both columns are strongly positively skewed, so the mean is pulled far above the "typical" value by a small number of extreme entries (most houses simply have 0 for both features). The median (0 for both) is the far more representative central tendency and was used for imputation. After applying `fillna()`, `df[top_columns].isnull().sum()` confirmed **zero remaining nulls** in both `MiscVal` and `PoolArea`.

---

## 10. Spearman Rank Correlation vs Pearson

`df.corr(method='spearman')` was computed and compared against the Pearson matrix from Section 8, focused on each feature's correlation with `SalePrice`.

**Three column pairs (each vs. SalePrice) with the largest |Spearman − Pearson| difference:**

| Feature (vs. SalePrice) | Pearson | Spearman | \|Difference\| |
|---|---|---|---|
| LotArea | 0.264 | 0.456 | 0.193 |
| OpenPorchSF | 0.316 | 0.478 | 0.162 |
| YearBuilt | 0.523 | 0.653 | 0.130 |

### Interpretation
- **LotArea:** |Spearman| > |Pearson| → the relationship with `SalePrice` is monotonic but non-linear — larger lots consistently associate with higher prices, but not in direct proportion (a few very large lots with modest prices pull the linear/Pearson measure down).
- **OpenPorchSF:** |Spearman| > |Pearson| → similarly monotonic-but-non-linear; having a porch consistently nudges price upward, but porch size doesn't scale price linearly.
- **YearBuilt:** |Spearman| > |Pearson| → newer houses consistently sell for more, but the relationship isn't a straight line (likely because very old, renovated, or historic homes break the linear trend).

**Feature-selection guidance for Part 2:** Pearson correlation will be used for the Linear/Ridge Regression models since they assume linear relationships, while Spearman correlation is kept as a secondary signal — especially for `LotArea`, `OpenPorchSF`, and `YearBuilt`, where the true relationship is stronger than Pearson alone suggests. Tree-based models in Part 3 can exploit this non-linear monotonic signal even though Pearson underestimates it.

---

## 11. Grouped Aggregation

`df.groupby('Neighborhood', observed=False)['SalePrice'].agg(['mean', 'std', 'count'])`

(25 neighborhood groups computed in total; key results below)

| Neighborhood | Mean | Std | Count |
|---|---|---|---|
| NoRidge | 335,295.32 | 121,412.66 | 41 |
| NridgHt | 316,270.62 | 96,392.54 | 77 |
| StoneBr | 310,499.00 | 112,969.68 | 25 |
| MeadowV | 98,576.47 | 23,491.05 | 17 |
| IDOTRR | 100,123.78 | 33,376.71 | 37 |

### Interpretation
- **Highest mean group:** **`NoRidge`** (mean SalePrice ≈ 335,295)
- **Highest standard deviation group:** **`NoRidge`** (std ≈ 121,413) — the same neighborhood also has the widest internal spread.
- **Within-group variance concern:** Since `NoRidge` has both the highest mean *and* the highest std, knowing a house is in `NoRidge` alone doesn't reliably predict its exact price — prices there range widely, so other features (size, quality, etc.) are still needed to predict individual house prices within that group.
- **Mean ratio (highest/lowest group mean):** 335,295.32 / 98,576.47 = **3.40**. A ratio this large (over 3×) indicates `Neighborhood` carries strong predictive signal and is worth keeping as a feature in Part 2.

---

## 12. Saving the Cleaned Dataset

Saved using `df.to_csv('cleaned_data.csv', index=False)`. This file is reused directly in Parts 2 and 3.



# Part 2 – Supervised Machine Learning Model

## Objective

After completing data cleaning and exploratory analysis in Part 1, the next step was to develop predictive machine learning models using the cleaned dataset.

The primary goal of this phase was to build two supervised learning models:

- A **Regression Model** for predicting the continuous selling price of a house.
- A **Binary Classification Model** for identifying whether a house belongs to the higher-priced or lower-priced category.

To ensure reliable model performance, the entire workflow followed standard machine learning practices, including proper preprocessing, leak-free train-test splitting, feature scaling, model training, evaluation, and performance comparison.

The complete workflow performed in this part includes:

- Loading the cleaned dataset
- Defining regression and classification targets
- Encoding categorical variables
- Train-test splitting
- Feature scaling
- Linear Regression
- Ridge Regression
- Logistic Regression
- Handling class imbalance
- Model evaluation
- ROC Curve analysis
- Decision threshold analysis
- Regularization experiments
- Bootstrap confidence interval estimation

The cleaned dataset generated in Part 1 was used throughout this section to maintain consistency and ensure that every model was trained on identical preprocessed data.

---


## 1. Loading the Cleaned Dataset

`cleaned_data.csv` from Part 1 was loaded directly — shape confirmed as **(1460, 81)**, matching what Part 1 saved.

## 2. Defining the Targets

- **Regression target:** `y_reg = SalePrice`
- **Classification target:** `y_clf = (SalePrice > SalePrice.median()).astype(int)`
- **Feature matrix:** `X = df.drop("SalePrice", axis=1)` → **(1460, 80)** columns before encoding.

Since the split is at the median, the classification target comes out almost perfectly balanced by construction: **732 rows labeled 0, 728 labeled 1.**

## 3. Encoding Categorical Features

Two different encoding strategies were used, depending on whether a column has a natural order:

**Ordinal encoding** — `ExterQual`, `KitchenQual`, and `HeatingQC` were mapped `Po=1, Fa=2, TA=3, Gd=4, Ex=5`. These are quality grades with a genuine low-to-high order, so representing them as ordered integers lets the model use that ranking directly instead of treating "Excellent" and "Poor" as unrelated categories.

**One-hot encoding** — everything else (`Neighborhood`, `MSZoning`, `SaleCondition`, `RoofMatl`, etc.) was encoded with `pd.get_dummies(drop_first=True)`. These columns have no inherent order, so one-hot avoids implying a false ranking — numbering neighborhoods 1 to 25 would wrongly suggest neighborhood #20 is "more" than neighborhood #5, which means nothing.

Before encoding, any remaining missing values in categorical columns (columns like `PoolQC`, `Alley`, `Fence` that Part 1 intentionally left untouched) were filled with the string `"None"` — since for this dataset, missing usually genuinely means "this house doesn't have this feature," not "we don't know." This turned out to matter more than expected — see Section 5.

After encoding, the feature matrix grew to **(1460, 254)**.

## 4. Leak-Free Train-Test Split and Scaling

`train_test_split(X, y, test_size=0.2, random_state=42)` gave a **1168-row training set** and a **292-row test set**.

`StandardScaler` was fit **only on `X_train`**, then used to transform both `X_train` and `X_test`. This matters because fitting the scaler on the full dataset before splitting would leak information about the test set's mean and spread into the "unseen" data — the model would effectively get a sneak peek at test-set statistics, making the evaluation look better than it would in the real world.

## 5. Linear Regression — and an Unexpected Result

| Metric | Value |
|---|---|
| MSE | 7,853,676,624.65 |
| R² Score | **−0.0239** |

This is worth sitting with for a moment rather than glossing over: a negative R² means the plain Linear Regression model performs *worse* than simply predicting the average sale price for every house. Looking at the coefficients explains why:

| Feature | Coefficient |
|---|---|
| PoolArea | +276,582.04 |
| PoolQC_None | +246,036.51 |
| RoofMatl_CompShg | −132,333.52 |

`PoolArea` and `PoolQC_None` are carrying almost the same information — a house with no pool has `PoolArea = 0` *and* `PoolQC_None = 1` at the same time. That near-perfect overlap (multicollinearity) makes the underlying matrix inversion in OLS numerically unstable, and the two coefficients blow up to huge, opposite-signed values to compensate for each other. With 254 features on only 1168 training rows, the model simply doesn't have enough independent information to pin down stable coefficients.

### What a coefficient means (in principle)
A large **positive** coefficient means increasing that (scaled) feature pushes the predicted price up, holding everything else constant. A large **negative** coefficient means the opposite. In a well-behaved model these values are informative — here, they're mostly a symptom of instability rather than genuine feature importance.

## 6. Ridge Regression

| Metric | Value |
|---|---|
| MSE | 1,385,254,032.91 |
| R² Score | **0.8194** |

### Comparison Table

| Model | MSE | R² Score |
|---|---|---|
| Linear Regression | 7.853677e+09 | −0.0239 |
| Ridge Regression | 1.385254e+09 | 0.8194 |

This comparison is exactly why Ridge Regression exists. Ridge adds an L2 penalty that shrinks large coefficients toward zero, which directly counteracts the multicollinearity problem above — instead of `PoolArea` and `PoolQC_None` fighting each other with extreme values, Ridge keeps both modest and lets the model actually generalize. The jump from an unusable R² of −0.02 to a solid 0.82 is a clean illustration of what `alpha` (the regularization strength) buys you: a small amount of bias in exchange for a large reduction in variance. `alpha=1.0` was used here.

## 7. Logistic Regression — Checking for Class Imbalance First

Before jumping to imbalance-handling techniques, the actual class distribution in the training set was checked:

| Class | Count | % |
|---|---|---|
| 0 (lower-priced) | 586 | 50.17% |
| 1 (higher-priced) | 582 | 49.83% |

```
Minority class %: 49.83%
Classes already balanced -> SMOTE not required, using data as-is
```

Since the minority class is well above the 35% threshold, this dataset doesn't actually need imbalance correction — the median split naturally produces near-50/50 classes. Rather than silently skip this requirement, the code explicitly checks the percentage and only applies SMOTE if the imbalance is real; here it correctly recognized the classes were already balanced and trained on the data as-is.

## 8. Classification Performance (Baseline, C=1.0, threshold = 0.5)

| Metric | Value |
|---|---|
| Accuracy | 0.8801 |
| Precision | 0.8776 |
| Recall | 0.8836 |
| F1 Score | 0.8805 |

**Confusion Matrix** (derived from the metrics above, on 292 test rows with 146 actual positives and 146 actual negatives):

| | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | TN = 128 | FP = 18 |
| **Actual 1** | FN = 17 | TP = 129 |

## 9. ROC Curve and AUC

**AUC Score: 0.9598**

### Precision & Recall, spelled out
```
Precision = TP / (TP + FP) = 129 / (129 + 18) = 0.8776
Recall    = TP / (TP + FN) = 129 / (129 + 17) = 0.8836
```

### Which one matters more here?
For this task, **Recall** carries a bit more weight than Precision. A False Negative means a genuinely higher-priced house gets classified as lower-priced — effectively undervaluing it. That's a costlier mistake than the reverse (flagging a lower-priced house as higher-priced), so when tuning the decision threshold later, leaning toward recall makes practical sense.

### What the AUC means
0.9598 is very close to 1.0, which means the model is very good at ranking — if you picked one higher-priced house and one lower-priced house at random from the test set, the model would correctly rank the higher-priced one above the other about 96% of the time.

## 10. Decision Threshold Sensitivity

| Threshold | Precision | Recall | F1 Score |
|---|---|---|---|
| 0.30 | 0.8571 | 0.9041 | 0.8800 |
| 0.40 | 0.8562 | 0.8973 | 0.8763 |
| 0.50 | 0.8776 | 0.8836 | 0.8805 |
| 0.60 | 0.8819 | 0.8699 | 0.8759 |
| 0.70 | 0.8873 | 0.8630 | 0.8750 |

The **F1-maximizing threshold is 0.50** (F1 = 0.8805) — the default threshold happens to already be close to optimal here, with 0.30 a very close second (F1 = 0.8800) and noticeably better recall (0.9041 vs 0.8836). Given that recall was flagged as the priority metric in Section 9, there's a reasonable argument for nudging the threshold down to 0.30 in practice — it trades a small amount of precision for a meaningful recall gain, catching more of the genuinely higher-priced houses.

## 11. Regularization Experiment (C = 1.0 vs C = 0.01)

| Model | Precision | Recall | F1 Score | AUC |
|---|---|---|---|---|
| Logistic Regression (C=1.0) | 0.8873 | 0.8630 | 0.8750 | 0.9598 |
| Logistic Regression (C=0.01) | 0.9167 | 0.9041 | 0.9103 | 0.9742 |

`C` is the inverse of regularization strength, so a smaller value (0.01) applies much stronger regularization — it shrinks the model's coefficients more aggressively, forcing a simpler decision boundary. Every single metric improved when regularization was strengthened, which is a sign that the C=1.0 model was mildly overfitting to the training data (consistent with the coefficient instability seen back in Section 5) — the simpler C=0.01 model generalizes noticeably better on unseen data.

## 12. Bootstrap Confidence Interval for the AUC Difference

To check whether the C=0.01 model's advantage is a reliable pattern or just a lucky split, 500 bootstrap samples were drawn from the test set (`np.random.seed(42)`), and the AUC difference (C=1.0 minus C=0.01) was recomputed on each resample.

| Statistic | Value |
|---|---|
| Mean AUC Difference | −0.0144 |
| 2.5th Percentile | −0.0267 |
| 97.5th Percentile | −0.0052 |

The 95% confidence interval, **[−0.0267, −0.0052], does not include zero.** Since the entire interval sits below zero, this isn't just noise from one particular split — the C=0.01 model reliably outperforms the C=1.0 model across resampled versions of the test set, which gives real confidence in preferring the more regularized model going forward.




# Part 3 – Advanced Modeling, Ensemble Learning and ML Pipeline

## Objective

After developing baseline regression and classification models in Part 2, the next objective was to evaluate more powerful machine learning algorithms and determine which model performs best on unseen data.

This part focuses on improving predictive performance using ensemble learning techniques, hyperparameter tuning, cross-validation, and a fully reproducible machine learning pipeline.

The workflow includes:

- Decision Tree baseline
- Controlled Decision Tree
- Gini vs Entropy comparison
- Random Forest
- Feature importance analysis
- Feature ablation study
- Gradient Boosting
- Cross-validation
- Hyperparameter tuning using GridSearchCV
- Manual learning curve analysis
- Model serialization
- Final model comparison

The best-performing model is finally saved for deployment and later reused in Part 4.

---

# 1. Decision Tree Baseline

A baseline Decision Tree Classifier was trained using the default sklearn parameters.

```python
DecisionTreeClassifier(random_state=42)
```

Both training accuracy and testing accuracy were calculated.

### Interpretation

The unconstrained Decision Tree achieved extremely high training accuracy but noticeably lower testing accuracy.

This behaviour indicates **overfitting**.

The model learns many small patterns and even noise present in the training data, allowing it to perform exceptionally well on familiar data while generalizing poorly to unseen observations.

Decision Trees are often described as **high-variance models** because every split is selected greedily based on the current subset of data, and earlier decisions are never revisited. Small changes in the training dataset can therefore produce very different tree structures.

---

# 2. Controlled Decision Tree

To reduce overfitting, a second Decision Tree was trained with controlled complexity.

```python
DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=20,
    random_state=42
)
```

Training accuracy and testing accuracy were again evaluated.

### Why These Parameters?

**max_depth**

Limits how deep the tree is allowed to grow.

Restricting tree depth prevents the model from memorizing individual observations and usually improves generalization.

**min_samples_split**

Requires a minimum number of samples before creating another split.

This prevents unnecessary splits on very small subsets that are often caused by random noise.

### Interpretation

Compared with the unconstrained tree, the controlled tree showed a smaller gap between training and testing accuracy.

Although the training accuracy decreased slightly, the testing performance became more reliable, indicating improved generalization and reduced overfitting.



# 3. Gini vs Entropy Comparison

To compare different splitting criteria, two Decision Tree models were trained using the same hyperparameters while changing only the impurity measure.

```python
DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)

DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    random_state=42
)
```

The test accuracy of both models was recorded and compared.

### Gini Impurity

The Gini Impurity used by CART Decision Trees is defined as:

\[
Gini = 1 - \sum p_i^2
\]

where \(p_i\) represents the probability of each class within a node.

### Entropy

Entropy measures the amount of uncertainty present in a node.

\[
Entropy = - \sum p_i \log_2(p_i)
\]

### Interpretation

A node with **Gini = 0** is considered perfectly pure because every observation belongs to a single class.

Both Gini and Entropy aim to produce pure child nodes, although they calculate impurity differently.

In this project, both criteria produced very similar predictive performance. The difference in accuracy was minimal, indicating that either impurity measure can effectively separate the classes for this dataset.

---

# 4. Random Forest Classifier

A Random Forest classifier was trained to improve predictive performance through ensemble learning.

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
```

The following evaluation metrics were computed:

- Training Accuracy
- Test Accuracy
- ROC-AUC Score

Compared with a single Decision Tree, Random Forest produced more stable predictions and showed better generalization on unseen data.

### Feature Importance

Feature importance values were obtained using:

```python
model.feature_importances_
```

The five most important features along with their importance scores were identified and ranked.

### Interpretation

Random Forest measures feature importance based on the **average reduction in Gini impurity** contributed by each feature across all trees in the forest.

Features that consistently produce better splits receive higher importance scores because they contribute more toward reducing classification uncertainty.

Unlike Linear Regression coefficients, feature importance values do not indicate whether a feature increases or decreases the prediction. Instead, they only measure how useful a feature is for making accurate splits during tree construction.

### Bagging Concept

Random Forest improves prediction accuracy using **Bootstrap Aggregating (Bagging).**

Each tree is trained on a different bootstrap sample created by randomly selecting training observations **with replacement**.

Additionally, at every split, only a random subset of approximately **√(number of features)** is considered instead of all available features.

Because every tree learns from slightly different data and different subsets of features, the individual prediction errors become less correlated.

The final prediction is obtained by combining predictions from all trees, which significantly reduces variance and makes Random Forest much less prone to overfitting than a single deep Decision Tree.

---

# 5. Gradient Boosting Classifier

A Gradient Boosting classifier was also trained for comparison.

```python
GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
```

The following performance metrics were evaluated:

- Training Accuracy
- Test Accuracy
- ROC-AUC Score

### Interpretation

Unlike Random Forest, which trains many trees independently, Gradient Boosting builds trees sequentially.

Each new tree attempts to correct the prediction errors made by the previous trees.

Because every successive model focuses on reducing the remaining errors, Gradient Boosting often achieves higher predictive accuracy on structured tabular datasets.

However, this sequential learning process also makes the algorithm more sensitive to hyperparameter selection and increases training time compared with Random Forest.

The Gradient Boosting model was later included in the cross-validation comparison to evaluate whether its improved complexity translated into better generalization performance.


# 6. Feature Ablation Study

To understand whether all input features were contributing meaningfully to the model, a feature ablation study was performed.

Using the feature importance scores obtained from the Random Forest model, the five least important features were identified.

A second Random Forest model was then trained after removing these five features while keeping all other hyperparameters unchanged.

The ROC-AUC scores of both models were compared.

- Full Model (All Features)
- Reduced Model (Lowest 5 Features Removed)

### Interpretation

If the reduced model achieves nearly the same ROC-AUC as the full model, it suggests that the removed features contributed very little useful information and mainly added complexity.

On the other hand, if the ROC-AUC decreases noticeably after removing those features, they still provide useful predictive information despite having relatively low importance scores.

In this project, the comparison helped determine whether a simpler model could be deployed without sacrificing predictive performance.

Reducing unnecessary features can lower computational cost, simplify maintenance, and improve inference speed, provided that predictive accuracy remains nearly unchanged.

---

# 7. Cross-Validated Model Comparison

Instead of relying on a single train-test split, all major classification models were evaluated using **5-fold Stratified Cross-Validation**.

The following models were compared:

- Logistic Regression
- Controlled Decision Tree
- Random Forest
- Gradient Boosting

Cross-validation was performed using:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Each model was evaluated using **ROC-AUC** as the scoring metric.

For every model, the following statistics were reported:

- Mean ROC-AUC
- Standard Deviation of ROC-AUC

### Why Cross-Validation?

A single train-test split may accidentally produce optimistic or pessimistic results depending on how the data is divided.

Cross-validation repeatedly trains and evaluates the model on different subsets of the data.

This provides a much more reliable estimate of how the model is expected to perform on unseen data.

The standard deviation of the cross-validation scores also indicates the stability of the model.

A lower standard deviation suggests that the model performs consistently across different subsets of the dataset.

---

# 8. Hyperparameter Tuning using GridSearchCV

To obtain the best-performing Random Forest model, hyperparameter tuning was performed using **GridSearchCV**.

A machine learning pipeline was created to ensure that preprocessing and model training occurred together without introducing data leakage.

```python
Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("randomforestclassifier", RandomForestClassifier(random_state=42))
])
```

The following hyperparameters were explored:

- n_estimators = [50, 100, 200]
- max_depth = [5, 10, None]
- min_samples_leaf = [1, 5]

The pipeline was evaluated using:

- 5-fold Stratified Cross-Validation
- ROC-AUC scoring

After completion, the following outputs were recorded:

- Best Hyperparameters
- Best Cross-Validation ROC-AUC Score

### Number of Models Evaluated

The parameter grid contains:

- 3 values for n_estimators
- 3 values for max_depth
- 2 values for min_samples_leaf

Total combinations:

```
3 × 3 × 2 = 18 models
```

Since each combination was evaluated using **5-fold Cross-Validation**, the total number of model fits performed was:

```
18 × 5 = 90 model trainings
```

### Grid Search vs Randomized Search

Grid Search evaluates every possible hyperparameter combination, making it thorough but computationally expensive.

Randomized Search evaluates only a random subset of combinations, making it much faster for large search spaces while often achieving comparable performance.

For this project, Grid Search was selected because the parameter grid was relatively small and exhaustive evaluation was feasible.

---

# 9. Manual Learning Curve Analysis

To understand how training data size affects model performance, a manual learning curve was created using the best pipeline obtained from GridSearchCV.

The model was trained using progressively larger portions of the training data:

- 20%
- 40%
- 60%
- 80%
- 100%

For each training fraction, two metrics were calculated:

- Training ROC-AUC
- Test ROC-AUC

### Interpretation

As expected, the training ROC-AUC gradually decreased as more training samples were included.

Small datasets are easier for a model to memorize, resulting in very high training performance.

As additional training data became available, memorization became more difficult, causing training performance to decrease slightly.

Meanwhile, the test ROC-AUC generally improved as more data was used for training.

This indicates that the model benefited from additional training examples and learned more generalized decision boundaries.

If the test ROC-AUC continues increasing even at 100% of the available training data, collecting more data would likely improve future model performance.

If the test ROC-AUC has already plateaued, model improvements are more likely to come from better algorithms or feature engineering rather than additional data.

---

# 10. Model Serialization

The best-performing pipeline obtained from GridSearchCV was saved for future use.

```python
joblib.dump(best_pipeline, "best_model.pkl")
```

The saved model was later reloaded using:

```python
loaded_model = joblib.load("best_model.pkl")
```

The reloaded model was then used to make predictions on manually created test samples.

Successfully loading the model and generating predictions confirmed that the serialization process preserved the complete preprocessing pipeline as well as the trained machine learning model.

Saving trained models eliminates the need to retrain the model every time predictions are required and makes deployment significantly easier.

---

# 11. Final Model Comparison

All classification models developed in Parts 2 and 3 were compared using the following evaluation metrics:

- 5-Fold Cross-Validation Mean ROC-AUC
- 5-Fold Cross-Validation Standard Deviation
- Test ROC-AUC

This comparison provided a fair assessment of each model's predictive performance and consistency.

### Final Recommendation

Among all evaluated models, the model achieving the highest cross-validated ROC-AUC while maintaining stable performance across folds was selected as the final recommendation.

Although simpler models such as Logistic Regression are easier to interpret, ensemble methods demonstrated stronger predictive capability by capturing more complex relationships within the data.

Considering predictive accuracy, robustness, and generalization performance, the tuned ensemble model was selected as the best solution for deployment.

This serialized model (`best_model.pkl`) is reused in **Part 4** to provide LLM-powered prediction explanations.



# Part 4 – LLM Powered Intelligent Feature

## Objective

The final phase of this project integrates a Large Language Model (LLM) with the machine learning pipeline developed in the previous parts.

Instead of only predicting whether a house belongs to the higher-priced or lower-priced category, the LLM provides a structured explanation describing why the prediction was made.

This layer improves the interpretability of the machine learning model and demonstrates how traditional predictive models can be combined with modern language models to generate human-readable insights.

For this project, **Track C — Model Prediction Explanation Pipeline** was selected.

The workflow consists of:

- Loading the best trained model
- Generating predictions and prediction probabilities
- Sending model outputs to an LLM
- Receiving structured JSON explanations
- Validating the JSON responses
- Applying safety guardrails before every API call

---

# 1. LLM API Connection

The LLM was accessed through an HTTP API using the Python **requests** library.

For security reasons, the API key was **not hardcoded** inside the notebook.

Instead, it was stored as an environment variable.

```python
os.environ["LLM_API_KEY"]
```

This approach prevents accidental exposure of sensitive credentials when sharing the repository publicly.

A reusable helper function named `call_llm()` was implemented.

The function performs the following operations:

- Creates the JSON request payload
- Adds authentication headers
- Sends an HTTP POST request
- Checks the response status code
- Extracts the generated response
- Returns only the model output

Before integrating the function into the complete pipeline, a simple test prompt was executed requesting the model to return only the word **"hello"**.

The successful response confirmed that the API connection was functioning correctly.

---

# 2. Prompt Engineering

To obtain consistent and structured explanations, a carefully designed system prompt was used.

### System Prompt

The LLM was instructed to behave as a machine learning prediction explanation assistant.

Its responsibilities included:

- Explaining predictions
- Producing only valid JSON
- Avoiding unnecessary text
- Providing concise reasoning
- Returning a consistent output structure

The user prompt contained:

- Feature values
- Predicted class
- Prediction probability

The model was instructed to generate a JSON explanation containing fields such as:

- prediction_label
- confidence_level
- top_reason
- second_reason
- next_step

### Why Temperature = 0?

All LLM requests used:

```python
temperature = 0
```

A temperature of zero makes the model deterministic.

This means that identical inputs consistently produce nearly identical outputs.

Such deterministic behavior is essential for structured JSON generation because unpredictable responses increase the likelihood of parsing or validation errors.

---

# 3. Temperature Comparison

To study the effect of randomness, every test input was evaluated twice.

The first request used:

```python
temperature = 0
```

The second request used:

```python
temperature = 0.7
```

The outputs from both settings were compared.

### Interpretation

Responses generated with **temperature = 0** remained highly consistent and followed the requested JSON schema very closely.

Responses generated with **temperature = 0.7** were more diverse and occasionally used different wording or additional explanations.

While higher temperatures may improve creativity, they also increase variability.

For structured prediction explanations, deterministic outputs are more reliable and therefore temperature 0 was selected for the final implementation.


# 4. Structured Output Validation

The responses generated by the LLM were expected to follow a predefined JSON structure so that they could be processed automatically by the application.

To ensure consistency, a JSON schema containing the required fields was defined before making any API calls.

The schema included the following required fields:

- prediction_label
- confidence_level
- top_reason
- second_reason
- next_step

Each field was defined as a scalar value, ensuring that every explanation returned by the LLM followed the same predictable format.

After receiving the response from the API, leading and trailing whitespace was removed using:

```python
response.strip()
```

The cleaned response was then converted into a Python dictionary using:

```python
json.loads()
```

Since LLMs may occasionally generate malformed JSON, the parsing process was wrapped inside a `try-except` block.

If the response could not be parsed successfully, a `JSONDecodeError` was caught and a fallback dictionary containing null values was returned instead of stopping the program.

After successful parsing, the output was validated using:

```python
jsonschema.validate()
```

This ensured that all required fields were present and matched the expected data types.

Any validation failures were captured using a `ValidationError`, and the same fallback dictionary was returned.

Using schema validation makes the pipeline significantly more reliable because downstream code always receives data in a predictable format.

---

# 5. Model Prediction Explanation Pipeline

The best-performing machine learning model obtained in Part 3 was loaded using:

```python
joblib.load("best_model.pkl")
```

Three manually created feature vectors were prepared to evaluate the complete prediction pipeline.

For each input record, the following sequence was performed:

- Encode the feature values
- Predict the output class
- Predict the class probability
- Construct the LLM prompt
- Send the prompt to the API
- Receive the structured explanation
- Parse the JSON response
- Validate the schema
- Display the final explanation

Each prompt supplied to the language model contained:

- Input feature values
- Predicted class
- Prediction probability

The LLM then generated a structured explanation describing why the model produced its prediction and what the prediction implies.

This approach combines the predictive capability of machine learning with the interpretability provided by large language models.

---

# 6. PII Guardrail

Before every request was sent to the LLM, the user input was checked for personally identifiable information (PII).

Regular expressions were used to detect common sensitive information such as:

- Email addresses
- Phone numbers

If sensitive information was detected, the API request was blocked immediately.

Instead of calling the language model, the application displayed:

```
Input blocked: PII detected.
```

This prevents confidential information from being transmitted to external services.

Two demonstrations were included in the notebook.

### Test Case 1

Input containing an email address.

Result:

```
Blocked
```

The API call was not executed.

### Test Case 2

Input containing no sensitive information.

Result:

```
Allowed
```

The prediction explanation pipeline executed normally.

Implementing this guardrail improves privacy protection and demonstrates responsible use of LLM APIs.

---

# 7. End-to-End Demonstration

The complete prediction explanation pipeline was executed on three different input records.

For every record, the notebook displayed:

- Feature values
- Predicted class
- Prediction probability
- Raw LLM response
- Parsed JSON
- Validation result

Each generated explanation successfully followed the predefined schema, allowing the responses to be processed automatically without manual correction.

This demonstrated that the entire workflow—from machine learning prediction to LLM-generated explanation—operated correctly from start to finish.

---

# Part 4 Summary

The final stage of this project demonstrated how Large Language Models can be integrated with traditional machine learning systems to improve interpretability.

Unlike conventional models that only return numerical predictions, the LLM generated structured explanations describing the reasoning behind each prediction in an easy-to-understand format.

Several practical software engineering practices were incorporated into the implementation, including secure API key management, deterministic prompt engineering, JSON schema validation, exception handling, and privacy guardrails.

Together, these additions make the prediction pipeline more reliable, secure, and suitable for real-world deployment scenarios where both prediction accuracy and explainability are equally important.



# Installation

Clone the repository using Git:

```bash
git clone https://github.com/Rashid-imam/Applied_AI_ML_Capstone.git
```

Navigate to the project directory:

```bash
cd Applied_AI_ML_Capstone
```

Install all required Python dependencies:

```bash
pip install -r requirements.txt
```

If you are using Jupyter Notebook, launch it using:

```bash
jupyter notebook
```

---

# Project Execution

Run the notebooks in the following sequence to reproduce the complete workflow:

1. **Part1_EDA.ipynb**
   - Loads the raw dataset
   - Performs data cleaning
   - Handles missing values and duplicates
   - Generates exploratory data analysis (EDA)
   - Saves the cleaned dataset as `cleaned_data.csv`

2. **Part2_ML.ipynb**
   - Loads `cleaned_data.csv`
   - Trains Regression and Classification models
   - Performs model evaluation
   - Generates performance metrics and ROC curve

3. **Part3_AdvancedModels.ipynb**
   - Builds ensemble models
   - Performs feature ablation study
   - Executes GridSearchCV
   - Creates the best machine learning pipeline
   - Saves the trained model as `best_model.pkl`

4. **Part4_LLM.ipynb**
   - Loads `best_model.pkl`
   - Connects with the OpenRouter LLM API
   - Generates structured prediction explanations
   - Validates JSON responses
   - Demonstrates privacy guardrails

Each notebook depends on the outputs generated by the previous notebook, so they should be executed in the above order.

---

# Repository Structure

```
Applied_AI_ML_Capstone/
│
├── data/
│   ├── train.csv
│   └── cleaned_data.csv
│
├── models/
│   └── best_model.pkl
│
├── notebooks/
│   ├── Part1_EDA.ipynb
│   ├── Part2_ML.ipynb
│   ├── Part3_AdvancedModels.ipynb
│   └── Part4_LLM.ipynb
│
├── README.md
└── requirements.txt
```

---

# Technologies Used

## Programming Language

- Python 3.x

## Data Analysis

- Pandas
- NumPy

## Data Visualization

- Matplotlib
- Seaborn

## Machine Learning

- Scikit-learn
- Imbalanced-learn (SMOTE)

## Model Serialization

- Joblib

## LLM Integration

- Requests
- JSON
- JSONSchema
- Regular Expressions (re)
- OpenRouter API

## Development Tools

- Jupyter Notebook
- Visual Studio Code
- Git
- GitHub

---

# Learning Outcomes

This project provided practical experience with the complete machine learning lifecycle.

Major concepts implemented during the project include:

- Data Cleaning and Preprocessing
- Missing Value Imputation
- Exploratory Data Analysis (EDA)
- Statistical Analysis
- Correlation Analysis
- Feature Engineering
- Linear Regression
- Ridge Regression
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Cross Validation
- Hyperparameter Tuning
- GridSearchCV
- Feature Importance Analysis
- Model Serialization using Joblib
- LLM Integration using OpenRouter API
- Prompt Engineering
- JSON Schema Validation
- Privacy Guardrails using Regular Expressions

Overall, this project demonstrates a complete end-to-end Artificial Intelligence and Machine Learning workflow starting from raw data preprocessing to explainable predictions using a Large Language Model.

---

# Author

**Name:** Rashid Imam

**Course:** Applied AI & Machine Learning

**Project:** End-to-End Data Intelligence System

**Institution:** Masai School

**GitHub:** https://github.com/Rashid-imam

**Project Repository:** https://github.com/Rashid-imam/Applied_AI_ML_Capstone
---

