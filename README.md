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

# Dataset

## Dataset Name

House Prices – Advanced Regression Techniques

## Source

Kaggle

## Dataset Description

The dataset contains detailed information about residential houses along with multiple numerical and categorical attributes describing their physical characteristics, construction quality, location, size, utilities, and selling price.

The target variable used for regression is:

SalePrice

which represents the final selling price of each property.

The dataset satisfies all assignment requirements:

- More than 500 rows
- More than 5 numerical features
- More than 2 categorical features
- Suitable for both regression and binary classification

For the classification task, a binary target was created by comparing SalePrice with its median value.

PriceCategory = (SalePrice > SalePrice.median()).astype(int)

This creates two balanced classes representing lower-priced and higher-priced houses.

## Dataset Justification

This dataset was selected because it provides a realistic real-estate prediction problem involving both numerical and categorical variables.

It contains missing values, skewed numerical distributions, categorical attributes, and outliers, making it suitable for demonstrating data cleaning, exploratory data analysis, feature engineering, regression, classification, ensemble learning, hyperparameter tuning, and LLM integration.

The dataset is widely used in machine learning research and therefore provides an excellent benchmark for evaluating different predictive models.

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

The dataset was loaded into a Pandas DataFrame using:

```python
pd.read_csv()
```

After loading the dataset, the following inspections were performed:

- Displayed the first five rows using head()
- Printed dataset dimensions using shape
- Displayed column data types using dtypes

These initial checks helped verify that the dataset was imported successfully and allowed identification of columns requiring further preprocessing.


## 2. Missing Value Analysis

Missing values were identified using:

```python
df.isnull().sum()
```

The percentage of missing values for every column was calculated as:

```python
(df.isnull().sum()/len(df))*100
```

Columns containing more than 20% missing values were identified separately for further consideration.

For numerical columns containing less than 20% missing values, missing entries were imputed using the column median.

### Why Median Instead of Mean?

The dataset contains several highly skewed numerical variables.

The mean is strongly influenced by extreme observations (outliers), whereas the median remains much more stable even when unusually high or low values exist.

Since housing datasets typically contain expensive luxury properties that create right-skewed distributions, median imputation provides a more representative estimate of the central tendency than the mean.

This approach reduces bias introduced by outliers while preserving the overall distribution of the data.


## 3. Duplicate Detection and Removal

Duplicate observations were identified using:

```python
df.duplicated().sum()
```

Detected duplicate rows were removed using:

```python
df.drop_duplicates()
```

After duplicate removal:

- Total duplicate rows removed were reported.
- Dataset shape was verified again.
- Missing value percentages were recalculated to confirm whether duplicate removal affected null-value statistics.

Removing duplicate records prevents the model from learning repeated information and helps improve generalization performance.


## 4. Data Type Correction

Several columns required data type optimization.

The following preprocessing steps were performed:

### Numeric Conversion

Columns stored as object but representing numeric values were converted using:

```python
pd.to_numeric(errors="coerce")
```

This safely converted invalid values into missing values for later processing.

### Category Conversion

Highly repetitive string columns were converted into the category datatype.

Example categorical columns include:

- Neighborhood
- MSZoning
- SaleCondition

Converting these columns significantly reduces memory usage because repeated text values are internally stored only once.

Memory usage before and after optimization was measured using:

```python
df.memory_usage(deep=True).sum()
```

The optimized dataset required substantially less memory while preserving identical information.


## 5. Descriptive Statistics

Descriptive statistics were generated for all numerical variables using:

```python
df.describe()
```

The following statistics were reviewed:

- Count
- Mean
- Standard Deviation
- Minimum
- 25th Percentile
- Median
- 75th Percentile
- Maximum

These statistics provide an overview of each variable's distribution and highlight unusually large ranges or potential outliers.


## 6. Skewness Analysis

The skewness of every numerical feature was calculated using:

```python
df[col].skew()
```

The feature with the highest absolute skewness was identified for detailed analysis.

### Interpretation

Positive skewness indicates that the distribution contains a long right tail. Most observations are concentrated at lower values, while a few extremely large observations pull the distribution toward the right.

Negative skewness indicates that the distribution contains a long left tail. Most observations lie toward higher values while a few extremely small observations pull the distribution toward the left.

Because highly skewed variables are influenced by extreme observations, the median provides a more reliable estimate of central tendency than the mean. Therefore, median imputation was preferred for skewed numerical features.

The histogram produced later confirms the distribution pattern of the most skewed feature.


## 7. Outlier Detection using Interquartile Range (IQR)

Outliers were identified using the Interquartile Range (IQR) method.

For each selected numerical feature:

- Q1 (25th percentile)
- Q3 (75th percentile)
- IQR = Q3 − Q1
- Lower Bound = Q1 − 1.5 × IQR
- Upper Bound = Q3 + 1.5 × IQR

Rows falling outside these bounds were counted as potential outliers.

### Interpretation

Outliers were intentionally **not removed** because house-price datasets naturally contain premium and luxury properties whose values are genuinely high rather than erroneous.

Removing these observations could reduce the model's ability to learn the complete price distribution.

Instead, these outliers were documented and retained for Part 2, where machine learning models such as Ridge Regression and ensemble methods are expected to handle them more robustly.


## 8. Exploratory Data Analysis (EDA)

Several visualization techniques were used to better understand the dataset.

### Line Plot

A line plot was generated for a numerical variable to observe changes across the dataset.

The visualization helped identify overall trends, fluctuations, and unusual observations.

---

### Bar Chart

A bar chart was created to compare the average value of a numerical feature across categories.

This visualization highlighted how different categories contribute differently to the selected numerical variable.

---

### Histogram

A histogram of the most skewed numerical feature was plotted using 20 bins.

The histogram confirmed the skewed distribution identified through skewness statistics.

The long tail indicates the presence of relatively few extreme observations compared to the majority of samples.

---

### Scatter Plot

A scatter plot was generated between two numerical variables expected to have a relationship.

The plot demonstrated the direction and approximate strength of correlation.

A positive trend indicates that as one variable increases, the other generally increases as well.

---

### Box Plot

A box plot compared a numerical feature across categorical groups.

The plot revealed:

- Differences in median values
- Differences in spread (IQR)
- Presence of outliers
- Variation among different categories

This visualization helps understand whether categorical variables have predictive power.


## 9. Pearson Correlation Analysis

Pearson correlation coefficients were calculated for all numerical variables using:

```python
df.corr()
```

The resulting correlation matrix was visualized using a heatmap.

### Interpretation

The heatmap highlights both strong positive and strong negative linear relationships between variables.

The pair with the highest absolute correlation was identified for further analysis.

Although a strong correlation suggests an association, it does **not** necessarily imply causation.

A third variable may simultaneously influence both correlated variables.

For example, larger houses often have more rooms and higher sale prices.

Therefore, the observed correlation may actually arise because house size influences both variables rather than one directly causing the other.


## 10. Mean vs Median Imputation Comparison

For the two numerical variables with the highest absolute skewness, both the mean and median were calculated before imputation.

The comparison demonstrated that the mean shifted toward extreme observations, whereas the median remained much more stable.

Based on the skewness direction, median imputation was selected because:

- Positive skew → mean pulled upward by very high values.
- Negative skew → mean pulled downward by very low values.

After imputation, missing-value counts were verified again using:

```python
df.isnull().sum()
```

The output confirmed that no missing values remained in the selected numerical features.


## 11. Spearman Rank Correlation

Along with Pearson correlation, Spearman Rank Correlation was also computed.

Unlike Pearson correlation, Spearman measures monotonic relationships using ranked values instead of raw values.

The absolute difference between Pearson and Spearman coefficients was calculated for every feature pair.

The three feature pairs with the largest differences were identified.

### Interpretation

If

|Spearman| > |Pearson|

the relationship is likely monotonic but non-linear.

If

|Pearson| ≥ |Spearman|

the relationship is approximately linear.

For feature selection in later machine learning models, Pearson correlation was used for identifying linear relationships, while Spearman correlation provided additional insight into monotonic relationships that Pearson may underestimate.


## 12. Grouped Aggregation Analysis

Grouped statistical analysis was performed using:

```python
df.groupby(categorical_column)[numeric_column].agg(['mean','std','count'])
```

For each category, the following statistics were calculated:

- Mean
- Standard Deviation
- Sample Count

### Interpretation

The category with the highest mean indicates the group having the largest average value for the selected numerical feature.

The category with the highest standard deviation exhibits the greatest internal variability.

High within-group variability suggests that the categorical feature alone cannot completely explain differences in the numerical variable.

The ratio between the highest group mean and the lowest group mean was also calculated.

A large ratio indicates that the categorical feature carries useful predictive information for future machine learning models.


## 13. Saving the Clean Dataset

After completing all preprocessing and exploratory analysis steps, the cleaned dataset was saved using:

```python
df.to_csv("cleaned_data.csv", index=False)
```

This dataset serves as the standardized input for Part 2 (Supervised Machine Learning) and Part 3 (Advanced Modeling).



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

## 1. Loading the Clean Dataset

The cleaned dataset generated during Part 1 was loaded into a Pandas DataFrame using:

```python
pd.read_csv("cleaned_data.csv")
```

Using the cleaned dataset ensures that all preprocessing tasks—including missing value handling, duplicate removal, datatype correction, and exploratory analysis—have already been completed before model training begins.

This approach keeps the machine learning workflow reproducible and prevents repeating unnecessary preprocessing steps in later stages of the project.

After loading the dataset, its dimensions and data types were verified once again to confirm that the exported file was successfully imported without any modification.

---

## 2. Target Variable Definition

Two separate prediction problems were created from the cleaned dataset.

### Regression Target

The continuous target selected for the regression task was:

**SalePrice**

The regression model attempts to predict the actual selling price of a house based on its features.

---

### Classification Target

A binary classification target was created by comparing each house's selling price with the median selling price of the dataset.

```python
PriceCategory = (SalePrice > SalePrice.median()).astype(int)
```

This transformation divides the dataset into two classes:

- **0 → Lower-priced house**
- **1 → Higher-priced house**

Using the median creates nearly balanced classes, making the classification task more suitable for Logistic Regression and reducing bias toward any single class.

Both target definitions were used throughout the remaining experiments in this project.

---

## 3. Encoding Categorical Features

Machine learning algorithms require numerical input features. Since the dataset contains both numerical and categorical variables, categorical features were encoded before model training.

Two different encoding techniques were applied depending on the nature of the categorical variable.

### Ordinal Encoding

Categorical variables having a meaningful order were encoded using integer values that preserve their natural ranking.

For example:

```
Poor < Fair < Average < Good < Excellent
```

Maintaining this ordering allows the model to understand the relative relationship between categories.

---

### One-Hot Encoding

Categorical variables without any natural order were converted into dummy variables using:

```python
pd.get_dummies(drop_first=True)
```

Dropping the first dummy variable helps prevent multicollinearity by avoiding redundant information.

One-Hot Encoding treats every category independently and prevents the model from incorrectly assuming that one category is numerically larger or smaller than another.

This makes it the preferred encoding technique for nominal categorical variables such as neighborhood names, zoning types, or sale conditions.

---


## 4. Train-Test Split

Before training any machine learning model, the cleaned dataset was divided into separate training and testing subsets.

The split was performed using:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
```

The dataset was divided as follows:

- **80% Training Data**
- **20% Testing Data**

The training set was used to learn the model parameters, while the testing set was kept completely unseen during training and used only for final evaluation.

A fixed `random_state=42` was used to ensure that the same train-test split can be reproduced whenever the notebook is executed again.

Keeping an independent test set provides a realistic estimate of how well the trained model is expected to perform on unseen data.

---

## 5. Feature Scaling

Many machine learning algorithms perform better when numerical features are on a similar scale.

Therefore, numerical features were standardized using **StandardScaler** from Scikit-learn.

```python
scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

### Why was the scaler fitted only on the training data?

The scaler was intentionally fitted **only on the training dataset**.

The same fitted scaler was then used to transform both the training and testing datasets.

This prevents **data leakage**, which occurs when information from the test set accidentally influences the training process.

If the scaler had been fitted on the complete dataset before splitting, the mean and standard deviation of the testing samples would already be known during training.

This would make the evaluation unrealistically optimistic because the model would indirectly receive information from data that should remain completely unseen.

Using training statistics only ensures that the evaluation reflects real-world model performance on new data.

---

## 6. Linear Regression

The first predictive model developed for the regression task was **Linear Regression**.

The model was trained using the standardized training features.

```python
LinearRegression()
```

Predictions were generated on the test dataset and evaluated using the following performance metrics:

- Mean Squared Error (MSE)
- Coefficient of Determination (R² Score)

The model coefficients were extracted together with their corresponding feature names.

The three features having the largest absolute coefficient values were identified because they contribute the most toward predicting the selling price.

### Interpretation of Regression Coefficients

A **positive coefficient** indicates that increasing the corresponding standardized feature increases the predicted house price while all other variables remain constant.

A **negative coefficient** indicates that increasing the feature decreases the predicted selling price when the remaining features are unchanged.

Features with larger absolute coefficient values have a stronger influence on the model's predictions.

Although Linear Regression provides an easily interpretable baseline model, it can become unstable when highly correlated features are present.

For this reason, Ridge Regression was also evaluated.

---

## 7. Ridge Regression

To reduce the effect of multicollinearity and overly large coefficients, a Ridge Regression model was trained using:

```python
Ridge(alpha=1.0)
```

The same training-testing split and standardized features used for Linear Regression were reused to ensure a fair comparison.

Model performance was evaluated using:

- Mean Squared Error (MSE)
- R² Score

The evaluation results of Ridge Regression were compared directly with those obtained from ordinary Linear Regression.

### Comparison with Linear Regression

A comparison table was prepared in the README containing:

| Model | Mean Squared Error | R² Score |
|-------|--------------------|----------|
| Linear Regression | 2.498443 | 0.674272 |
| Ridge Regression | 2.120632 | 0.723528 |

(The actual values are generated automatically when the notebook is executed.)

### Why did Ridge Regression produce different results?

Unlike ordinary Linear Regression, Ridge Regression introduces an **L2 regularization penalty** that discourages excessively large coefficient values.

Instead of allowing coefficients to grow freely, Ridge shrinks them toward zero, making the model more stable when several predictors are highly correlated.

This often improves generalization performance by reducing overfitting.

The **alpha** parameter controls the strength of regularization.

- A **small alpha** behaves similarly to ordinary Linear Regression.
- A **larger alpha** applies stronger coefficient shrinkage, reducing model complexity but potentially increasing bias.

Selecting an appropriate alpha value provides a balance between bias and variance, allowing the model to perform better on previously unseen data.


---

## 8. Logistic Regression

For the binary classification task, a **Logistic Regression** model was trained using the standardized training dataset.

The model was initialized with:

```python
LogisticRegression(max_iter=1000)
```

A higher value of `max_iter` was used to ensure that the optimization algorithm converged successfully during training.

The trained model was then used to generate:

- Predicted class labels
- Predicted class probabilities

The predicted probabilities were later used for ROC Curve analysis, AUC calculation, threshold sensitivity analysis, and bootstrap confidence interval estimation.

---

## 9. Handling Class Imbalance

Before training the Logistic Regression model, the distribution of the target classes was examined using:

```python
y_train.value_counts()
```

The assignment requires checking whether one class contains fewer than **35%** of the training samples.

If significant imbalance exists, the model may become biased toward predicting the majority class.

To address this issue, one of the following approaches can be used:

- SMOTE (Synthetic Minority Over-sampling Technique)
- `class_weight="balanced"`

In this project, **class_weight="balanced"** was selected.

```python
LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)
```

### Why was class_weight="balanced" chosen?

Instead of creating synthetic observations, the algorithm automatically assigns larger penalties to mistakes made on the minority class.

This allows the model to pay more attention to underrepresented samples while preserving the original training data.

The class distribution before and after balancing was reported to demonstrate whether imbalance handling was required.

---

## 10. Classification Performance Evaluation

After training the Logistic Regression model, predictions were generated on the test dataset.

The following evaluation metrics were calculated:

- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1 Score

The complete classification report was generated using:

```python
classification_report()
```

The confusion matrix summarizes the prediction results into four categories:

- True Positive (TP)
- True Negative (TN)
- False Positive (FP)
- False Negative (FN)

This provides a more detailed evaluation than accuracy alone by identifying the types of prediction errors made by the model.

---

## 11. ROC Curve and AUC

Since Logistic Regression produces prediction probabilities, the Receiver Operating Characteristic (ROC) curve was plotted.

The ROC Curve was generated using:

```python
roc_curve()
```

The Area Under the Curve (AUC) was calculated using:

```python
roc_auc_score()
```

The ROC plot includes:

- False Positive Rate (FPR) on the x-axis
- True Positive Rate (TPR) on the y-axis
- AUC value displayed on the figure

### Interpretation of AUC

The Area Under the ROC Curve measures how effectively the model distinguishes between the two classes.

- An AUC close to **1.0** indicates excellent class separation.
- An AUC around **0.5** indicates performance similar to random guessing.
- Higher AUC values represent better classification performance across different decision thresholds.

In this project, the obtained AUC demonstrates the model's ability to distinguish between lower-priced and higher-priced houses based on the selected features.

---

## 12. Precision and Recall

Although overall accuracy is useful, it does not always provide a complete picture of classification performance.

Therefore, Precision and Recall were also evaluated.

### Precision Formula

```
Precision = TP / (TP + FP)
```

Precision measures the proportion of predicted positive observations that are actually positive.

A high Precision indicates that the model makes relatively few false positive predictions.

---

### Recall Formula

```
Recall = TP / (TP + FN)
```

Recall measures the proportion of actual positive observations that are correctly identified by the model.

A high Recall indicates that the model successfully detects most positive cases while producing fewer false negatives.

---

### Which Metric is More Important?

For this project, **Recall** was considered slightly more important than Precision.

The objective is to correctly identify as many higher-priced houses as possible.

Missing a genuinely high-value property (False Negative) may result in losing valuable information for pricing analysis.

Although increasing Recall may introduce some additional False Positives, detecting most valuable properties is generally more beneficial for this prediction task.

---

## 13. Decision Threshold Sensitivity Analysis

By default, Logistic Regression predicts the positive class whenever the predicted probability is greater than **0.50**.

To study the effect of different thresholds, prediction probabilities were generated using:

```python
model.predict_proba(X_test_scaled)[:,1]
```

The decision threshold was varied from:

- 0.30
- 0.40
- 0.50
- 0.60
- 0.70

For each threshold, the following metrics were calculated:

- Precision
- Recall
- F1 Score

The results were summarized in a comparison table.

| Threshold | Precision | Recall | F1 Score |
|-----------|-----------|---------|----------|
| 0.30 | 0.865385 | 0.924658 | 0.894040 |
| 0.40 | 0.867550 | 0.897260 | 0.882155 |
| 0.50 | 0.866667 | 0.890411 | 0.878378 |
| 0.60 | 0.877551 | 0.883562 | 0.880546 |
| 0.70 | 0.880282 | 0.856164 | 0.868056 |

(The values shown above are automatically generated when the notebook is executed.)

### Interpretation

Lowering the decision threshold increases Recall because more observations are classified as positive.

However, this also increases the number of False Positives, reducing Precision.

Increasing the threshold has the opposite effect:

- Precision increases.
- Recall decreases.

The threshold producing the highest F1 Score was selected as the best balance between Precision and Recall for this dataset.

Depending on the application, the threshold may be increased to prioritize Precision or decreased to prioritize Recall.

---

## 14. Regularization Experiment using Different C Values

To study the effect of regularization, a second Logistic Regression model was trained using:

```python
LogisticRegression(C=0.01)
```

This model was compared with the baseline model:

```python
LogisticRegression(C=1.0)
```

The following evaluation metrics were compared:

- Precision
- Recall
- ROC-AUC

A comparison table was included in the README.

| Model | Precision | Recall | ROC-AUC |
|--------|-----------|---------|----------|
| Logistic Regression (C=1.0) | 0.880282 | 0.856164 | 0.961953 |
| Logistic Regression (C=0.01) | 0.918919 | 0.931507 | 0.975605 |

### What does C control?

The parameter **C** controls the strength of regularization.

It is the inverse of regularization strength.

- Larger C → weaker regularization
- Smaller C → stronger regularization

Reducing C forces the model coefficients to become smaller, producing a simpler model that is generally less prone to overfitting.

The experimental results were analyzed to determine whether stronger regularization improved or reduced classification performance on this dataset.

---

## 15. Bootstrap Confidence Interval for AUC Difference

To evaluate whether the performance difference between the two Logistic Regression models was statistically reliable, a bootstrap analysis was performed.

A total of **500 bootstrap samples** were generated from the test dataset using sampling with replacement.

For each bootstrap sample:

- ROC-AUC was calculated for the baseline model (C=1.0).
- ROC-AUC was calculated for the strongly regularized model (C=0.01).
- The difference between both AUC values was recorded.

Finally, the following statistics were computed:

- Mean AUC Difference
- 2.5th Percentile
- 97.5th Percentile

These values form the **95% Bootstrap Confidence Interval**.

### Interpretation

If the confidence interval does **not** include zero, the observed performance difference is considered consistent across different samples and is therefore likely to be statistically reliable.

If the confidence interval **includes zero**, the difference between the two models may simply be due to sampling variability rather than a genuine improvement.

Bootstrap analysis provides additional confidence that the selected model consistently performs better instead of relying on a single train-test split.



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

