# House Price Prediction using Machine Learning and LLM

## Project Overview

This project was developed as part of the Applied AI / Machine Learning Capstone Assignment.

The objective of this project is to build a complete machine learning pipeline starting from raw data preprocessing to advanced model development and finally integrating a Large Language Model (LLM) powered feature.

The project is divided into four parts:

- Part 1 – Data Cleaning and Exploratory Data Analysis (EDA)
- Part 2 – Supervised Machine Learning
- Part 3 – Advanced Machine Learning Models
- Part 4 – LLM Powered Feature (Track C – Model Prediction Explanation Pipeline)

---

# Dataset

**Dataset Name**

House Prices – Advanced Regression Techniques

**Source**

Kaggle

The dataset contains residential house information with numerical and categorical features. The target variable is **SalePrice**, which represents the selling price of each house.

---

# Repository Structure

```
Applied_AI_ML_Capstone/
│
├── data/
│   ├── train.csv
│   └── cleaned_data.csv
│
├── notebooks/
│   ├── Part1_Data_Cleaning_EDA.ipynb
│   ├── Part2_Supervised_ML.ipynb
│   ├── Part3_Advanced_ML.ipynb
│   └── Part4_LLM_Powered_Feature.ipynb
│
├── models/
│   └── best_model.pkl
│
├── images/
│
└── README.md
```

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Requests
- JSON
- JSONSchema
- OpenRouter API
- GPT-OSS-20B (Free)

---

# Part 1 – Data Cleaning and Exploratory Data Analysis (EDA)

## Objective

The objective of Part 1 was to understand the dataset, identify data quality issues, perform preprocessing, handle missing values, detect duplicates and outliers, analyze feature distributions, and generate visualizations before building machine learning models.

---

## Steps Performed

The following preprocessing steps were performed:

- Loaded the dataset using Pandas.
- Displayed the first few records.
- Checked dataset dimensions.
- Inspected data types of all features.
- Identified missing values.
- Calculated missing value percentages.
- Filled missing numerical values using the median.
- Filled missing categorical values using the mode where appropriate.
- Checked duplicate records.
- Converted suitable columns to categorical data types.
- Generated descriptive statistics.
- Calculated skewness for numerical features.
- Compared mean and median for highly skewed columns.
- Detected outliers using the Interquartile Range (IQR) method.
- Created multiple visualizations for data exploration.
- Generated Pearson correlation matrix.
- Generated Spearman correlation matrix.
- Performed grouped aggregation analysis.
- Saved the cleaned dataset as `cleaned_data.csv`.

---

## Missing Value Handling

Missing values were handled according to feature type.

- Numerical columns were filled using the **median**.
- Categorical columns were filled using the **mode** whenever necessary.

Using different strategies for numerical and categorical variables helps preserve the statistical characteristics of the dataset.

---

## Why Median was Preferred over Mean

Many numerical features in the housing dataset are positively skewed and contain outliers.

The **mean** is sensitive to extreme values, whereas the **median** remains stable even when outliers are present.

Therefore, median imputation provides a more reliable estimate of central tendency for skewed numerical features.

---

## Skewness Analysis

Skewness was calculated for numerical features to understand their distributions.

- Positive skewness indicates a long right tail.
- Negative skewness indicates a long left tail.
- Features with high skewness were carefully analyzed before preprocessing.

Comparing the mean and median also helped identify skewed variables.

---

## Outlier Detection

Outliers were detected using the **Interquartile Range (IQR)** method.

Instead of removing all outliers, they were retained because many extreme observations represent genuine high-value houses rather than incorrect records.

Removing such observations could reduce the model's ability to learn important market patterns.

---

## Exploratory Data Analysis

Several visualizations were created to better understand the dataset.

The following plots were generated:

- Histogram
- Scatter Plot
- Box Plot
- Bar Chart
- Line Plot
- Correlation Heatmap

These visualizations helped identify feature distributions, relationships between variables, and potential outliers.

---

## Correlation Analysis

Two correlation methods were used.

### Pearson Correlation

Pearson correlation measures the strength of **linear relationships** between numerical variables.

### Spearman Correlation

Spearman correlation measures **rank-based monotonic relationships** and is more robust to non-linear relationships and outliers.

Using both methods provided a better understanding of feature relationships with the target variable.

---

## Grouped Aggregation

Grouped aggregation was performed using summary statistics such as:

- Mean
- Standard Deviation
- Count

This analysis helped compare different categories and understand variations among groups.

---

## Output

The cleaned dataset was saved as:

```text
cleaned_data.csv
```

This cleaned dataset is used in all subsequent parts of the project.

---

# Part 2 – Supervised Machine Learning

## Objective

The objective of Part 2 was to build and evaluate both regression and classification models using the cleaned dataset. Different preprocessing techniques, feature engineering methods, regularization techniques, class balancing methods, evaluation metrics, and threshold analysis were performed to compare model performance.

---

## Target Variables

Two prediction tasks were performed.

### Regression Target

- SalePrice

### Classification Target

- PriceCategory

The classification target was created by comparing each house price with the median sale price.

```python
PriceCategory = (SalePrice > SalePrice.median()).astype(int)
```

This creates two balanced classes:

- 0 → Lower Price
- 1 → Higher Price

---

## Data Preprocessing

The cleaned dataset generated in Part 1 was used.

The following preprocessing steps were performed:

- Selected input features.
- Separated regression and classification targets.
- Applied One-Hot Encoding to categorical variables.
- Dropped the first dummy column to avoid multicollinearity.
- Split the dataset into training and testing sets.
- Applied StandardScaler.
- Fitted the scaler only on training data.
- Transformed both training and testing datasets.

Fitting the scaler only on the training dataset prevents **data leakage**, ensuring that the test set remains completely unseen during training.

---

# Regression Models

Two regression models were implemented.

## Linear Regression

Evaluation metrics:

- Mean Squared Error (MSE)
- R² Score

The learned coefficients were analyzed to understand feature influence on house prices.

---

## Ridge Regression

Ridge Regression was trained using:

```python
alpha = 1.0
```

Ridge Regression adds **L2 Regularization**, reducing excessively large coefficients and improving model generalization.

The model was compared with Linear Regression using:

- MSE
- R² Score

---

# Classification Model

## Logistic Regression

Binary classification was performed using Logistic Regression.

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curve
- ROC-AUC Score

---

## SMOTE

SMOTE (Synthetic Minority Oversampling Technique) was demonstrated to balance the training dataset.

SMOTE generates synthetic minority-class samples, helping reduce bias toward the majority class and improving classification performance.

---

## Precision and Recall

Precision

```text
Precision = TP / (TP + FP)
```

Recall

```text
Recall = TP / (TP + FN)
```

Precision measures how many predicted positive samples are actually positive.

Recall measures how many actual positive samples are correctly identified.

The F1 Score provides a balanced measure by combining Precision and Recall.

---

## ROC Curve and AUC

The ROC Curve was generated using predicted probabilities from Logistic Regression.

The ROC-AUC score measures the model's ability to distinguish between lower-priced and higher-priced houses.

Higher AUC values indicate better classification performance.

---

## Decision Threshold Analysis

Instead of relying only on the default threshold of **0.50**, multiple thresholds were evaluated.

Thresholds tested:

- 0.30
- 0.40
- 0.50
- 0.60
- 0.70

For each threshold, the following metrics were calculated:

- Precision
- Recall
- F1 Score

Lower thresholds generally improve Recall, while higher thresholds improve Precision.

---

## Logistic Regression Regularization

A second Logistic Regression model was trained using:

```python
C = 0.01
```

Smaller values of **C** apply stronger regularization.

Both Logistic Regression models were compared using:

- Precision
- Recall
- F1 Score
- ROC-AUC Score

---

## Bootstrap Confidence Interval

Bootstrap sampling was performed using **500 iterations**.

The AUC difference between both Logistic Regression models was calculated for each bootstrap sample.

The following statistics were reported:

- Mean AUC Difference
- Lower 95% Confidence Limit
- Upper 95% Confidence Limit

The confidence interval was examined to determine whether the observed performance difference was statistically significant.

---

## Output

The notebook successfully generated:

- Regression Metrics
- Classification Metrics
- Confusion Matrix
- ROC Curve
- Threshold Comparison
- Regularization Comparison
- Bootstrap Confidence Interval


---

# Part 3 – Advanced Machine Learning

## Objective

The objective of Part 3 was to improve classification performance by implementing advanced machine learning algorithms, comparing different models, performing hyperparameter tuning, evaluating feature importance, validating model stability, and saving the best-performing model for future use.

---

# Decision Tree (Baseline Model)

A baseline Decision Tree Classifier was trained using the default parameters.

## Results

- Training Accuracy: **100.00%**
- Testing Accuracy: **87.33%**

### Interpretation

The baseline Decision Tree achieved perfect accuracy on the training dataset but lower accuracy on the testing dataset.

This indicates **overfitting**, where the model memorizes the training data instead of learning general patterns.

Decision Trees are high-variance models because they continue splitting until nearly every training sample is classified correctly.

---

# Controlled Decision Tree

To reduce overfitting, a second Decision Tree model was trained with controlled complexity.

```python
max_depth = 5
min_samples_split = 20
```

## Results

- Training Accuracy: **93.84%**
- Testing Accuracy: **90.41%**

### Interpretation

Limiting the maximum depth reduces model complexity.

Increasing the minimum number of samples required for splitting prevents unnecessary branches and improves generalization.

The smaller gap between training and testing accuracy indicates reduced overfitting.

---

# Gini vs Entropy Comparison

Two Decision Tree models were trained using different splitting criteria.

## Results

| Criterion | Testing Accuracy |
|-----------|-----------------:|
| Gini | 89.38% |
| Entropy | 91.44% |

### Gini Impurity

```text
Gini = 1 − Σ(pi²)
```

### Entropy

```text
Entropy = − Σ(pi log₂(pi))
```

### Interpretation

Both criteria measure node impurity.

For this dataset, the Entropy-based Decision Tree achieved slightly better testing accuracy.

---

# Random Forest Classifier

A Random Forest Classifier was trained using:

```python
n_estimators = 100
max_depth = 10
random_state = 42
```

## Results

- Training Accuracy: **99.74%**
- Testing Accuracy: **92.81%**
- ROC-AUC Score: **0.9796**

### Top 5 Important Features

| Feature | Importance |
|---------|-----------:|
| GrLivArea | 0.067951 |
| YearBuilt | 0.059878 |
| FullBath | 0.055129 |
| OverallQual | 0.054327 |
| GarageCars | 0.051691 |

### Interpretation

Random Forest combines multiple Decision Trees using the **Bagging** technique.

Each tree is trained on a different bootstrap sample while considering random subsets of features during splitting.

The final prediction is obtained by combining predictions from all trees, improving robustness and reducing overfitting.

---

# Gradient Boosting Classifier

Gradient Boosting was trained using:

```python
n_estimators = 100
learning_rate = 0.1
random_state = 42
```

## Results

- Training Accuracy: **99.32%**
- Testing Accuracy: **93.49%**
- ROC-AUC Score: **0.9816**

### Interpretation

Gradient Boosting builds trees sequentially.

Each new tree attempts to correct the errors made by previous trees.

This sequential learning process generally improves prediction accuracy while reducing bias.

---

# Feature Ablation Study

The five least important features identified by Random Forest were removed.

The Random Forest model was retrained using the reduced feature set.

## Removed Features

- Condition2_PosN
- Neighborhood_BrDale
- Condition2_RRNn
- MiscFeature_TenC
- Exterior2nd_Other

## Results

| Model | ROC-AUC |
|--------|---------:|
| Full Model | 0.9796 |
| Reduced Model | 0.9770 |

### Interpretation

Removing the least important features caused only a very small reduction in ROC-AUC.

This suggests that these features contributed very little to prediction performance.

A simpler model with fewer features can reduce computational cost while maintaining similar accuracy.

---

# Cross Validation

A **5-Fold Stratified Cross Validation** was performed.

## Results

| Model | Mean ROC-AUC | Standard Deviation |
|------|-------------:|-------------------:|
| Logistic Regression | 0.9514 | 0.0125 |
| Decision Tree | 0.9296 | 0.0107 |
| Random Forest | 0.9807 | 0.0056 |
| Gradient Boosting | 0.9787 | 0.0073 |

### Interpretation

Cross-validation provides a more reliable estimate of model performance because every sample is used for both training and validation.

Random Forest achieved the highest average ROC-AUC with the lowest variation across folds.

---

# Hyperparameter Tuning using GridSearchCV

A machine learning pipeline was created using:

- SimpleImputer
- StandardScaler
- RandomForestClassifier

The following hyperparameters were tuned:

- n_estimators = [50, 100, 200]
- max_depth = [5, 10, None]
- min_samples_leaf = [1, 5]

## Best Parameters

| Parameter | Value |
|-----------|------:|
| n_estimators | 200 |
| max_depth | 10 |
| min_samples_leaf | 1 |

### Best Mean ROC-AUC

**0.9812**

A total of **18 parameter combinations** were evaluated.

Using **5-fold cross-validation**, the Grid Search performed:

```text
18 × 5 = 90 model fits
```

---

# Manual Learning Curve

The best pipeline was trained using different fractions of the training data.

| Training Fraction | Training ROC-AUC | Testing ROC-AUC |
|------------------:|-----------------:|----------------:|
| 20% | 1.0000 | 0.9796 |
| 40% | 1.0000 | 0.9795 |
| 60% | 1.0000 | 0.9779 |
| 80% | 1.0000 | 0.9813 |
| 100% | 0.9999 | 0.9806 |

### Interpretation

The testing ROC-AUC remained nearly constant around **0.98**.

This indicates that the model has reached stable performance.

Further improvements are more likely to come from feature engineering rather than simply collecting more training data.

---

# Model Serialization

The best pipeline obtained from GridSearchCV was saved using Joblib.

```python
joblib.dump(best_pipeline, "best_model.pkl")
```

The saved model was later loaded using:

```python
loaded_model = joblib.load("best_model.pkl")
```

Predictions generated using the loaded model matched the original model, confirming successful serialization.

---

# Overall Model Comparison

| Model | Test Accuracy | ROC-AUC | 5-Fold Mean ROC-AUC |
|------|--------------:|---------:|--------------------:|
| Logistic Regression | 87.67% | 0.9620 | 0.9514 |
| Decision Tree | 87.33% | — | — |
| Controlled Decision Tree | 90.41% | — | 0.9296 |
| Random Forest | 92.81% | 0.9796 | 0.9807 |
| Gradient Boosting | **93.49%** | **0.9816** | 0.9787 |

---

# Final Recommendation

Among all evaluated models, **Random Forest** and **Gradient Boosting** produced the best performance.

Gradient Boosting achieved the highest testing accuracy and ROC-AUC.

Random Forest achieved the highest average cross-validation performance while also providing feature importance for model interpretation.

Therefore, **Random Forest** was selected as the final model for deployment because it provides an excellent balance between predictive performance, stability, interpretability, and robustness.


---

# Part 4 – LLM Powered Feature

## Selected Track

**Track C – Model Prediction Explanation Pipeline**

This track combines the best-performing machine learning model with a Large Language Model (LLM). The trained model first predicts the class and probability for a given house, after which the LLM generates a structured JSON explanation describing the prediction.

---

## Objective

The objective of this part was to integrate an LLM on top of the trained machine learning model to generate human-readable prediction explanations while ensuring structured outputs, schema validation, deterministic prompting, and basic safety guardrails.

---

# LLM Provider

The project uses:

- **Provider:** OpenRouter
- **Model:** `openai/gpt-oss-20b:free`

The API was accessed using the Python `requests` library.

---

# API Key Handling

The API key is **not hardcoded** inside the notebook.

Instead, it is loaded from an environment variable.

Example:

```python
api_key = os.environ["LLM_API_KEY"]
```

This keeps the API key secure and prevents accidental exposure in the GitHub repository.

---

# Reusable LLM Function

A reusable function named `call_llm()` was implemented.

The function performs the following tasks:

- Creates the JSON request payload.
- Sends an HTTP POST request to the OpenRouter API.
- Includes authorization headers.
- Checks whether the API request was successful.
- Returns only the generated response content.

The function was successfully tested using the prompt:

```
Reply with only the word: hello
```

The returned output confirmed that the API connection was working correctly.

---

# System Prompt

The exact system prompt used in this project is shown below.

```text
You are an AI assistant that explains machine learning predictions.

Return ONLY valid JSON.

The JSON must contain exactly these fields:

prediction_label
confidence_level
top_reason
second_reason
next_step
```

---

# User Prompt Template

The following template was used for every prediction.

```text
Feature Values:
{feature_values}

Predicted Class:
{predicted_class}

Prediction Probability:
{predicted_probability}

Explain this prediction.

Return ONLY valid JSON using this format:

{
"prediction_label":"",
"confidence_level":"",
"top_reason":"",
"second_reason":"",
"next_step":""
}
```

---

# Why Temperature = 0?

Temperature was set to **0** because this task requires structured and deterministic output.

At temperature **0**, the model consistently selects the highest-probability next token, producing stable JSON responses that are easier to validate.

Higher temperatures introduce randomness, which can change wording, formatting, or even the structure of the generated JSON.

---

# Temperature Comparison

The LLM was tested using two temperature values.

| Input | Output at Temperature = 0 | Output at Temperature = 0.7 | Key Difference |
|-------|---------------------------|-----------------------------|---------------|
| Record 1 | Deterministic JSON | More varied wording | Higher temperature changes explanations |
| Record 2 | Deterministic JSON | Different wording | Slight variation in reasoning |
| Record 3 | Deterministic JSON | More creative explanation | Lower consistency |

### Interpretation

Temperature **0** always produces the most probable output, making it suitable for structured tasks such as JSON generation.

Temperature **0.7** samples from a wider probability distribution, making responses more diverse but less predictable.

---

# Structured Output Validation

Every LLM response follows a predefined JSON schema.

The schema contains five required fields:

- prediction_label
- confidence_level
- top_reason
- second_reason
- next_step

The validation pipeline performs the following steps:

1. Strip whitespace.
2. Parse the response using `json.loads()`.
3. Validate the parsed JSON using `jsonschema.validate()`.
4. Catch JSON decoding errors.
5. Catch schema validation errors.
6. Return a fallback dictionary containing `None` values if validation fails.

This ensures that only valid structured responses are accepted.

---

# PII Guardrail

Before every LLM call, a regular expression checks whether the user input contains Personally Identifiable Information (PII).

The following information is blocked:

- Email addresses
- Phone numbers

If PII is detected, the API call is skipped and the following message is displayed:

```
Input blocked: PII detected.
```

---

# Guardrail Demonstration

| Test Input | Result |
|------------|--------|
| user@example.com | Blocked |
| House feature record | Allowed |

The guardrail successfully blocked inputs containing personal information while allowing normal model explanations.

---

# End-to-End Demonstration

The complete pipeline was executed for three different house records.

For each record, the following steps were performed:

1. Load the saved model.
2. Encode the feature vector.
3. Predict the class.
4. Predict the probability.
5. Generate an LLM explanation.
6. Parse the JSON response.
7. Validate the JSON schema.
8. Report the validation result.

---

# Demonstration Results

| Input | Predicted Class | Probability | Valid JSON | Guardrail |
|------|----------------:|------------:|-----------|-----------|
| Record 1 | 1 | 0.9917 | PASS | Allowed |
| Record 2 | 1 | 0.7774 | PASS | Allowed |
| Record 3 | 1 | 0.9808 | PASS | Allowed |

---

# Conclusion

The LLM-powered explanation pipeline was successfully integrated with the best-performing machine learning model.

The implementation satisfies all assignment requirements:

- API key stored using an environment variable.
- Reusable `call_llm()` function implemented.
- OpenRouter API successfully connected.
- Track C completed.
- Best model loaded using `joblib`.
- Predictions generated using `.predict()` and `.predict_proba()`.
- Structured JSON output generated.
- JSON validated using `jsonschema`.
- Invalid responses handled using fallback values.
- PII guardrail implemented.
- Temperature comparison completed.
- Three end-to-end demonstrations successfully executed.

The final system combines machine learning prediction with LLM-generated explanations, making model outputs easier to understand while maintaining structured and validated responses.

---

# How to Run the Project

## Step 1

Clone the repository.

```bash
git clone <your-github-repository-link>
```

---

## Step 2

Move into the project directory.

```bash
cd Applied_AI_ML_Capstone
```

---

## Step 3

Install the required libraries.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib requests jsonschema
```

---

## Step 4

Set your OpenRouter API Key.

**Windows PowerShell**

```powershell
$env:LLM_API_KEY="YOUR_OPENROUTER_API_KEY"
```

**Linux / macOS**

```bash
export LLM_API_KEY="YOUR_OPENROUTER_API_KEY"
```

---

## Step 5

Run the notebooks in the following order:

1. Part 1 – Data Cleaning & EDA
2. Part 2 – Supervised Machine Learning
3. Part 3 – Advanced Machine Learning
4. Part 4 – LLM Powered Feature

---

# Required Libraries

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib
- requests
- jsonschema

---

# Repository Contents

```
Applied_AI_ML_Capstone/
│
├── data/
│   ├── train.csv
│   └── cleaned_data.csv
│
├── notebooks/
│   ├── Part1_Data_Cleaning_EDA.ipynb
│   ├── Part2_Supervised_ML.ipynb
│   ├── Part3_Advanced_ML.ipynb
│   └── Part4_LLM_Powered_Feature.ipynb
│
├── models/
│   └── best_model.pkl
│
├── images/
│
└── README.md
```

---

# Assignment Summary

This project successfully covers all four parts of the Applied AI / Machine Learning Capstone assignment.

✔ Part 1 – Data Cleaning and Exploratory Data Analysis

✔ Part 2 – Supervised Machine Learning

✔ Part 3 – Advanced Machine Learning

✔ Part 4 – LLM Powered Feature (Track C)

The project includes:

- Data preprocessing
- Feature engineering
- Model comparison
- Hyperparameter tuning
- Cross-validation
- Model serialization
- OpenRouter API integration
- Prompt engineering
- JSON schema validation
- PII guardrails
- Temperature comparison
- End-to-end LLM prediction explanation pipeline

---

# Author

**Rashid Imam**

B.Tech – Information Technology

Guru Ghasidas Vishwavidyalaya, Bilaspur

---

# License

This repository is submitted for academic purposes as part of the Applied AI / Machine Learning Capstone assignment.