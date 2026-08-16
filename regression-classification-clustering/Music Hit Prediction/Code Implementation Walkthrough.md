# Regression with XGBoost — Music Hit Prediction

## 1. Project Objective

The objective of this project is to use **XGBoost regression** to predict the popularity of songs based on their available musical characteristics.

The model uses nine musical features:

* Danceability
* Energy
* Instrumentalness
* Liveness
* Loudness
* Speechiness
* Tempo
* Valence
* Acousticness

The target variable is **popularity**, represented by a numerical value.

The project has two stages:

1. **Baseline XGBoost regression** — train a standard XGBoost model and evaluate its predictive performance.
2. **Optimised XGBoost regression** — use **K-Fold Cross-Validation and Randomized Search** to identify a better combination of XGBoost hyperparameters.

---

# Part 1 — Data Preparation

## 2. Import the Required Libraries

The first step imports the libraries required for:

* Data manipulation
* Dataset loading
* Train/test splitting
* Model evaluation
* XGBoost regression
* Hyperparameter optimisation

```python
import numpy as np
import pandas as pd

from datasets import load_dataset

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor

from scipy.optimize import differential_evolution
```

The most important library for the model itself is **XGBRegressor**, which implements XGBoost for regression problems.

---

## 3. Import the Dataset

The Spotify dataset is loaded from Hugging Face:

```python
dataset = load_dataset("maharshipandya/spotify-tracks-dataset")
data = dataset["train"].to_pandas()
```

The dataset is then converted into a Pandas DataFrame so that the columns can be selected and manipulated easily.

Conceptually:

```text
Spotify dataset
       ↓
Training data
       ↓
Pandas DataFrame
```

---

# Part 2 — Selecting Features and Target

## 4. Select the Musical Features

The model uses nine musical characteristics as input variables:

```python
feature_columns = [
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo",
    "valence",
    "acousticness"
]
```

These variables become the **features (`X`)**.

The target variable is:

```python
target_column = "popularity"
```

Therefore:

```python
X = data[feature_columns].copy()
y = data[target_column].copy()
```

The structure is:

```text
Musical Features
      ↓
      X
      ↓
XGBoost Regression
      ↓
Popularity
      ↑
      y
```

The model is therefore trying to learn the relationship:

> **Musical characteristics → predicted song popularity**

---

# Part 3 — Training/Test Split

## 5. Split the Dataset

The dataset is divided into two parts:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0
)
```

Here, **80% of the data is used for training** and **20% is reserved for testing**.

The purpose is to evaluate the model using songs that were not used during training.

```text
Dataset
   │
   ├── 80% → Training data
   │
   └── 20% → Test data
```

This helps determine whether the model can generalise beyond the songs it has already seen.

---

# Part 4 — Baseline XGBoost Model

## 6. Create the XGBoost Regressor

A standard XGBoost regression model is created:

```python
regressor = XGBRegressor()
```

Unlike classification, where the model predicts categories, **XGBRegressor predicts a numerical value**.

In this case, the numerical value is song popularity.

---

## 7. Train the XGBoost Model

The model is trained using the training data:

```python
regressor.fit(X_train, y_train)
```

During training, XGBoost builds a sequence of decision trees.

Each new tree attempts to improve the errors made by the previous trees.

Conceptually:

```text
Training data
      ↓
Decision Tree 1
      ↓
Identify errors
      ↓
Decision Tree 2
      ↓
Correct previous errors
      ↓
Decision Tree 3
      ↓
Continue improving
      ↓
Final XGBoost model
```

The model therefore learns increasingly useful relationships between the musical features and popularity.

---

# Part 5 — Predicting Popularity

## 8. Predict the Test Set

After training, the model predicts popularity for the previously unseen test songs:

```python
y_pred = regressor.predict(X_test)
```

The model produces a numerical popularity prediction for each song.

For example:

```text
Actual popularity    Predicted popularity

        72                   68
        45                   51
        81                   76
        32                   39
```

The difference between the actual and predicted values is then used to evaluate the model.

---

# Part 6 — Evaluating the Model

## 9. Calculate MAE

Mean Absolute Error measures the average absolute difference between the predicted and actual popularity values.

```python
mae = mean_absolute_error(y_test, y_pred)
```

For example, an MAE of 8 would approximately mean:

> The model's predictions are, on average, about 8 popularity points away from the actual values.

**Lower MAE is better.**

---

## 10. Calculate RMSE

Root Mean Squared Error is another measure of prediction error:

```python
rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)
```

RMSE gives greater importance to larger prediction errors.

Therefore, it is useful for identifying whether the model occasionally makes very large mistakes.

**Lower RMSE is better.**

---

## 11. Calculate R²

R² measures how much of the variation in popularity is explained by the model:

```python
r2 = r2_score(y_test, y_pred)
```

Conceptually:

```text
R² closer to 1 → stronger predictive relationship
R² around 0    → limited explanatory power
R² below 0     → model performs poorly
```

For this project, R² is particularly useful because it helps determine whether the selected musical characteristics actually contain useful information for predicting popularity.

---

# Part 7 — Bonus: Optimising the XGBoost Model

The second part attempts to improve the baseline model by finding better **XGBoost hyperparameters**.

Instead of simply accepting the default XGBoost configuration, the model tests different configurations.

---

# 12. Define K-Fold Cross-Validation

```python
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=0
)
```

The training dataset is divided into **five folds**.

The model is repeatedly trained and validated using different combinations of these folds.

Conceptually:

```text
Round 1: [Test][Train][Train][Train][Train]

Round 2: [Train][Test][Train][Train][Train]

Round 3: [Train][Train][Test][Train][Train]

Round 4: [Train][Train][Train][Test][Train]

Round 5: [Train][Train][Train][Train][Test]
```

The five validation results are then combined to provide a more reliable estimate of model performance.

This reduces the possibility that the result depends too heavily on one particular train/validation split.

---

# 13. Define the XGBoost Model for Optimisation

```python
xgb_model = XGBRegressor(
    objective="reg:squarederror",
    random_state=0,
    n_jobs=-1
)
```

This creates the XGBoost model that will be tested during the optimisation process.

The objective:

```text
reg:squarederror
```

means that the model is solving a regression problem using squared error as its learning objective.

---

# 14. Define the Hyperparameters to Optimise

The code creates a range of possible XGBoost configurations.

For example:

```python
"n_estimators": [200, 300, 400, 500, 600]
```

controls the number of trees.

```python
"learning_rate": [0.01, 0.03, 0.05, 0.1]
```

controls how strongly each new tree contributes to the model.

```python
"max_depth": [3, 4, 5, 6]
```

controls the maximum depth of each tree.

Other parameters control:

* Minimum number of observations required for additional tree complexity
* The proportion of observations used by each tree
* The proportion of features considered by each tree
* How conservative the model is when creating new tree branches

The purpose is to find a combination that provides better generalisation.

---

# Part 8 — Randomized Search

## 15. Search Through Different Configurations

Instead of testing every possible combination, the code uses:

```python
RandomizedSearchCV()
```

with:

```python
n_iter=30
```

This means that **30 different combinations** of hyperparameters are randomly selected from the specified ranges.

Each configuration is evaluated using the five-fold cross-validation procedure.

Conceptually:

```text
Hyperparameter ranges
        ↓
Randomly select configuration
        ↓
5-Fold Cross-Validation
        ↓
Calculate R²
        ↓
Select another configuration
        ↓
Repeat 30 times
        ↓
Find best configuration
```

This is more computationally efficient than testing every possible combination.

---

# Part 9 — Identify the Best Model

## 16. Train the Search

```python
search.fit(
    X_train,
    y_train
)
```

The search evaluates the different XGBoost configurations using only the training data.

This is important because the test set should remain unseen until the final evaluation.

---

## 17. Display the Best Parameters

```python
for parameter, value in search.best_params_.items():
    print(f"{parameter}: {value}")
```

This identifies the hyperparameter configuration that achieved the best cross-validated R² score.

For example, the result might identify:

```text
n_estimators: 400
learning_rate: 0.03
max_depth: 4
...
```

The exact values depend on the dataset and the search results.

The best cross-validated R² is also displayed:

```python
search.best_score_
```

This provides an estimate of how well the selected configuration performed across the validation folds.

---

# Part 10 — Select the Optimised Model

## 18. Retrieve the Best XGBoost Model

```python
optimized_regressor = search.best_estimator_
```

The model with the best cross-validation performance is now selected.

The workflow has therefore changed from:

```text
Default XGBoost
       ↓
Prediction
```

to:

```text
Multiple XGBoost configurations
       ↓
5-Fold Cross-Validation
       ↓
Randomized Search
       ↓
Best hyperparameters
       ↓
Optimised XGBoost model
```

---

# Part 11 — Evaluate the Optimised Model

## 19. Predict the Test Set

The optimised model is applied to the original test set:

```python
y_pred_optimized = optimized_regressor.predict(X_test)
```

Importantly, **the test data has not been used to select the hyperparameters**.

This provides a more realistic evaluation of the optimised model.

---

## 20. Calculate the Optimised MAE, RMSE and R²

The same three metrics are calculated:

```text
MAE
RMSE
R²
```

This allows the baseline and optimised models to be compared.

For example:

| Model             |    MAE |   RMSE |     R² |
| ----------------- | -----: | -----: | -----: |
| Baseline XGBoost  | Higher | Higher |  Lower |
| Optimised XGBoost |  Lower |  Lower | Higher |

Ideally, optimisation should reduce MAE and RMSE while increasing R².

However, **optimisation does not guarantee a substantial improvement**. If the available features contain limited information about popularity, improving the XGBoost configuration cannot fully solve that limitation.

---

# Overall Process

The complete methodology can be summarised as:

```text
Spotify Dataset
      ↓
Select Musical Features
      ↓
Select Popularity as Target
      ↓
Train/Test Split
      ↓
Baseline XGBoost Regressor
      ↓
Predict Popularity
      ↓
Evaluate MAE, RMSE and R²
      ↓
────────────────────────
     BONUS: OPTIMISATION
────────────────────────
      ↓
Define Hyperparameter Ranges
      ↓
Randomized Search
      ↓
5-Fold Cross-Validation
      ↓
Evaluate Multiple XGBoost Configurations
      ↓
Select Best Configuration
      ↓
Optimised XGBoost Model
      ↓
Predict Test Set
      ↓
Evaluate MAE, RMSE and R²
```

# Important Limitation — Predicting Music Popularity

A key consideration in this project is that **the available musical features may not be the only, or necessarily the most important, factors determining song popularity**.

The model currently considers characteristics such as:

```text
Danceability
Energy
Instrumentalness
Liveness
Loudness
Speechiness
Tempo
Valence
Acousticness
```

However, popularity can potentially depend on many other factors that are not represented by these variables, such as **artist recognition, marketing, playlist exposure, release timing, genre, social media activity, cultural trends, collaborations, and listener behaviour**.

Therefore, the model should not be interpreted as proving that a particular combination of musical characteristics *causes* higher popularity.

More importantly, **there may not be a simple linear relationship between the selected musical features and popularity**. XGBoost is useful precisely because it can model complex **non-linear relationships and interactions** between features. Nevertheless, even a highly optimised XGBoost model cannot reliably predict information that is absent from the dataset.

The interpretation is therefore:

> **The model estimates popularity based on the relationships that can be learned from the available musical features and historical observations. It does not establish that these features are the fundamental determinants of song popularity.**

This makes the optimisation exercise useful not only for finding a better-performing model, but also for demonstrating an important machine-learning principle:

**Better algorithms and hyperparameter tuning cannot compensate indefinitely for incomplete or weak predictive features.**
