# Financial Price Prediction Using an LSTM Recurrent Neural Network

## Overview

This project implements a **Long Short-Term Memory (LSTM) recurrent neural network** to predict the future price of the **ADA/USD (ADAUSD) cryptocurrency trading pair**.

The model uses two input features:

* **Price** — the historical ADAUSD trading price.
* **Volume** — the amount of ADA traded during the corresponding observation.

The objective is to use historical sequences of price and volume to predict the **next ADAUSD price**.

The implementation follows three main stages:

1. **Data preprocessing**
2. **LSTM model construction and training**
3. **Price prediction and model evaluation**

---

# 1. Data Preprocessing

## 1.1 Import the required libraries

The first step is to import the libraries required for data manipulation, preprocessing, model construction and evaluation.

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from datasets import load_dataset

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
```

### Purpose

The main libraries have the following roles:

| Library               | Purpose                            |
| --------------------- | ---------------------------------- |
| NumPy                 | Numerical and array operations     |
| Pandas                | Data manipulation                  |
| Matplotlib            | Visualisation                      |
| Hugging Face Datasets | Loading the cryptocurrency dataset |
| MinMaxScaler          | Scaling price and volume           |
| TensorFlow/Keras      | Building and training the LSTM     |
| Scikit-learn metrics  | Evaluating prediction performance  |

---

# 2. Import the Dataset

The cryptocurrency dataset is loaded and converted into a Pandas DataFrame.

```python
dataset = load_dataset("GotThatData/kraken-trading-data")

data = dataset["train"].to_pandas()
```

The dataset contains information for multiple cryptocurrency trading pairs.

Because the objective is specifically to predict ADAUSD, the dataset is filtered to retain only this trading pair.

```python
ada_data = data[data["pair"] == "ADAUSD"].copy()
```

The observations are then sorted chronologically.

```python
ada_data = ada_data.sort_values("timestamp")
```

This step is particularly important for time-series prediction because the model must learn from the **past to predict the future**.

Finally, price and volume are selected as the model's input features:

```python
features = ada_data[["price", "volume"]].values
```

The resulting structure is conceptually:

```text
Time       Price       Volume
--------------------------------
t1         P1          V1
t2         P2          V2
t3         P3          V3
...
```

---

# 3. Split the Data into Training and Testing Sets

The data is divided chronologically into training and testing sets.

```python
training_size = int(len(features) * 0.8)

train_set = features[:training_size]
test_set = features[training_size:]
```

Here, **80% of the observations are used for training** and the remaining **20% are used for testing**.

Unlike a conventional machine-learning problem, the data should not be randomly shuffled.

The chronological structure must be preserved:

```text
Past                                      Future
│                                           │
▼                                           ▼
Training data                    Testing data
      80%                              20%
```

The model therefore learns from historical observations and is evaluated on observations that occur later in time.

---

# 4. Feature Scaling

Price and volume can have very different numerical ranges.

For example:

```text
Price   → relatively small values
Volume  → potentially much larger values
```

To prevent the larger numerical values from dominating the neural network, both variables are scaled between 0 and 1.

```python
sc = MinMaxScaler(feature_range=(0, 1))

train_set_scaled = sc.fit_transform(train_set)
test_set_scaled = sc.transform(test_set)
```

An important detail is that the scaler is **fitted only on the training data**:

```python
sc.fit_transform(train_set)
```

The same transformation is then applied to the test data:

```python
sc.transform(test_set)
```

This prevents information from the test period from influencing the training process.

---

# 5. Create the Time-Series Sequences

An LSTM does not simply receive individual rows of data.

Instead, it receives a **sequence of previous observations**.

In this implementation, the model uses the previous **60 observations** to predict the next price.

```python
timesteps = 60
```

For example:

```text
Observations 1–60  → Predict observation 61
Observations 2–61  → Predict observation 62
Observations 3–62  → Predict observation 63
```

The training sequences are created with:

```python
X_train = []
y_train = []

for i in range(timesteps, len(train_set_scaled)):

    X_train.append(
        train_set_scaled[i-timesteps:i, :]
    )

    y_train.append(
        train_set_scaled[i, 0]
    )
```

The important distinction is:

### Input — X

The input contains:

* the previous 60 prices
* the previous 60 volumes

Therefore:

```text
X = 60 previous observations
    ├── Price
    └── Volume
```

### Target — y

The target is:

```text
The price at the next observation
```

Only the price is predicted, which is why:

```python
train_set_scaled[i, 0]
```

is used.

The `0` refers to the first feature, which is `price`.

---

# 6. Convert the Sequences into NumPy Arrays

The lists are converted into NumPy arrays:

```python
X_train, y_train = np.array(X_train), np.array(y_train)
```

The resulting input has three dimensions:

```text
Number of sequences × Number of timesteps × Number of features
```

For example:

```text
X_train.shape

(number of observations, 60, 2)
```

The `2` represents:

```text
1. Price
2. Volume
```

This three-dimensional structure is required by the LSTM.

---

# 7. Create the Test Sequences

The test data requires special treatment.

The model still needs the previous 60 observations when making its first test prediction.

Therefore, the last 60 observations from the training data are combined with the test data:

```python
test_input = np.concatenate(
    (train_set_scaled[-timesteps:], test_set_scaled),
    axis=0
)
```

This allows the first test prediction to use historical information immediately preceding the test period.

The test sequences are then created:

```python
X_test = []
y_test = []

for i in range(timesteps, len(test_input)):

    X_test.append(
        test_input[i-timesteps:i, :]
    )

    y_test.append(
        test_input[i, 0]
    )
```

Finally:

```python
X_test, y_test = np.array(X_test), np.array(y_test)
```

The model can now use:

```text
X_test → Previous 60 price/volume observations
y_test → Actual next price
```

---

# 8. Verify the Data Shapes

The dimensions of the datasets are displayed:

```python
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
```

The expected structure is approximately:

```text
X_train → (samples, 60, 2)
y_train → (samples,)

X_test  → (samples, 60, 2)
y_test  → (samples,)
```

The important point is that the LSTM receives **60 time steps and 2 features at each time step**.

---

# 9. Build the LSTM Network

The model is created using Keras' `Sequential` architecture.

```python
regressor = Sequential()
```

A sequential model means that the layers are processed one after another.

The architecture is:

```text
Price + Volume
      │
      ▼
   LSTM 50
      │
    Dropout
      │
      ▼
   LSTM 50
      │
    Dropout
      │
      ▼
   LSTM 50
      │
    Dropout
      │
      ▼
   LSTM 50
      │
    Dropout
      │
      ▼
   Dense
      │
      ▼
 Predicted Price
```

---

# 10. First LSTM Layer

```python
regressor.add(
    LSTM(
        units=50,
        return_sequences=True,
        input_shape=(X_train.shape[1], X_train.shape[2])
    )
)
```

The first LSTM contains **50 units**.

The input shape is:

```text
60 timesteps × 2 features
```

Therefore, each training example looks conceptually like:

```text
Time step 1   → Price + Volume
Time step 2   → Price + Volume
...
Time step 60  → Price + Volume
```

The LSTM processes this sequence and attempts to identify temporal relationships between previous observations.

---

# 11. Dropout Regularisation

After the first LSTM layer, dropout is applied:

```python
regressor.add(Dropout(0.2))
```

A dropout rate of `0.2` means that approximately 20% of the neurons are randomly ignored during training.

The purpose is to reduce **overfitting**.

Without regularisation, the network could potentially learn the training data too closely and perform poorly on unseen market data.

---

# 12. Additional LSTM Layers

Three additional LSTM layers are added.

```python
regressor.add(
    LSTM(
        units=50,
        return_sequences=True
    )
)

regressor.add(Dropout(0.2))
```

This pattern is repeated for the third LSTM layer.

```python
regressor.add(
    LSTM(
        units=50,
        return_sequences=True
    )
)

regressor.add(Dropout(0.2))
```

The final LSTM layer is slightly different:

```python
regressor.add(
    LSTM(units=50)
)

regressor.add(Dropout(0.2))
```

The final LSTM does not use:

```python
return_sequences=True
```

because it is the final recurrent layer and only needs to produce a representation that can be passed to the output layer.

---

# 13. Output Layer

The final layer is a dense neural-network layer:

```python
regressor.add(Dense(units=1))
```

There is only one output neuron because the objective is to predict one value:

```text
Future ADAUSD price
```

The overall prediction process is therefore:

```text
60 historical observations
        │
        ▼
   LSTM layers
        │
        ▼
Temporal representation
        │
        ▼
   Dense layer
        │
        ▼
Predicted price
```

---

# 14. Compile the Model

The network is configured using:

```python
regressor.compile(
    optimizer="adam",
    loss="mean_squared_error"
)
```

### Adam optimizer

Adam determines how the neural network's weights should be updated during training.

### Mean Squared Error

The loss function measures the difference between the predicted and actual prices.

Conceptually:

```text
Predicted price
      │
      ▼
Compare with
      │
      ▼
Actual price
      │
      ▼
Calculate error
      │
      ▼
Update network weights
```

The objective is to minimise this error.

---

# 15. Early Stopping

An early-stopping mechanism is introduced:

```python
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)
```

This prevents the model from continuing to train once its validation performance stops improving.

The model waits for up to 10 epochs for improvement.

If no improvement occurs, training stops and the best-performing model weights are restored.

This helps reduce overfitting and unnecessary training.

---

# 16. Train the LSTM

The model is trained using:

```python
regressor.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    shuffle=False
)
```

The important parameters are:

| Parameter              | Meaning                                                    |
| ---------------------- | ---------------------------------------------------------- |
| `X_train`              | Historical price/volume sequences                          |
| `y_train`              | Target prices                                              |
| `epochs=100`           | Maximum number of training iterations                      |
| `batch_size=32`        | Number of sequences processed together                     |
| `validation_split=0.1` | 10% of training data used for validation                   |
| `early_stop`           | Stops training when validation performance stops improving |
| `shuffle=False`        | Maintains chronological order                              |

`shuffle=False` is particularly important for financial time-series data because the temporal ordering of observations should be preserved.

---

# 17. Generate Predictions

After training, the model predicts the prices in the test period:

```python
predicted_scaled = regressor.predict(X_test)
```

The predictions are still in the scaled 0–1 range.

Therefore, they need to be converted back to their original price scale.

---

# 18. Convert Predictions Back to Real Prices

Because the scaler was fitted using both:

```text
Price
Volume
```

it expects two columns when performing the inverse transformation.

A dummy volume column is therefore created:

```python
dummy_volume = np.zeros(
    (len(predicted_scaled), 1)
)
```

The predicted price and dummy volume are combined:

```python
predicted_with_volume = np.concatenate(
    (predicted_scaled, dummy_volume),
    axis=1
)
```

The original scale is then restored:

```python
predicted_original = sc.inverse_transform(
    predicted_with_volume
)
```

Only the first column is required because it represents price:

```python
predicted_price = predicted_original[:, 0].reshape(-1, 1)
```

The result is now expressed in the original ADAUSD price units.

---

# 19. Convert Actual Prices Back to the Original Scale

The same process is applied to the actual test prices.

```python
dummy_volume_actual = np.zeros(
    (len(y_test), 1)
)
```

The scaled price and dummy volume are combined:

```python
actual_scaled_with_volume = np.column_stack(
    (y_test, dummy_volume_actual[:, 0])
)
```

The values are converted back to their original scale:

```python
actual_original = sc.inverse_transform(
    actual_scaled_with_volume
)
```

Only the price column is retained:

```python
actual_price = actual_original[:, 0].reshape(-1, 1)
```

---

# 20. Visualise the Predictions

The actual and predicted prices are plotted together:

```python
plt.figure(figsize=(14, 7))

plt.plot(
    actual_price,
    label="Real ADAUSD Price"
)

plt.plot(
    predicted_price,
    label="Predicted ADAUSD Price"
)

plt.title("ADAUSD Price Prediction")
plt.xlabel("Time")
plt.ylabel("ADAUSD Price")
plt.legend()

plt.show()
```

This provides a visual comparison between:

```text
Actual ADAUSD price
        vs.
Predicted ADAUSD price
```

A model that performs well should generally follow the overall movement of the actual price series.

However, visually following the price is not sufficient to establish that the model is useful for trading.

---

# 21. Evaluate the Model

Three regression metrics are calculated.

## Mean Absolute Error — MAE

```python
mae = mean_absolute_error(
    actual_price,
    predicted_price
)
```

MAE measures the average absolute difference between predicted and actual prices.

Lower values are better.

---

## Root Mean Squared Error — RMSE

```python
rmse = np.sqrt(
    mean_squared_error(
        actual_price,
        predicted_price
    )
)
```

RMSE gives greater importance to larger prediction errors.

Lower values are better.

---

## R² Score

```python
r2 = r2_score(
    actual_price,
    predicted_price
)
```

R² measures how much of the variation in the target variable is explained by the model.

Generally:

```text
Closer to 1 → better fit
Closer to 0 → weak explanatory power
Negative → model performs poorly relative to a simple baseline
```

---

# 22. Overall Process

The complete methodology can be summarised as:

```text
Kraken Trading Data
        │
        ▼
Select ADAUSD
        │
        ▼
Sort chronologically
        │
        ▼
Select Price + Volume
        │
        ▼
Train/Test Split
        │
        ▼
Feature Scaling
        │
        ▼
Create 60-step sequences
        │
        ▼
┌─────────────────────┐
│      LSTM 50        │
├─────────────────────┤
│      Dropout        │
├─────────────────────┤
│      LSTM 50        │
├─────────────────────┤
│      Dropout        │
├─────────────────────┤
│      LSTM 50        │
├─────────────────────┤
│      Dropout        │
├─────────────────────┤
│      LSTM 50        │
├─────────────────────┤
│      Dropout        │
├─────────────────────┤
│    Dense Output     │
└─────────────────────┘
        │
        ▼
Predicted ADAUSD Price
        │
        ▼
Inverse Scaling
        │
        ▼
Actual vs Predicted
        │
        ▼
MAE / RMSE / R²
```

# Key Concepts

The most important concepts demonstrated by this implementation are:

### 1. Time-series sequencing

The LSTM does not look at one observation at a time. It receives a sequence of the previous **60 observations**.

### 2. Multiple input features

The model uses both **price and trading volume**, allowing it to potentially identify relationships between market activity and subsequent price movements.

### 3. Temporal learning

LSTM networks are specifically designed to capture patterns and dependencies across sequences of observations.

### 4. Feature scaling

Price and volume are normalised so that differences in their numerical magnitudes do not adversely affect model training.

### 5. Dropout

Dropout helps reduce overfitting by preventing the network from becoming overly dependent on particular neurons.

### 6. Early stopping

Early stopping prevents unnecessary training and restores the model's best validation performance.

### 7. Regression

This is a **regression problem**, because the model predicts a continuous numerical value—the future ADAUSD price.

---

# Important Consideration for Financial Prediction

This model should be interpreted as a **price forecasting experiment**, rather than as a guaranteed trading strategy.

A high R² or low RMSE does not necessarily mean that the model can generate profitable trades. Financial markets are affected by many factors that are not included in this model, such as market sentiment, Bitcoin movements, liquidity, macroeconomic events and order-book dynamics.

For a more realistic trading model, the next development would be to predict **future returns or price direction** rather than the absolute price, and evaluate the model using **walk-forward validation and trading-specific metrics** such as directional accuracy, cumulative return, maximum drawdown and Sharpe ratio.
