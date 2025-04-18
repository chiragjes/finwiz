---
title: Notebook 1 - Introduction to Financial Time Series Prediction with LSTMs
---

# Introduction

This workbook will help you understand how to build a simple machine learning model to predict stock prices. The learning objectives are:

1. **Data Preprocessing**: Understanding why data preprocessing is important and how Python can simplify the process.  
2. **LSTM Models**: Learning about the architecture and computations in an LSTM model, and why it is suitable for stock market price prediction.  
3. **Model Performance**: Understanding how to assess a model's performance and differentiate between underfitting and overfitting.




```python

## Import Libraries and set information
from pandas_datareader import data as pdr
from datetime import date

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import keras
from keras import Model
from keras.layers import Input, Dense, LSTM, GRU, Dropout, concatenate
from keras.optimizers import Adam
from keras.utils import plot_model
import yfinance as yf


# Get Current Date
today = date(2024, 4, 5)
currentDate = today.strftime("%Y-%m-%d")

# Set Info
start_date = '2015-01-01'
end_date = currentDate
stockName = ['AMZN']
import seaborn as sns
```

## Setting Up

The code below imports all the necessary libraries from relevant packages. We will be using Yahoo Finance’s API to collect stock price data.

We are analyzing 13 years of data from four different companies to make predictions. You are encouraged to experiment by trying other combinations of companies or by adding the time series for a stock market index.

The input data selected here determines what our model will be trained on. By choosing the type of input data, you’re essentially deciding what kind of knowledge your model will capture. Other models have been built using:

1. **Macroeconomic Data**: Interest rates, unemployment rates, and GDP changes all influence the stock market. These time series can either be part of your LSTM model or included as separate inputs.
2. **Fundamental Indicators**: Metrics such as profit, EBITDA, and book value specific to a company. A model can learn how changes in these values relate to changes in stock prices. This approach closely mirrors real-world investment strategies and is a valuable extension of this project.
3. **News and Sentiment Data**: Investors and researchers have used commentary from forums like Twitter (now X), Reddit, or news articles to perform sentiment analysis, examining its impact on stock prices. Predicting from text data with LSTMs is challenging, as LSTMs are primarily designed for time-series data. However, with the right feature engineering, such models can potentially achieve higher accuracy.



```python
# Downloading the data from yfinance
amzn_ticker = yf.Ticker(stockName[0])
stock_amzn = amzn_ticker.history(start=start_date, end=end_date)
stock_amzn.index = stock_amzn.index.strftime('%Y-%m-%d')
```


```python
stock_amzn.head()
```


```python
import plotly.express as px

fig = px.line(stock_amzn, y = 'Close')
fig.update_xaxes(rangeslider_visible=True)
fig.update_layout(title_text='Amazon Close Price')
fig.show()
```


```python
df_amzn = stock_amzn
df_amzn = df_amzn.filter(['Close'])
dataset_amzn = df_amzn.values
```


```python
df_amzn.tail()
```

| Date       |   Open |   High |    Low |   Close |      Volume |   Dividends |   Stock Splits |
|:-----------|-------:|-------:|-------:|--------:|------------:|------------:|---------------:|
| 2024-03-28 | 180.17 | 181.7  | 179.26 |  180.38 | 3.80516e+07 |           0 |              0 |
| 2024-04-01 | 180.79 | 183    | 179.95 |  180.97 | 2.91745e+07 |           0 |              0 |
| 2024-04-02 | 179.07 | 180.79 | 178.38 |  180.69 | 3.26115e+07 |           0 |              0 |
| 2024-04-03 | 179.9  | 182.87 | 179.8  |  182.41 | 3.10466e+07 |           0 |              0 |
| 2024-04-04 | 184    | 185.1  | 180    |  180    | 4.16243e+07 |           0 |              0 |

## How does a typical ML model work?


Before diving into LSTMs, it is important to understand how a typical machine learning model pipeline functions. There are three main components:

1. **Data Preprocessing and Feature Engineering**  
2. **Initializing and Specifying Model Parameters**  
3. **Training, Testing, and Cross-Validating the Model**

Finally, predictions are generated and compared to actual data.

These steps will be broken down further throughout this workbook. Focus on the importance of each step and the intuitive modeling decisions you make by choosing one method over another.


# Example 1: Are the Numbers Too Big?

Before fitting our model, we need to preprocess our data. LSTMs take windows of a sequence as input. This means they consider a subset of sequential data points at any given time, allowing them to capture temporal relationships within the data.

Imagine an Excel sheet tracking stock prices over time, where each row represents one day's data. An LSTM might look at a 5-day window to predict the price on the 6th day.

In our case, the LSTM considers a 15-day window of stock prices to predict the price on the 16th day. Then, the window slides forward by one day, repeating the process. In this way, the model learns from past patterns to make future predictions, assuming that historical patterns will repeat.

Its important to note that with univariate data (as we have here), we are only using past prices of a stock to predict its future value.



```python
# Storing the number of data points in the array
num_data_amzn = len(dataset_amzn)
num_days_used = 15
data_used_amzn = np.array([dataset_amzn[i : i + num_days_used].copy() for i in range(num_data_amzn - num_days_used)])
data_to_predict_amzn = np.array(dataset_amzn[num_days_used:, :1])

# Creating a dates array for the dates that were used by data_used
dates_used_amzn = stock_amzn.index[num_days_used:num_data_amzn]

display(data_used_amzn.shape,data_to_predict_amzn.shape, dates_used_amzn.shape)
```

### Line by Line Breakdown

1. `num_data_amzn = len(dataset_amzn)`: Counts how many data points (e.g., days of stock prices) are in the dataset.  
2. `num_days_used = 15`: Defines the size of the sliding window—15 days.  
3. `data_used_amzn = np.array([])`: Initializes a list to hold 15-day windows of stock prices for processing.

**Setting Up Prediction Data:**

4. `data_to_predict_amzn = np.array(dataset_amzn[num_days_used:, :1])`: Extracts the stock price for each day after the initial 15 days—used for prediction targets.  
5. `dates_used_amzn = stock_amzn.index[num_days_used:num_data_amzn]`: Creates a list of dates corresponding to each 15-day window.  
6. `display(data_used_amzn.shape, data_to_predict_amzn.shape, dates_used_amzn.shape)`: Displays the shape (i.e., dimensions) of each dataset—how many rows and columns are in each array.



```python

train_split = 0.8
#Finding data size with the help of the shape of the dataframe
data_size = data_used_amzn.shape[0]
num_features_amzn = data_used_amzn.shape[2]
train_size_amzn = int(data_size * train_split)
test_size_amzn = data_size - train_size_amzn
#Splitting data into test and train data

#Based on the proportion of training data, we use everything from beginning until the testing point.
X_train_amzn = data_used_amzn[0:train_size_amzn, :, :]

#These are all the actual values of the stock prices for the training data
y_train_amzn = data_to_predict_amzn[0:train_size_amzn, :]

#We use the dates array to store the dates for the training data
dates_train_amzn = dates_used_amzn[0:train_size_amzn]

#we use everything from the training point until the end for testing
X_test_amzn = data_used_amzn[train_size_amzn:, :, :]
y_test_amzn = data_to_predict_amzn[train_size_amzn:, :]
dates_test_amzn = dates_used_amzn[train_size_amzn:]

#Since we can't apply a scaler on a dataframe, we would need to use numpy arrays
#we are also slicing them based on training and testing
unscaled_y_train_amzn = df_amzn['Close'].to_numpy()[(num_days_used):][0:train_size_amzn]
unscaled_y_test_amzn = df_amzn['Close'].to_numpy()[(num_days_used):][train_size_amzn:]

display("X_train shape:", X_train_amzn.shape, "X_test shape:", X_test_amzn.shape, "y_train shape:", y_train_amzn.shape, "y_test shape:", y_test_amzn.shape)
```

## Training and Testing Sets

In most supervised machine learning models we split the dataset into training and testing sets. The purpose of this division is to evaluate the model’s performance on unseen data, ensuring it can generalize well beyond the specific examples it was trained on.

Normally, data is split randomly. However, this approach doesn’t work well for time-series data, such as financial markets, because the data is sequential. Today’s market behavior is influenced by what happened yesterday or in previous days.

If we randomly split this kind of data, we risk including future data in our training set and past data in our testing set. This setup would lead to a misleading evaluation of the model's performance because the model might get "hints" about the future in its training phase. In reality, these future insights would not be available when making real-world predictions. For instance, if the model is trained on data including next week's stock prices and tested on this week's prices, it could unrealistically appear highly accurate, because it's effectively 'cheating' by using knowledge from the future.

To avoid this, we should split time-series data chronologically. Typically, earlier data is used for training, and the most recent data, which the model has never seen during training, is reserved for testing. This approach mirrors real-world scenarios where predictions are made for future events based on past and present data. By training on past data and testing on future data (chronologically), we can more accurately assess the model's ability to make predictions about unseen future events.



## Single Layer LSTM model
Before we start defining the LSTM model, we need to understand how these models actually process the data. Be sure to read through it a couple of times and understand the different components involved in training.

A LSTM has the ability to let the information flow through it via the cell state, (the line running on the top). The flow of information, addition of new information and removal of irrelevant bits and pieces from the past is controlled by using different types of gates.

### Inputs In An LSTM Cell

1. X_t: The current input data. This is the information that we have selected for this specific time period.
2. h_t-1: This cell contains the processed output from the previous cell. It has gone through all the computations we have described below in its previous cell and we would be using it (or atleast some proportion, depends on the sigmoid function),  to decide the next output.
3. C_t-1: The memory from the previous cell. It is represented by the pipe running across the cell. This memory is kept continuous across all the cells in the past, with each unit deciding how much of it do we need to retain and send ahead. Below, you'll see how we process it and send the updates to the next cell.

Now, moving on, we'll see what gates are present in the cell and how they manage the data amongst them:


1. Forget Valve/Gate:
The forget gate is the first critical component of the LSTM unit. Its primary function is to decide what information should be discarded from the cell state, which is crucial for the model's ability to forget irrelevant data. The gate uses a sigmoid activation function that outputs values between 0 and 1.  A value close to 0 signifies that the information should be forgotten, while a value close to 1 indicates that the information is important and should be retained. The forget gate takes the previous memory state and the current input as its inputs, combining them to decide the fate of each piece of information in the cell state.

2. Input Gate
Following the forget gate is the input gate, which has a dual function: it decides which values in the cell state to update and also creates a vector of new candidate values that could be added to the state. This gate also uses a sigmoid layer to decide which values will be updated, outputting a range between 0 and 1, where 1 indicates a complete update and 0 signifies no update. Simultaneously, a tanh layer creates a vector of new candidate values. These values are in the range of -1 to 1, thanks to the tanh activation function, which helps regulate the network. The tanh output represents a normalized version of the new information to be added to the cell state. The input gate's decision about which values to update and the new candidate values together determine how the cell state is modified.


While input and output gates are working to manage the combination between past and current information, the cell state is updated based on the outputs of the forget and input gates. The forget gate output decides which parts of the cell state are retained, while the input gate provides new information to be added.
Think of the cell state as the memory unit in the network and the other gates controlling what gets memorized.
This update process is crucial for the LSTM's ability to retain long-term dependencies, allowing it to remember information over long sequences of data, which is particularly important when it comes to stock market data.

3. Output Gate:
The final gate in the LSTM unit is the output gate. Its role is to determine the next hidden state, which is a filtered version of the cell state. This gate also uses a sigmoid function to decide which parts of the cell state will be output. The actual output is further processed by a tanh function, ensuring that the values are between -1 and 1, and then it's multiplied by the output of the sigmoid gate. This process allows the LSTM to output information that is based on both the current input and the long-term context stored in the cell state. The output then becomes part of the input for the next step in the sequence, or it can be used as the final output of the network.





```python
# Creating Model 1
input_amzn = Input(shape=(15, 1), name = 'input_amzn')
x1 = LSTM(160, return_sequences=False, name='LSTM1_amzn')(input_amzn)
output1 = Dense(1, name='amzn_final')(x1)
model1 = Model(inputs = input_amzn, outputs = output1)
adam = Adam(learning_rate=0.005)
model1.compile(optimizer=adam, loss='mse')
model1.summary()
```

In the cell above, we have declared a very simple 160 neuron LSTM model. We will learn more about the syntax specifically later on.


```python
#Visualize the model
plot_model(model1, to_file='model3.png', show_shapes=True, show_layer_names=True)
```


```python
history = model1.fit(x=X_train_amzn, y=y_train_amzn, batch_size=32, epochs=40, validation_split=0.2, shuffle=False)
evaluation = model1.evaluate(X_test_amzn,y_test_amzn)
print(evaluation)
```


```python
y_train_amzn_pred = model1.predict(X_train_amzn)
y_test_amzn_pred = model1.predict(X_test_amzn)
```


```python
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()
```
![Model results](../../../assets/nbk1_1.png)

This graph, which shows the training and validation loss is key in evaluating the model's performance.
Let's begin with understanding what training and validation loss are, and what they represent.

1. Training Loss

It is a measure of how well a machine learning model is performing on the data it is being trained on. Imagine you're teaching someone to play a new game. Each mistake they make while learning is like a 'loss'. In machine learning, this 'loss' is calculated using a specific formula that measures the difference between the model's predictions and the actual outcomes (in our case, that's the stock prices predicted by the model and actual prices in the market). A high training loss means the model is making many mistakes and is struggling to learn well. A low training loss indicates that the model is doing a good job at learning from the training data.

2. Validation Loss

Validation loss, on the other hand, measures how well the model performs on a set of data it hasn't seen during training. Going back to the game analogy, imagine now testing the person on a slightly different version of the game they learned. How they perform now is akin to the validation loss. It's crucial because it helps you understand if the person can transfer what they've learned to new, slightly different context. A high validation loss means the model isn't good at applying what it's learned to new data, which is a problem known as overfitting. A low validation loss is ideal since it indicates the model generalizes well to new data.

### The Graph

We can see that the training loss is fairly high at approximately 1000 units, which means the model misses the stock price by approximately $30 in either direction everytime (the calculation here is based on Mean Square Error metric, which you can see in the model definition).

The validation loss is much higher at 6000 units. This is extremely problematic because this implies the model is struggling to apply the information it has learned to the new context. Or, has it even learned anything? You can try answering that question by looking at the next graph.


```
start_pt = len(y_train_amzn)
end_pt = start_pt + len(y_test_amzn_pred2)
x_axis_pred = [i for i in range(start_pt, end_pt)]
```


```python
# Prediction data test
plt.gcf().set_size_inches(12, 7)
plt.plot(dataset_amzn[15:, :], label='real')
plt.plot(y_train_amzn_pred[:, :], label='trained')
plt.plot(x_axis_pred, y_test_amzn_pred, label='predicted')

#plt.legend(['real amzn','real googl','real bll','real qcom','predict amzn','predic googl','predict bll','predict qcom'])
plt.ylabel('Price (USD)')
plt.title('Real vs. Predicted')

plt.show()
```
![Model results](../../../assets/nbk1_2.png)

```python
print("Mean Absolute Error:", mean_absolute_error(y_test_amzn, y_test_amzn_pred))
print("Mean Squared Error:", mean_squared_error(y_test_amzn, y_test_amzn_pred))
```

```
Mean Absolute Error: 25.00092089922845
Mean Squared Error: 975.3245557179334
```

### Results
In the graph above, you can see the training predictions in orange, the model's training predictions in green. It is clearly evident that the the model loses track of the entire process at some point. The straight lines parallel to the x-axis indicates the inability to capture any variations in the price movements. The model essentially output the same number as its prediction since it could not fit the values that were passed in x-train beyond that point.

## Consequences of Not Standardizing


Now, why could this happen? Is it because our model is too simple perhaps or could it have something to do with how we processed our inputs?

Let's think about the inputs theory. The prices changed quickly after 2018, the model was still able to capture a lot of these fluctuations but it completely loses track when the prices soar rapidly in 2020. There is a possibility that this sudden change in training data might have made it difficult for the model to relate it to the past data it had been trained on.


```python
normalizer = MinMaxScaler(feature_range=(0,1)) # instantiate scaler
normalized_amzn = normalizer.fit_transform(dataset_amzn) # values between 0,1
```

## Example 2: A fixed Model
So, in order to see how this model performs when we rescale the inputs, we perform normalization with min max scaler.
This is important because it ensures that no single feature dominates the model's learning process due to its scale. Among various methods, the MinMaxScaler is commonly used for normalization.

The MinMaxScaler transforms features by scaling each feature to a given range, usually 0 to 1, or -1 to 1. The transformation is calculated using the formula:




By applying this formula, MinMaxScaler shifts and rescales the data into the range [0, 1] (if no other range is specified). This range scaling makes the data more suitable for algorithms that are sensitive to the scale of input data, such as gradient descent-based algorithms, by ensuring that each feature contributes approximately proportionately to the final decision.


Normalizing the prices means a model won't be unduly influenced by the scale of the fluctuations but instead can focus on the relative changes and trends, which are more informative. It would help in making the importance of all the features more uniform and therefore ideally increase its ability to learn well.

We will see as we move ahead how true this statement is.


```python

data_used_amzn = np.array([normalized_amzn[i : i + num_days_used].copy() for i in range(num_data_amzn - num_days_used)])
data_to_predict_amzn = np.array(normalized_amzn[num_days_used:, :1])

#Based on the proportion of training data, we use everything from beginning until the testing point.
X_train_amzn = data_used_amzn[0:train_size_amzn, :, :]

#These are all the actual values of the stock prices for the training data
y_train_amzn = data_to_predict_amzn[0:train_size_amzn, :]

#we use everything from the training point until the end for testing
X_test_amzn = data_used_amzn[train_size_amzn:, :, :]
y_test_amzn = data_to_predict_amzn[train_size_amzn:, :]

display("X_train shape:", X_train_amzn.shape, "X_test shape:", X_test_amzn.shape, "y_train shape:", y_train_amzn.shape, "y_test shape:", y_test_amzn.shape)
```
```
X_train shape:
(1851, 15, 1)
X_test shape:
(463, 15, 1)
y_train shape:
(1851, 1)
y_test shape:
(463, 1)
```
```python
history = model1.fit(x=X_train_amzn, y=y_train_amzn, batch_size=32, epochs=40, validation_split=0.2)
evaluation = model1.evaluate(X_test_amzn,y_test_amzn)
```


```python
y_train_amzn_pred = model1.predict(X_train_amzn)
y_test_amzn_pred = model1.predict(X_test_amzn)
```


```python
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs[1:], loss[1:], 'b', label='Training loss')
plt.plot(epochs[1:], val_loss[1:], 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()
```
![Model results](../../../assets/nbk1_3.png)

In the previous cell, we did not change the specification of the model, or the parameters for fitting it. The only difference was standardizing the inputs with the help of min-max alogorithms. As you can clearly see how the training and validation loss has improved drastically by this change.

Not only is the loss much lower than before, we can also see how the gap between both of them has narrowed down. This implies our model was actually able to generalize what it learned from its training data to the validation dataset.


```python
mae_norm = mean_absolute_error(y_test_amzn, y_test_amzn_pred)
mse_norm = mean_squared_error(y_test_amzn, y_test_amzn_pred)
```


```python
print("Mean Absolute Error:",normalizer.inverse_transform(np.array([mae_norm]).reshape(-1,1)))
print("Mean Squared Error:", normalizer.inverse_transform(np.array([mse_norm]).reshape(-1,1)))
```




```python
# Prediction data test
plt.plot(normalized_amzn[15:, :], label='real')
plt.plot(y_train_amzn_pred[:, :], label='trained')
plt.plot(x_axis_pred, y_test_amzn_pred, label='predicted')
#plt.legend(['real amzn','real googl','real bll','real qcom','predict amzn','predic googl','predict bll','predict qcom'])
plt.ylabel('Price (USD)')
plt.title('Real vs. Predicted')
plt.gcf().set_size_inches(12, 5)
plt.show()
```
![Model results](../../../assets/nbk1_4.png)

As you can see from the graph, this change alone led to better training AND testing performance. When we passed absolute values, a lot of information about the prices increasing wildly was lost. But now, it is visible how the model was able to focus on the pattern of price movement.

## Example 3: Can a model be TOO good?

So far, you saw how lack of standardizing the inputs can lead to the model losing its ability to learn from the data. We have not touched upon the model specifications and how they contribute to its effectiveness yet. This section shows how having a model that can learn "quickly" with an increased sensitivity to training data might actually be a bad choice.


```python
# Creating Model 2
input_amzn = Input(shape=(15, 1), name = 'input_amzn')

x2 = LSTM(300, return_sequences=False, name='LSTM1_amzn')(input_amzn)

output2 = Dense(1, name='amzn_final')(x2)
model2 = Model(inputs = input_amzn, outputs = output2)
adam = Adam(learning_rate=0.005)
model2.compile(optimizer=adam, loss='mse')
model2.summary()
```

In the code sample above, 2 paramters have changed, can you guess what they are?

Answer: The LSTM model has more cells, which means the data would be learned by the model quiet extensively. The second element that has changed is, the learning rate in the optimizer. For this example, we have increased it ten times compared to before.

You'd assume that this makes the model much better and improves all the accuracy scores, enabling us to predict the market more effectively. But, focus on the next graph and see what has happened instead.


```
history2 = model2.fit(x=X_train_amzn, y=y_train_amzn, batch_size=32, epochs=40, validation_split=0.2, shuffle=False)
evaluation2 = model2.evaluate(X_test_amzn,y_test_amzn)
```


```python
loss = history2.history['loss']
val_loss = history2.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs[15:], loss[15:], 'b', label='Training loss')
plt.plot(epochs[15:], val_loss[15:], 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()
```
![Model results](../../../assets/nbk1_5.png)

## Validation Results

If you look closely, you'd find that the validation loss has stagnated at a particular level and does not decrease from that point. This implies that model has reached a stable state and its predictions do not vary as much.

Low training loss indicates that this model has fit the data well, since the output of the model on the data its trained on does not vary when compared to its source.


```python
# Prediction data test
plt.plot(normalized_amzn[15:, :], label='real')
plt.plot(y_train_amzn_pred2[:, :], label='trained')
plt.plot(x_axis_pred, y_test_amzn_pred2, label='predicted')

#plt.legend(['real amzn','real googl','real bll','real qcom','predict amzn','predic googl','predict bll','predict qcom'])
plt.ylabel('Price (USD)')
plt.title('Real vs. Predicted')
plt.gcf().set_size_inches(12, 5)
plt.show()
```
![Model results](../../../assets/nbk1_6.png)

```python
plt.plot(x_axis_pred, y_test_amzn_pred2, label='predicted')
plt.plot(x_axis_pred, normalized_amzn[1866:,0], label='real')
plt.title('Real vs. Predicted (zoomed in'))
plt.legend()
```

![Model results](../../../assets/nbk1_7.png)


## Overfitting
In the graph above, the model's predictions in orange seem to trace perfectly with the stock price. Essentially, the model has fit the training data too well, capturing spurious patterns and anomalies that do not generalize to unseen data.

This also makes sense when you reconsider the validation loss graph above. The model's performance stays constant when it comes to unseen validation data, suggesting the lack of generalization in prediction.

This is problematic because the primary goal of a machine learning model is to make accurate predictions on new, unseen data, not just to perform well on the data it was trained on. Overfitting undermines this goal by making the model less adaptable and more error-prone when exposed to new situations or data sets.

Your model's performance isn't exclusively evaluated by its training accuracy. Be cautious when your model is "too accurate", it is entirely possible that it has overfit to training data and wouldn't perform well in real life.


## Conclusion
This notebook has introduced you to the basics of machine learning. The first model showed us how it is important to properly rescale the inputs in order to allow the model to capture information from them and generalize. Then, looking at LSTMs architecture and skimming through it would have helped in envisioning what calculation take plce inside a cell. The next two examples helped you see what underfitting and overfitting the data means.

It is completely fine if you don't understand everything throughly. These ideas will become clearer as you go through the next 2 notebooks.

## What's Next?

In the next notebook, we will see how we can optimize a model's performance by using multiple stocks as features and what other models apart from LSTMs can be used for time-series prediction. The next notebook also covers how we can write cleaner and more modular code to pre-process the data and train the models.
