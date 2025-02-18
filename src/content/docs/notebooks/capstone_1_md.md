---
title: Notebook 1 - Introduction to Financial Time Series Prediction with LSTMs
---

# Introduction
This workbook will help you understand how to create a simple machine learning model to predict stock prices. The learning objectives are:
1. Data Preprocessing: Understanding why preprocessing data is important and how we can use python to simplify it.
2. LSTM models: Learning about the architecture and caulculations performed in a LSTM model and how it is suitable for stock market price prediction.
3. Model Performance: Differentiating when a model is performing as intended and learning about underfitting or overfitting.

## Setting Up

The code below imports all the necessary libraries from relevant packages. We will be using Yahoo Finance's API to collect stock price data.

We are analyzing 13 years of data from four different companies to make these predictions. You are welcome to experiment with a combination of other companies or by adding the time series for the stock market index.

The input data chosen here will determine what our model is trained on. By selecting the type of input data, you are deciding what knowledge your model can capture. Other models have been built using:
1. Macroeconomic Data: Interest rates, unemployment rates, and GDP changes all influence the stock market. These time series can either be a part of your LSTM model or can be added separately.
2. Fundamental Indicators: These include metrics such as profit, EBITDA, and book value related to a specific company. A model can approximate how changes in these values translate to changes in stock prices. This approach closely mirrors the real world and is a viable extension to this project.
3. News and Sentiment Data: Researchers and investors have used commentary by retail investors on open forums like Twitter (now X) and Reddit, or news articles, to perform sentiment analysis and see how that could affect stock prices. Predicting text data with LSTMs can be challenging, as these are primarily designed for time-series data. However, if you can figure out how to perform feature engineering, you might build a model with higher accuracy.

## How does a typical ML model work?
The first step before we dive deep into LSTMs, we need to underestand how a typical machine learning model pipeline works. There are three main elements of a machine learning pipeline:
1. Data Preprocessing and Feature Engineering
2. Initializing and specifying Model Parameters
3. Training, testing and cross-validating the model
Finally, after doing all of that, you can generate Predictions and compare that to real data.

These steps can be further broken down into specific components as we go through the workbook today. You need to focus on the importance of each step in the entire model, and what modelling decisions you end up making intutively by choosing one thing over the other.

# Example 1: Are the numbers too big?



Before we fit our model, we need to preprocess our data. LSTMs take windows of a sequence as inputs. This means they consider a subset of sequential data points at any given time. This approach allows LSTMs to capture temporal relationships within the data. For example, imagine an Excel sheet tracking stock prices over time. Each row represents a day's worth of data. An LSTM might look at a window of 5 rows (days) at a time to predict the price on the 6th day.

In this case, The LSTM looks at a small section of all our stock prices, say 15 days at a time, to guess what the price will be on the 16th day. Then, it moves this window one day forward and repeats the process. This way, it learns from the past to predict the future, by seeing the patterns that have already appeared and we're assuming that is what will happen in the future.

It is important to note that in case of univariate data (what we have here), we are solely considering the past prices of a stock to predict its value in the future.



### Line by Line Breakdown

1. num_data_amzn = len(dataset_amzn): This line counts how many data points (like days of stock prices) you have.
2. num_days_used = 15: This sets up the size of your sliding window. Here, it's 15 days.
3. data_used_amzn = np.array([]): This line creates a new list where each item is a 15-day window of stock prices. It does this for the entire dataset except for the last 15 days.


Setting Up Prediction Data:

4. data_to_predict_amzn = np.array(dataset_amzn[num_days_used:, :1]): This takes the stock price for each day after the first 15 days (since you need the first 15 days to make your first prediction).


5. dates_used_amzn = stock_amzn.index[num_days_used:num_data_amzn]: This line creates a list of dates corresponding to each 15-day window.

6. display(data_used_amzn.shape, data_to_predict_amzn.shape, dates_used_amzn.shape): Finally, this line shows the size or 'shape' of your data. It's like checking how many rows and columns you have in each list: one for the 15-day windows, one for the days you're predicting, and one for the dates of these windows.



## Training and Testing Sets

In most supervised machine learning models we split the dataset into training and testing sets. The purpose of this division is to evaluate the model’s performance on unseen data, ensuring it can generalize well beyond the specific examples it was trained on.

Normally, the data is randomly split into training and testing sets. However, in time-series data, like financial markets, this approach can be problematic. Financial data is inherently sequential; today's market behavior is often influenced by what happened yesterday or in the past days.

If we randomly split this kind of data, we risk including future data in our training set and past data in our testing set. This setup would lead to a misleading evaluation of the model's performance because the model might get "hints" about the future in its training phase. In reality, these future insights would not be available when making real-world predictions. For instance, if the model is trained on data including next week's stock prices and tested on this week's prices, it could unrealistically appear highly accurate, because it's effectively 'cheating' by using knowledge from the future.

To avoid this, we should split time-series data chronologically. Typically, earlier data is used for training, and the most recent data, which the model has never seen during training, is reserved for testing. This approach mirrors real-world scenarios where predictions are made for future events based on past and present data. By training on past data and testing on future data (chronologically), we can more accurately assess the model's ability to make predictions about unseen future events.



# Single Layer LSTM model
Before we start defining the LSTM model, we need to understand how these models actually process the data. Be sure to read through it a couple of times and understand the different components involved in training.

A LSTM has the ability to let the information flow through it via the cell state, (the line running on the top). The flow of information, addition of new information and removal of irrelevant bits and pieces from the past is controlled by using different types of gates.

## Inputs In An LSTM Cell

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




In the cell above, we have declared a very simple 160 neuron LSTM model. We will learn more about the syntax specifically later on.

This graph, which shows the training and validation loss is key in evaluating the model's performance.
Let's begin with understanding what training and validation loss are, and what they represent.

1. Training Loss

It is a measure of how well a machine learning model is performing on the data it is being trained on. Imagine you're teaching someone to play a new game. Each mistake they make while learning is like a 'loss'. In machine learning, this 'loss' is calculated using a specific formula that measures the difference between the model's predictions and the actual outcomes (in our case, that's the stock prices predicted by the model and actual prices in the market). A high training loss means the model is making many mistakes and is struggling to learn well. A low training loss indicates that the model is doing a good job at learning from the training data.

2. Validation Loss

Validation loss, on the other hand, measures how well the model performs on a set of data it hasn't seen during training. Going back to the game analogy, imagine now testing the person on a slightly different version of the game they learned. How they perform now is akin to the validation loss. It's crucial because it helps you understand if the person can transfer what they've learned to new, slightly different context. A high validation loss means the model isn't good at applying what it's learned to new data, which is a problem known as overfitting. A low validation loss is ideal since it indicates the model generalizes well to new data.

### The Graph

We can see that the training loss is fairly high at approximately 1000 units, which means the model misses the stock price by approximately $30 in either direction everytime (the calculation here is based on Mean Square Error metric, which you can see in the model definition).

The validation loss is much higher at 6000 units. This is extremely problematic because this implies the model is struggling to apply the information it has learned to the new context. Or, has it even learned anything? You can try answering that question by looking at the next graph.

### Results
In the graph above, you can see the training predictions in orange, the model's training predictions in green. It is clearly evident that the the model loses track of the entire process at some point. The straight lines parallel to the x-axis indicates the inability to capture any variations in the price movements. The model essentially output the same number as its prediction since it could not fit the values that were passed in x-train beyond that point.

## Consequences of Not Standardizing


Now, why could this happen? Is it because our model is too simple perhaps or could it have something to do with how we processed our inputs?

Let's think about the inputs theory. The prices changed quickly after 2018, the model was still able to capture a lot of these fluctuations but it completely loses track when the prices soar rapidly in 2020. There is a possibility that this sudden change in training data might have made it difficult for the model to relate it to the past data it had been trained on.

# Example 2: A fixed Model
So, in order to see how this model performs when we rescale the inputs, we perform normalization with min max scaler.
This is important because it ensures that no single feature dominates the model's learning process due to its scale. Among various methods, the MinMaxScaler is commonly used for normalization.

The MinMaxScaler transforms features by scaling each feature to a given range, usually 0 to 1, or -1 to 1. The transformation is calculated using the formula:




By applying this formula, MinMaxScaler shifts and rescales the data into the range [0, 1] (if no other range is specified). This range scaling makes the data more suitable for algorithms that are sensitive to the scale of input data, such as gradient descent-based algorithms, by ensuring that each feature contributes approximately proportionately to the final decision.


Normalizing the prices means a model won't be unduly influenced by the scale of the fluctuations but instead can focus on the relative changes and trends, which are more informative. It would help in making the importance of all the features more uniform and therefore ideally increase its ability to learn well.

We will see as we move ahead how true this statement is.

In the previous cell, we did not change the specification of the model, or the parameters for fitting it. The only difference was standardizing the inputs with the help of min-max alogorithms. As you can clearly see how the training and validation loss has improved drastically by this change.

Not only is the loss much lower than before, we can also see how the gap between both of them has narrowed down. This implies our model was actually able to generalize what it learned from its training data to the validation dataset.

As you can see from the graph, this change alone led to better training AND testing performance. When we passed absolute values, a lot of information about the prices increasing wildly was lost. But now, it is visible how the model was able to focus on the pattern of price movement.

# Example 3: Can a model be TOO good?

So far, you saw how lack of standardizing the inputs can lead to the model losing its ability to learn from the data. We have not touched upon the model specifications and how they contribute to its effectiveness yet. This section shows how having a model that can learn "quickly" with an increased sensitivity to training data might actually be a bad choice.

In the code sample above, 2 paramters have changed, can you guess what they are?

Answer: The LSTM model has more cells, which means the data would be learned by the model quiet extensively. The second element that has changed is, the learning rate in the optimizer. For this example, we have increased it ten times compared to before.

You'd assume that this makes the model much better and improves all the accuracy scores, enabling us to predict the market more effectively. But, focus on the next graph and see what has happened instead.

## Validation Results

If you look closely, you'd find that the validation loss has stagnated at a particular level and does not decrease from that point. Low training result indicates that this model has fit the data well, however mediocre validation result might imply that the model fails to generalize its predictions.

## Overfitting
In the graph above, the model's predictions in orange seem to trace perfectly with the stock price. Essentially, the model has fit the training data too well, capturing spurious patterns and anomalies that do not generalize to unseen data.

This also makes sense when you reconsider the validation loss graph above. The model's performance stays constant when it comes to unseen validation data, suggesting the lack of generalization in prediction.

This is problematic because the primary goal of a machine learning model is to make accurate predictions on new, unseen data, not just to perform well on the data it was trained on. Overfitting undermines this goal by making the model less adaptable and more error-prone when exposed to new situations or data sets.


## Conclusion
This notebook has introduced you to the basics of machine learning. The first model showed us how it is important to properly rescale the inputs in order to allow the model to capture information from them and generalize. Then, looking at LSTMs architecture and skimming through it would have helped in envisioning what calculation take plce inside a cell. The next two examples helped you see what underfitting and overfitting the data means. It is completely fine if you don't understand it well now. These ideas will get stronger as you go through the next 2 notebooks.

Your main takeaway should be understanding how a basic machine learning model works and how the following steps are involved: a) data pre-processing, b) Defining the model and fitting it
