**Sentiment Analyser**

This program uses SKLearn's sentiment analysis techniques to classify IMDB reviews.

There are 25,000 positive and 25,000 negative reviews.

It uses the basic SKLearn stochastic gradient descent classifier.

I reduced the features based on the chiSquare information gain. Keeping the top 1% approximately. This attains about the same to a slightly higher test score and runs in about a third of the time in practice.