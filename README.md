**Sentiment Analyser**

This program uses SKLearn's sentiment analysis techniques to classify IMDB reviews.

There are 25,000 positive and 25,000 negative reviews.

It uses the basic SKLearn stochastic gradient descent classifier. I'm not really sure how to improve it further. It gets around 87% on the test data.

I thought using a kernel estimator on the data might help but it does not. I'm not sure if this is a problem with how SKLearn constructs the ngrams or if this isn't the right tool for the problem.