**Sentiment Analyser**

This program uses SKLearn's sentiment analysis techniques to classify IMDB reviews.

There are 25,000 positive and 25,000 negative reviews.

It uses the basic SKLearn stochastic gradient descent classifier.

I reduced the features based on the chiSquare information gain. Keeping the top 1% approximately. This attains about the same to a slightly higher test score and runs in about a third of the time in practice.

Changed the loss function to hinge and commented out everything but the bigram classifiers.

Also tweaked the parameters to minimize bias and variance. Increased maxIter to 10000 and increased the regularization paramater (alpha) to .01.

Still only at 89% test accuracy (98% training accuracy). I believe the problem is now high variance but this stems from having too little data or improperly processed data. Increasing the regularization parameter further brings the train and test score closer together but lowers the test score. 

Could probably improve the data used. Used chisquare to reduce vocabulary but there are likely better, domain specific ways of choosing which words to consider.

Realized I tuned parameters on the test set so test accuracy may not be entirely accurate. Will generate cross validation sets and try again.