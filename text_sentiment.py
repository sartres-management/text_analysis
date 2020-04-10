# apply textblob into sentiment analysis

from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
import pandas as pd
import numpy as np
import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
#nltk.download('movie_reviews')

dataset = pd.read_csv("airbnb-reviews-part.csv",  sep = ';')

comments = dataset.iloc[0:500,5].values

#print(comments)

comment = TextBlob(comments[120], analyzer=NaiveBayesAnalyzer())
print(comment)
print(comment.tags)
print(comment.sentiment)


