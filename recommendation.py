"""
Created on Fri Mar 16 17:58:42 2018

@author: vaghanideep
"""

#import modules 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

#Upload Dataset
data_source= pd.read_csv("/Users/vaghanideep/Desktop/Sample-data.csv")


tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, 
                     stop_words='english')

tfidf_matrix = tf.fit_transform(data_source['description'])



#similarities matrix to generate score
cosine_similarities_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

#this will iterate over all the rows in the matrix and store score results in a list 
results = {}
for idx, row in data_source.iterrows():
    similar_indices = cosine_similarities_matrix[idx].argsort()[:-100:-1]
    similar_items = [(cosine_similarities_matrix[idx][i], data_source['id'][i]) 
    for i in similar_indices]
    results[row['id']] = similar_items[1:]
print('Complete!')    



def item(id):
    return data_source.loc[data_source['id'] == id]['description'].tolist()[0].split(' - ')[0]


def recommend_me(item_id, num):
    print("If you like " + item(item_id) + str(num) + 
          " then these are the top products you will also like" + "...")
    print("----------------------------------------")
    recs = results[item_id][:num]
    for rec in recs:
        print("Recommended: " + item(rec[1]) + " (score:" + str(rec[0]) + ")")
        
#Prints out the recommended items after the input from user        
recommend_me(item_id=35, num=3)
