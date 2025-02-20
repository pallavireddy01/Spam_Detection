import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df=pd.read_csv('fraud(in).csv')
df.head()

x_train,x_test,y_train,y_test=train_test_split(df['message'],df['label'],test_size=0.2,random_state=42)

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,classification_report

model=sklearn.pipeline.make_pipeline(TfidfVectorizer(),MultinomialNB())

model.fit(x_train,y_train)

y_predict=model.predict(x_test)

accuracy=accuracy_score(y_test,y_predict)
print(f'Accuracy: {accuracy * 100:.2f}%')
print("classification report",classification_report(y_test,y_predict))

def detect_fraud(msg):
  prediction=model.predict([msg])[0]
  return "Fraud Message Detected!" if prediction=="fraud" else "Legitimate Message."

new_msg=input("")
print(detect_fraud(new_msg))
