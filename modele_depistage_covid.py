import numpy
import seaborn
import matplotlib.pyplot as plt
import pandas
import csv
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df_covid_original = pandas.read_csv("dataset_covid.csv",sep = ",", encoding='latin') 

pandas.set_option("display.max_column", 111)

df_covid_original_sans_nan = df_covid_original.fillna(0)
df_covid_original_sans_nan = df_covid_original_sans_nan.replace("not_detected", 0)
df_covid_original_sans_nan = df_covid_original_sans_nan.replace("detected", 1)
df_covid_original_sans_nan
df_covid_original_sans_nan = df_covid_original_sans_nan.replace("negative", 0)
df_covid_original_sans_nan = df_covid_original_sans_nan.replace("positive", 1)
df_covid_original_sans_nan = df_covid_original_sans_nan.replace("absent", 0)
df_covid_original_sans_nan = df_covid_original_sans_nan.replace("clear", 1)
df_covid_original_sans_nan = df_covid_original_sans_nan.replace("not_done", 0)
print(df_covid_original_sans_nan)

df_covid = df_covid_original.drop(df_covid_original.iloc[:, 3:], inplace=True, axis=1)
df_covid2 = df_covid_original_sans_nan[["Patient age quantile"	, "SARS-Cov-2 exam result"]]

df_covid_lecture = df_covid_original.iloc[:, :3]
df_covid2 = df_covid_lecture[["Patient age quantile", "SARS-Cov-2 exam result"]]


y = df_covid2["SARS-Cov-2 exam result"]
X = df_covid2[["Patient age quantile"]]





X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state= 44) 



model = LogisticRegression(max_iter = 1000)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, predictions,labels = numpy.unique(predictions)))
print(confusion_matrix(y_test, predictions))