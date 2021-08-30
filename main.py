import numpy as np
import pandas as pd
import math
import operator
import random

dataCsv = pd.read_csv("breast-cancer.csv")
dataframe = dataCsv.fillna(dataCsv.mean()).drop(
    [203, 433, 549]).drop(columns=['ID'])
dataframe['Diagnosis'].replace('M', 1, inplace=True)
dataframe['Diagnosis'].replace('B', 0, inplace=True)

test_df = dataframe.sample(frac=0.3)
training_df = dataframe.sample(frac=0.7)


def calculate_euclidean_distance_between_dataframe_rows(test_dataframe_row, training_dataframe_row, column_count):
    distanceSum = 0
    for column in range(0, column_count):
        distanceSum += math.pow(test_dataframe_row[column] -
                                training_dataframe_row[column], 2)
    return math.sqrt(distanceSum)


def knn(training_dataframe, testing_dataframe_row, K):
    distances = {}

    for index in range(len(training_dataframe)):
        euclideanDistance = calculate_euclidean_distance_between_dataframe_rows(
            testing_dataframe_row,
            training_dataframe.iloc[index],
            training_dataframe.shape[1]
        )
        distances[index] = euclideanDistance

    neighbors = sorted(distances, key=distances.get)[:K]

    is_cancer_qtd, is_not_cancer_qtd = 0, 0
    for row_index in neighbors:
        if training_dataframe.iloc[row_index]['Diagnosis'] == 0.0:
            is_not_cancer_qtd += 1
        else:
            is_cancer_qtd += 1

    return 1 if is_cancer_qtd > is_not_cancer_qtd else 0


correctAnswers, totalAnswers, wrongAnswers, K = 5, len(test_df), 0, 5
answersArray = {}
for index in range(len(test_df)):
    print(f'Calculating result for row {index+1} of total {len(test_df)}')
    testing_row = test_df.iloc[index]
    cancer_result = knn(training_df, testing_row, K)
    answersArray[index] = cancer_result
    print('Real value: ' + ('Cancer\n' if testing_row['Diagnosis'] == 1 else 'Not cancer'))
    print('Result: ' + ('Cancer\n' if cancer_result == 1 else 'Not cancer \n'))
    if testing_row['Diagnosis'] == cancer_result:
        correctAnswers += 1
wrongAnswers = totalAnswers - correctAnswers

print("Accuracy:", 100*correctAnswers/len(test_df))