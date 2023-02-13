import random
import pandas as pd
import numpy as np
from statistics import fmean, stdev
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from tabulate import tabulate
from fsga import GA_FeatureSelection
from rfs import RandomFeatureSelection


def test_data(X, Y, pstr, nv, al, md, al_string, md_string):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=100)
    if nv > 0:
        X_train = null_values(X_train, nv)
    if al_string != "No Feature Selection":
        X_train = al.fit_transform(X_train, Y_train)
    if nv > 0:
        X_train['Target'] = Y_train
        X_train.dropna(inplace=True)
        Y_train = X_train.Target.to_list()
        X_train.drop(columns=['Target'], inplace=True)
    md.fit(X_train, Y_train)
    if al_string != "No Feature Selection":
        X_test = al.transform(X_test)
    Y_pred = md.predict(X_test)
    score = f1_score(Y_test, Y_pred, average='weighted')
    if al_string == "No Feature Selection" or al_string == "Variance Threshold":
        print("The accuracy of", pstr, "using", al_string, "with the", md_string, "classifier is:",
              round(score, 6) * 100, "%")
    else:
        print("The accuracy of", pstr, "with", nv, "% Null using", al_string, "with the", md_string, "classifier is:",
              round(score, 6) * 100, "%")
    return score


def test_run(X, Y, pstr, md, md_string, rf_runs=10, al_runs=10):
    feature_selection_values = [None, VarianceThreshold(threshold=0.8), GA_FeatureSelection(),
                                RandomFeatureSelection()]
    feature_selection_string = ["No Feature Selection", "Variance Threshold", "Genetic Algorithm",
                                "Random Feature Selection"]
    if md_string != "RandomForestClassifier":
        Row_0 = [pstr]
        Row_1 = [pstr + " with 1% Null Values", "NULL", "NULL"]
        Row_5 = [pstr + " with 5% Null Values", "NULL", "NULL"]
        Row_10 = [pstr + " with 10% Null Values", "NULL", "NULL"]
        for a in range(len(feature_selection_values)):
            if a < 2:
                result = test_data(X, Y, pstr, 0, feature_selection_values[a], md,
                                   feature_selection_string[a], md_string)
                result = float(format(result, '.5f'))
                Row_0.append(result)
            else:
                result_list = []
                for b in range(al_runs):
                    result = test_data(X, Y, pstr, 0, feature_selection_values[a], md,
                                       feature_selection_string[a], md_string)
                    result_list.append(result)
                result = fmean(result_list)
                result = float(format(result, '.5f'))
                Row_0.append(result)
                deviation = stdev(result_list)
                deviation = float(format(deviation, '.5f'))
                Row_0.append(deviation)
                result_list = []
                for b in range(al_runs):
                    result = test_data(X, Y, pstr, 1, feature_selection_values[a], md,
                                       feature_selection_string[a], md_string)
                    result_list.append(result)
                result = fmean(result_list)
                result = float(format(result, '.5f'))
                Row_1.append(result)
                deviation = stdev(result_list)
                deviation = float(format(deviation, '.5f'))
                Row_1.append(deviation)
                result_list = []
                for b in range(al_runs):
                    result = test_data(X, Y, pstr, 5, feature_selection_values[a], md,
                                       feature_selection_string[a], md_string)
                    result_list.append(result)
                result = fmean(result_list)
                result = float(format(result, '.5f'))
                Row_5.append(result)
                deviation = stdev(result_list)
                deviation = float(format(deviation, '.5f'))
                Row_5.append(deviation)
                result_list = []
                for b in range(al_runs):
                    result = test_data(X, Y, pstr, 10, feature_selection_values[a], md,
                                       feature_selection_string[a], md_string)
                    result_list.append(result)
                result = fmean(result_list)
                result = float(format(result, '.5f'))
                Row_10.append(result)
                deviation = stdev(result_list)
                deviation = float(format(deviation, '.5f'))
                Row_10.append(deviation)
    else:
        Row_0 = [pstr]
        Row_1 = [pstr + " with 1% Null Values", "NULL", "NULL", "NULL", "NULL"]
        Row_5 = [pstr + " with 5% Null Values", "NULL", "NULL", "NULL", "NULL"]
        Row_10 = [pstr + " with 10% Null Values", "NULL", "NULL", "NULL", "NULL"]
        for a in range(len(feature_selection_values)):
            if a < 2:
                result_list = []
                for c in range(rf_runs):
                    result = test_data(X, Y, pstr, 0, feature_selection_values[a], md,
                                       feature_selection_string[a], md_string)
                    result_list.append(result)
                result = fmean(result_list)
                result = float(format(result, '.5f'))
                Row_0.append(result)
                deviation = stdev(result_list)
                deviation = float(format(deviation, '.5f'))
                Row_0.append(deviation)
            else:
                complete_result_list = []
                for c in range(rf_runs):
                    for b in range(al_runs):
                        result = test_data(X, Y, pstr, 0, feature_selection_values[a], md,
                                           feature_selection_string[a], md_string)
                        complete_result_list.append(result)
                result = fmean(complete_result_list)
                result = float(format(result, '.5f'))
                Row_0.append(result)
                deviation = stdev(complete_result_list)
                deviation = float(format(deviation, '.5f'))
                Row_0.append(deviation)
                complete_result_list = []
                for c in range(rf_runs):
                    for b in range(al_runs):
                        result = test_data(X, Y, pstr, 1, feature_selection_values[a], md,
                                           feature_selection_string[a], md_string)
                        complete_result_list.append(result)
                result = fmean(complete_result_list)
                result = float(format(result, '.5f'))
                Row_1.append(result)
                deviation = stdev(complete_result_list)
                deviation = float(format(deviation, '.5f'))
                Row_1.append(deviation)
                complete_result_list = []
                for c in range(rf_runs):
                    for b in range(al_runs):
                        result = test_data(X, Y, pstr, 5, feature_selection_values[a], md,
                                           feature_selection_string[a], md_string)
                        complete_result_list.append(result)
                result = fmean(complete_result_list)
                result = float(format(result, '.5f'))
                Row_5.append(result)
                deviation = stdev(complete_result_list)
                deviation = float(format(deviation, '.5f'))
                Row_5.append(deviation)
                complete_result_list = []
                for c in range(rf_runs):
                    for b in range(al_runs):
                        result = test_data(X, Y, pstr, 10, feature_selection_values[a], md,
                                           feature_selection_string[a], md_string)
                        complete_result_list.append(result)
                result = fmean(complete_result_list)
                result = float(format(result, '.5f'))
                Row_10.append(result)
                deviation = stdev(complete_result_list)
                deviation = float(format(deviation, '.5f'))
                Row_10.append(deviation)

    print(Row_0)
    print(Row_1)
    print(Row_5)
    print(Row_10)

    return Row_0, Row_1, Row_5, Row_10


def null_values(X, nm):
    df = X.copy()
    for x in range(df.shape[0]):
        for y in range(df.shape[1]):
            num = random.randint(1, 100)
            if num <= nm:
                df.iloc[x, y] = np.nan
    return df


# GSE113486 Bladder Cancers vs No Cancer Control
# https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE113486
df1 = pd.read_csv("datasets/GSE113486.txt", sep='\t', comment='!')
df1 = df1.transpose()
df1 = df1.rename(columns=df1.iloc[0])
df1 = df1[1:493]
df1 = df1.reset_index(drop=True)
list1 = [1] * 392
list2 = [0] * 100
list1.extend(list2)
tg1 = pd.Series(list1)

# GSE19804 Lung Cancer vs Lung Normal
# https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE19804
df2 = pd.read_csv("datasets/GSE19804.txt", sep='\t', comment='!')
df2 = df2.transpose()
df2 = df2.rename(columns=df2.iloc[0])
df2 = df2[1:]
df2 = df2.reset_index(drop=True)
list1 = [1] * 60
list2 = [0] * 60
list1.extend(list2)
tg2 = pd.Series(list1)

# GSE112264 Negative Prostate Biopsy vs Prostate Cancer
# https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE112264
df3 = pd.read_csv("datasets/GSE112264.txt", sep='\t', comment='!')
df3 = df3.transpose()
df3 = df3.rename(columns=df3.iloc[0])
df3_part1 = df3[401:642]
df3_part2 = df3[733:1542]
df3 = pd.concat([df3_part1, df3_part2], axis=0)
df3 = df3.reset_index(drop=True)
list1 = [0] * 241
list2 = [1] * 809
list1.extend(list2)
tg3 = pd.Series(list1)

# GSE120584 Alzheimer Disease vs Dementia with Lewy Bodies
# https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE120584
df4 = pd.read_csv("datasets/GSE120584.txt", sep='\t', comment='!')
df4 = df4.transpose()
df4 = df4.rename(columns=df4.iloc[0])
df4 = df4[1:1191]
df4 = df4.reset_index(drop=True)
list1 = [1] * 1021
list2 = [0] * 169
list1.extend(list2)
tg4 = pd.Series(list1)

ds = load_breast_cancer(as_frame=True)

df_list = [df1, df2, df3, df4, ds.data]
df_string = ["Bladder Cancer", "Lung Cancer", "Prostate Cancer", "AD", "Breast Cancer"]
tg_list = [tg1, tg2, tg3, tg4, ds.target]
classifier_values = [LogisticRegression(max_iter=30000), svm.SVC(), RandomForestClassifier()]
classifier_values_string = ["LogisticRegression", "svm", "RandomForestClassifier"]
headers = ["Dataset", "No Feature Selection", "VarianceThreshold", "Genetic Algorithm Average",
           "Genetic Algorithm Std. Dev.", "Random Feature Selection Average", "Random Feature Selection Std. Dev."]
headers_forest = ["Dataset", "No Feature Selection Average", "No Feature Selection Std. Dev.",
                  "VarianceThreshold Average", "VarianceThreshold Std. Dev.", "Genetic Algorithm Average",
                  "Genetic Algorithm Std. Dev.", "Random Feature Selection Average",
                  "Random Feature Selection Std. Dev."]

for e in range(len(classifier_values)):
    Rows_Complete = []
    print("Using classifier", classifier_values_string[e])
    for f in range(len(df_list)):
        Data_0, Data_1, Data_5, Data_10 =\
            test_run(df_list[f], tg_list[f], df_string[f], classifier_values[e], classifier_values_string[e])
        Rows_Complete.append(Data_0)
        Rows_Complete.append(Data_1)
        Rows_Complete.append(Data_5)
        Rows_Complete.append(Data_10)
    print("Table of results with the", classifier_values_string[e], "classifier")
    if e < 2:
        print(tabulate(Rows_Complete, headers=headers, tablefmt="fancy_grid"))
    else:
        print(tabulate(Rows_Complete, headers=headers_forest, tablefmt="fancy_grid"))
