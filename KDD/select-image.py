import pandas as pd
import numpy as np
import shutil as sh
import os

df_all_filepath_expression = pd.read_csv("/home/johnomena/Desktop/DataSet/AffectNET/Manually_Annotated_file_lists/training.csv", usecols = ["subDirectory_filePath","expression"])

df_filepath_expression_happy = df_all_filepath_expression[df_all_filepath_expression['expression'] == 2]

happy_filePath_with_bar = df_filepath_expression_happy['subDirectory_filePath'].values

happy_filePath_without_bar = []

for i in happy_filePath_with_bar:
    new_string = ''
    flag = 0
    for j in i:
        if flag == 1:
            new_string = new_string + j
        if j == '/':
            flag = 1
    happy_filePath_without_bar.append(new_string)        

happy_filePath_without_bar = np.array(happy_filePath_without_bar)

happy_filePath_without_bar_random = np.random.choice(happy_filePath_without_bar, size = 6000, replace = False)

for i in range(1,12):
    for j in happy_filePath_without_bar_random:
        if os.path.isfile("/home/johnomena/Desktop/DataSet/AffectNET/Images/Part" + str(i) + "/" + j):
            sh.copy("/home/johnomena/Desktop/DataSet/AffectNET/Images/Part" + str(i) + "/" + j, "/home/johnomena/data-min/sadness/" + j)
