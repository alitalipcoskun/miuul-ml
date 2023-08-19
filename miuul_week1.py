# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 15:19:09 2023

@author: ali_t
"""


import seaborn as sns
students = ["John", "Mark", "Venessa", "Mariam"]

students_no = ["John", "Venessa"]


m = [ student.upper() if student in students_no else student.lower() for student in students ]

print(m)


#dict comprehension


dicta = {i: i**2 for i in range(0,10) if i % 2 == 0}
print(dicta)


df = sns.load_dataset("car_crashes")
print(df.columns)

"""
df.columns = ["FLAG_" + column.upper() if "INS" in column.upper() else  "NO_FLAG_" + column.upper() for column in df.columns]
print(df.columns)
print(df.head())
"""


newList = ['mean', 'min', 'max', 'sum']

selected_columns = [column for column in df.columns if df[column].dtype != 'O']
#print(selected_columns)
newDict = {column: newList for column in selected_columns}
#print(newDict)
print(df[selected_columns].head())
print(df[selected_columns].agg(newDict))