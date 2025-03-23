"""
Our dataset named cd_dataset that we created in the previous question has 2 available attributes availale to describe our dataset.

Please print the values as strings for each attribute.
"""
from question_3 import cd_dataset

print(f"Annotations data: \n{cd_dataset.df}")
print(f"Classes: {cd_dataset.class_list}")