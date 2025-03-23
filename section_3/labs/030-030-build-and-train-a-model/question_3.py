# question_3.py
"""
Print the model layer name, size and first two values of each parameter
"""
# Import the model instance from question_2
from question_2 import model

# Iterate through named parameters of the model
for name, param in model.named_parameters():
    # Print layer name, size and first two values of each parameter
    print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]} \n")