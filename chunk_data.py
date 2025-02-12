import json
import pandas


with open('insert_data_file.json', 'r') as file:
    data = json.load(file)


for doc in data:
    