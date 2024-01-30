import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import pandas as pd
import re
import tflearn
import tensorflow as tf
import json
import pickle
import random
import matplotlib.pyplot as plt
import random

import os
import csv
import openai
import subprocess
import json
from dotenv import load_dotenv
from io import StringIO

#define the current, parent and data directories
working_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(working_dir)  # This gets the parent directory of the current script.

data_dir = os.path.join(parent_dir, '..', 'Data')
parent_dir = os.path.abspath(parent_dir) # get teh absolute path of teh parent directory to import necessary modules and then remove again
data_dir = os.path.abspath(data_dir)

import data_support as ds


load_dotenv()

org_key = os.getenv('OPENAI_API_ORG')
api_key = os.getenv('OPENAI_API_KEY')

openai.organization = org_key
openai.api_key = api_key

prompt_path = os.path.join(working_dir, 'input4.txt')
#prompt_path = '/Users/sam.fleming/Desktop/IT Projects/STRC Capstone/Prototype 3.3/P3-3/DataGen/input4.txt'

dataset_name = "small"

# #original breadth of categories
# dataset_target= {('TVFALLEN', 1000),  ('RDK&G', 1000), ('DAMAGEBIN', 1000), ('RDPOTHOLE', 1000),
#                 ('ABANDBIN', 1000), ('STOLENBIN', 1000), ('OHANGBRAN', 1000), ('MISSEDBIN', 1000),
#                 ('ABANDONVEH', 1000), ('TVTRIM', 1000), ('STRSWEEP', 1000)
# #                 ('BINADDCOMM', 500), ('BINADDRMUD', 500), ('NEWPUBBIN', 500), ('OVREPUBBIN', 500)}
#                 }

#test breadth
dataset_target = {('TVFALLEN', 100),  ('RDK&G', 100), ('DAMAGEBIN', 100)}

# #large scaled up breadth of categories
# dataset_target_large =  {('TVFALLEN', 1000),  ('RDK&G', 1000), ('DAMAGEBIN', 1000), ('RDPOTHOLE', 1000),
#                         ('ABANDBIN', 1000), ('STOLENBIN', 1000), ('OHANGBRAN', 1000), ('MISSEDBIN', 1000),
#                         ('ABANDONVEH', 1000), ('TVTRIM', 1000), ('STRSWEEP', 1000), 
#                         ('BINADDCOMM', 500), ('BINADDRSUD', 500), ('BINADDRMUD', 500), ('NEWPUBBIN', 500), ('NEWPUBBIN', 500), ('OVREPUBBIN', 500),
#                         ('ANIENQ', 1000), ('ANIFOUND', 1000), ('ANILOST', 1000),
#                         ('FEEDBACK', 1000), ('GENREQINFO', 1000), ('GENREQMEET', 1000), 
#                         ('PBPLNENQ', 1000), ('PBPLUENQ', 1000), ('PBNONCOMPL', 1000),
#                         ('TROBUSK', 1000), ('TROCAMPING', 1000)
#                         }

dataset_breadth = []
for cat in dataset_target:
    dataset_breadth.append(cat[0])


openai.Model.list()

def subprocess_test():
    '''
    Refer the official OpenAI API tutorial
    '''
    
    completed_process = subprocess.run(['ls', '-l'])
    print('Return code:', completed_process.returncode)
    
    completed_process = subprocess.run(['ls', '-l'], capture_output=True, text=True)
    completed_process = subprocess.run(['ls', '-l'], stdout=subprocess.PIPE, text=True)
    print('Return code:', completed_process.returncode)
    print('Have {} bytes in stdout:\n{}'.format(len(completed_process.stdout), completed_process.stdout))

    retcode = subprocess.call(['ls', '-l'])
    print('Return code:', retcode)

def openai_response_test(input_text, key, cat="[Unspecified]"):
    '''
    Refer the official OpenAI API tutorial
    '''
    cmd = [
        'curl',
        'https://api.openai.com/v1/chat/completions',
        '-H', 'Content-Type: application/json',
        '-H', f'Authorization: Bearer {key}',
        '-d', json.dumps({
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": input_text}],
            "temperature": 0.7
        })
    ]

    #run cmd and capture output - load as JSON
    result = subprocess.run(cmd, capture_output=True, text=True)
    # print("\nResult.stdout:")
    # print(result.stdout)
    if result.stdout.strip():
        response = json.loads(result.stdout)
    else:#handle cases where openai gives empty response
        print(f"\nEmpty Response Recieved for category: {cat}\n")
        return None

    # check if there was an error in the response
    if 'error' in response:
        print(f"\nError from OpenAI: {response['error']}\n")

    # extract response, if the 'choices' key exists
    try:
        response_text = response['choices'][0]['message']['content']
        return response_text
    except KeyError:
        print("\nUnexpected response structure from OpenAI.\n")
        return None


output_path = os.path.join(data_dir, f'{dataset_name}.csv')

print("\All the Directories:")
print(f"{working_dir}\n{parent_dir}\n{data_dir}")

subprocess_test()


set_h, set_h_target = ds.get_set_heirachy_tuples(dataset_target)
desc = ds.get_set_desc(set_h)

#get prompt skeleton from the file
with open(prompt_path, 'r') as file:
    prompt_file = file.read()


#get unique RequestTypes from set
unique_request_types = {tup[0] for tup in set_h}

#get unique RequestCategories from set
unique_request_categories = {tup[1] for tup in set_h}

counter = 0

initial_state = ds.get_current_state(set_h, output_path)
print("\n" + dataset_name + " Starting State:")
print(initial_state)
remaining_state = ds.get_remaining_counts(initial_state, set_h_target)
print("\nRemaining State:")
print(remaining_state)


while (sum(remaining_state.values()) > 0):
    non_zero_items = [k for k, v in remaining_state.items() if v > 0]
    cat = random.choice(non_zero_items)
    
    print("\nCategory: " + str(cat))
    #Get the description from the selected category
    set_desc = ''
    for item in desc:
        if cat[0] == item[0] and cat[1] == item[1] and cat[2] == item[2]:
            set_desc = item
    #write category and descripion into prompt
    prompt_text = prompt_file.replace("[set_desc]", ','.join(set_desc))
    cat_str = ','.join(cat)

    prompt_text = prompt_text.replace("[cat]", cat_str)
    #write number of samples into the prompt
    if remaining_state[cat] < 80:
        num = remaining_state[cat]
    else:
        num = 80
    prompt_text = prompt_text.replace("[num]", str(num))

    #run prompt in gpt dialog

    response = openai_response_test(prompt_text, api_key)

    #extract relevant rows from the response + write to csv
    rows = ds.extract_data(response, unique_request_types)

    ds.write_to_csv(rows, output_path)
    state = ds.get_current_state(set_h, output_path)
    #
    remaining_state = ds.get_remaining_counts(state, set_h_target)
    print("\nRemaining State:")
    print(remaining_state)

#After generating, clean data for any nuances
ds.dataset_clean(set_h, output_path, new_file = False)