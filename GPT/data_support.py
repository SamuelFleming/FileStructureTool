
'''
data_support.py
Stores all functions supporting data generation. Any function that runs before the csv file
interacts with the rest of the pipeline (including preliminary data cleaning of GPT-related
nuances, data_clean()) is stored in this file. 
'''

import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import pandas as pd
import tflearn
import tensorflow as tf
import json
import pickle
import random
import matplotlib.pyplot as plt
import sys

import os
import re
import csv
import openai
import subprocess
import json
from dotenv import load_dotenv



#define the current, parent and data directories
working_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(working_dir)  # This gets the parent directory of the current script.

data_dir = os.path.join(parent_dir, '..', 'Data')
parent_dir = os.path.abspath(parent_dir) # get teh absolute path of teh parent directory to import necessary modules and then remove again
data_dir = os.path.abspath(data_dir)


sys.path.append(parent_dir)


import support
from support import LossHistory
import dataPreprocessing as da
from ModelClasses import ClassModelComplete, ClassModelSet, Preprocessing
import trainers as tr
sys.path.remove(parent_dir)




old_data_path = data_dir + '/' + 'GPTGeneratedUserInputs1.csv'
heirachy_path = data_dir + '/' + 'ReqHeirachy.csv'

def get_unique_labels(csv_path = old_data_path):
    '''
    Returns the category breadth for a data set not in heirachical format
    - used in transfering old data into heirachichal format
    '''
    labels = set()
    
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            labels.add(row[1])
            
    return labels


def get_set_heirachy(categories_set = None, csv_path = heirachy_path):
    '''
    For a specific (non-hierarchically specified) set, returns the set but expressed with
    respect to request category hierarchy\n
    Params:\n
        categories_set: eg {'MISSEDBIN', 'ABANDBIN', 'OHANGBRAN', 'RDPOTHOLE', 'DAMAGEBIN', ...'STOLENBIN'}\n
        heirachy_path: absolute path to csv containing entire request heirachy\n
    Return:\n
        transformed_set: in format { (RequestType, RequestCategory, RequestSubCatgeory), ....}
    '''
    #print("\nCat Length: " + str(len(categories_set)))
    transformed_set = set()
    data = support.preprocess_csv(csv_path)
    #get rows in a dataframe with allocated columns
    df = pd.DataFrame(data[1:], columns=data[0])
    
    if categories_set != None:
        '''Case where catgeories has been supplied'''
        #get where 'RequestCategory' or 'RequestSubCategory' are in categories_set - add to return variable
        matching_rows = df[df['RequestCategory'].isin(categories_set) | df['RequestSubCategory'].isin(categories_set)]
            #in cases where categories_set has RequestCategory values with numerous subcategories, matching rows includes all subcategory rows.
        # #otherwise to exlude both:
        # matching_rows = df[ (df['RequestCategory'].isin(categories_set) & df['RequestSubCategory']=='') | df['RequestSubCategory'].isin(categories_set)]

        #add matched rows to the transformed_set
        transformed_set.update(matching_rows.apply(lambda row: (row['RequestType'], row['RequestCategory'], row['RequestSubCategory']), axis=1))
    else:
        '''Case where no categories have been provided'''
        for _, row in df.iterrows():
            transformed_set.add((row['RequestType'], row['RequestCategory'], row['RequestSubCategory']))

    
    #print("\nOutput Length: " + str(len(transformed_set)))
    return transformed_set



def get_set_heirachy_tuples(dataset_target, csv_path=heirachy_path):
    '''
    For a specific (non-hierarchically specified) dataset with targets, returns the set but expressed with
    respect to request category hierarchy\n
    Params:\n
        dataset_target - eg: {('MISSEDBIN', 1000), ('ABANDBIN', 1000), ...}\n
        heirachy_path - absolute path to csv containing entire request heirachy\n
    Return:\n
        transformed_set: 'categories_set' in heirachial format { (RequestType, RequestCategory, RequestSubCatgeory), ....}\n
        transformed_set_target: that same as transformed_set but annotated with target values: { ((RequestType, RequestCategory, RequestSubCatgeory), number), ....}
    '''
    
    # Split the dataset_target into categories_set and number mappings
    categories_set = {item[0] for item in dataset_target}
    number_mapping = {item[0]: item[1] for item in dataset_target}

    transformed_set = set()
    transformed_set_target = set()

    data = support.preprocess_csv(csv_path)
    df = pd.DataFrame(data[1:], columns=data[0])

    matching_rows = df[df['RequestCategory'].isin(categories_set) | df['RequestSubCategory'].isin(categories_set)]

    for _, row in matching_rows.iterrows():
        # Form the hierarchical tuple
        hierarchical_tuple = (row['RequestType'], row['RequestCategory'], row['RequestSubCategory'])
        transformed_set.add(hierarchical_tuple)

        # If either RequestCategory or RequestSubCategory is in number_mapping, add the number
        if row['RequestCategory'] in number_mapping:
            transformed_set_target.add((hierarchical_tuple, number_mapping[row['RequestCategory']]))
        elif row['RequestSubCategory'] in number_mapping:
            transformed_set_target.add((hierarchical_tuple, number_mapping[row['RequestSubCategory']]))

    return transformed_set, transformed_set_target


def get_set_desc(input_set, csv_path=heirachy_path):
    '''
    For a specific set, returns the set and category descriptions.
    params:
        input_set - not a set but a dictionary of the breath of request categories. With 'in_tuple()',
            this can be in either heriachical { ('WASTE', 'NEWBINALT', 'BINADDCOM'), ('ROADS', 'ABANDONVEH',),..} or not {'BINADDCOM', 'ABANDONVEH',..}
    '''
    #print("\nInput Length: " + str(len(input_set)))
    transformed_set = set()
    
    data = support.preprocess_csv(csv_path)
    df = pd.DataFrame(data[1:], columns=data[0])

    #checker to ensure entry set is in either basic or heirachichal format
    def in_tuple(tuple_entry, row):
        return tuple_entry[1] == row['RequestCategory'] and tuple_entry[2] == row['RequestSubCategory']

    #iterate through each row in the dataframe
    for _, row in df.iterrows():
        for entry in input_set:
                #robust 
            if isinstance(entry, str) and (entry == row['RequestCategory'] or entry == row['RequestSubCategory']):
                transformed_set.add((row['RequestType'], row['RequestCategory'], row['RequestSubCategory'], row['Description']))
            elif isinstance(entry, tuple) and in_tuple(entry, row):
                transformed_set.add((row['RequestType'], row['RequestCategory'], row['RequestSubCategory'], row['Description']))

    #print("\nOutput Length: " + str(len(transformed_set)))
    return transformed_set



def create_hierarchy_mapping(hierarchy_set):
    """
    Create a mapping dictionary based on the hierarchy set
        - for purpose fo traversing csv old structure to new
    """
    mapping = {}
    for entry in hierarchy_set:
        request_type, request_category, request_subcategory = entry
        mapping[request_category] = (request_type, request_subcategory)
    return mapping

def apply_hierarchy_label(original_df, mapping):
    """
    Update the original (non-heirachal) dataframe with hierarchical labels.
    """
    #create new columns for RequestType and RequestSubCategory, initializing them with default values
    original_df['RequestType'] = 'UNKNOWN'
    original_df['RequestSubCategory'] = ''

    for index, row in original_df.iterrows():
        category = row['tag']
        if category in mapping:
            original_df.at[index, 'RequestType'] = mapping[category][0]
            original_df.at[index, 'RequestSubCategory'] = mapping[category][1]

    #rename 'tag' to 'RequestCategory'
    original_df = original_df.rename(columns={'tag': 'RequestCategory'})

    return original_df[['RequestType', 'RequestCategory', 'RequestSubCategory', 'pattern']]


def extract_data(response, request_types):
    '''
    Using regex, match all rows in a text 'response' that contain one of the request_types
    (used in generating data)
    params:
        - 'response': the string output
        - 'request_types': set of request types, eg: {'ROADS', 'WASTE', 'TREES'}
    '''
    #handle when response = None (irregularity in the GPT response)
    if response == None:
        return None
    
    #create a regex pattern using the unique request types
    pattern_str = r'^((' + '|'.join(request_types) + r'),.*?)$'
    pattern = re.compile(pattern_str, re.MULTILINE)
    
    matches = pattern.findall(response)
    #we are interested in the entire matching line, not just the RequestType
    return [match[0] for match in matches]


#Not working
def extract_data_cat(response, request_categories):
    '''
    Using regex, match all rows in a text 'response' that contain one of the request_categories.
    params:
        - 'response': the string output
        - 'request_categories': set of request categories, eg: {'TVFALLEN', 'DAMAGEBIN'}
    '''
    # Create a regex pattern using the unique request categories
    pattern_str = r'^(.*?,' + '|'.join(request_categories) + r',.*?".*")$'
    pattern = re.compile(pattern_str, re.MULTILINE)
    
    matches = pattern.findall(response)
    # We return the entire matching line
    return matches

def diagnose_matched(rows):
    '''
    Looks at a set of matched rows from the GPT output, and checks whether
    they fit the expected format (3 commas - 4 columns)
    '''
    print("Which rows are not in expected fomat: ")
    for i, row in enumerate(csv.reader(rows)):
        if len(row) != 4:
            print(f"Row {i + 1} has {len(row)} columns: {row}")

def cleanup_pattern_string_2(s):
    '''
    Remove arbitrary number of double quotes, trim whitespace, 
    and then encase the cleaned string in a single set of double quotes.
    '''
    # Remove spaces at the beginning and the end of the string
    s = s.strip()
    
    # Remove all encasing double quotes
    while s.startswith('"') and s.endswith('"'):
        s = s[1:-1].strip()  # Also removing spaces between double quotes
    
    # Encase the cleaned string in a single set of double quotes
    return f'"{s}"'

def cleanup_pattern_string_3(s):
    '''
    Remove arbitrary number of double quotes, trim whitespace, 
    and then encase the cleaned string in a single set of double quotes.
    '''
    # Remove spaces at the beginning and the end of the string
    s = s.strip()
    
    # Remove all encasing double quotes
    while s.startswith('"') and s.endswith('"'):
        s = s[1:-1].strip()  # Also removing spaces between double quotes
    return s

def write_to_csv(matched_rows, output_path, header_indicator= 0):
    '''
    Takes a set of GPT generated 'matched_rows' (rows of a csv in 'RequestType,RequestCategory,RequestSubCategory,pattern' format),
    cleans it of quote uncertainties and whitespaces, and then appends it to a destination csv file
    params:
        matched_rows - output of 'extract_data' function
        output_path - the absolute path of the destination csv
        header_idicator - inidcator variable if the csv write is the first iteration (to include headers)
    '''
    #handle irregularities in GPT (matched_rows = None)
    if matched_rows == None:
        return
    
    header_indicator = check_csv(output_path)

    corrected_rows = []
    for r in matched_rows:
        r_list = r.split(',')
        if len(r_list) > 4:
            r_list = r_list[:3] + [','.join(r_list[3:])]
        # Clean up the pattern string
        #print("\n 'r_list[-1]' = " + r_list[-1])
        r_list[-1] = cleanup_pattern_string_3(r_list[-1])
        #print(r_list)
        corrected_rows.append(r_list)
    # print("corrected_rows: ")
    # for row in corrected_rows:
    #     print(row)

    df = pd.DataFrame(corrected_rows, columns=['RequestType', 'RequestCategory', 'RequestSubCategory', 'pattern'])

    # Write to CSV manually using csv.writer
    with open(output_path, 'a', newline='') as f:  
        writer = csv.writer(f)
        if header_indicator == 0:
            # Write header at first iteration
            writer.writerow(df.columns.tolist())
        #write rows
        for index, row in df.iterrows():
            # Write each column without quotes, except the last column
            writer.writerow(row)

def criteria_fourElements(df, set_h, prev_state):
    # Criteria: each line has four elements
    df = df[df.columns[df.columns.isin(['RequestType', 'RequestCategory', 'RequestSubCategory', 'pattern'])]]
    state = get_current_state_df(set_h, df)
    print("\nNon Four Elements: " + str(sum(prev_state.values())-sum(state.values())))
    return df, state

def criteria_emptyPatterns(df, set_h, prev_state):
    # Criteria: 'pattern' cannot be an empty value
    df = df[df['pattern'] != ""]
    state = get_current_state_df(set_h, df)
    print("\nEmpty Patterns: " + str(sum(prev_state.values())-sum(state.values())))
    return df, state

def criteria_quotes(df, set_h, prev_state):
    # Criteria: pattern can be encased in one pair of double quotes and no mismatched quotes
    df = df[~df['pattern'].str.count('"') % 2 == 1]
    state = get_current_state_df(set_h, df)
    print("\nMismatched Quotes: " + str(sum(prev_state.values())-sum(state.values())))
    return df, state

def criteria_inSet(df, set_h, prev_state):
    # Criteria: The tuple ('RequestType', 'RequestCategory', 'RequestSubCategory') should be present in set_h
    mask_tuple = df.apply(lambda row: (row['RequestType'], row['RequestCategory'], row['RequestSubCategory']) in set_h, axis=1)
    df = df[mask_tuple]
    state = get_current_state_df(set_h, df)
    print("\nNot In Set: " + str(sum(prev_state.values())-sum(state.values())))
    return df, state

def dataset_clean(set_h, csv_path = '/Users/sam.fleming/Desktop/IT Projects/STRC Capstone/Protoype 3.1/Data/GPTGeneratedUserInputs5.csv', 
                  extension = '_cleaned', new_file = False):
    '''
    Takes the absolute path of a heriachial dataset (RequestType, RequestCategory, RequestSubcategory, pattern), gets its existing state, 
    then checks each sample (line is the csv) that it is in it's absolute correct format:\n
        1. has the correct number (four) of elements\n
        2. hs no empty patterns\n
        3. has no mismathced stes of quotes\n
        4. label is in the set that has been specified\n
    If not, omits from the dataset and recreates a newer csv with all correct samples, and returns that csv's state as a comparison.\n
    Params:\n
        set_h: the exclusive breadth of categories for the specific dataset (given explicitly because it is possible they may be incorrect)\n
        csv_path: the absolute path for the csv\n
        extension: the suffix of the cleaned file name\n
        new_file: boolean, True = clean data should be writtent to a new file
    Returns:\n
        cleaned_data - the data inside csv_path but cleaned of all incorrect rows\n
    '''

    print('\nDataset: ' + csv_path)
    uncleaned_state = get_current_state(set_h, csv_path)
    num_samples = sum(uncleaned_state.values())
    print('Unlceaned State: ')
    print(uncleaned_state)
    print('Total Number of Samples: ' + str(num_samples))


    # Read the CSV into a DataFrame
    original_df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    df = original_df.copy()
    
    '''
    # Criteria 1: each line has four elements
    df = df[df.columns[df.columns.isin(['RequestType', 'RequestCategory', 'RequestSubCategory', 'pattern'])]]
    state1 = get_current_state_df(set_h, df)
    print("\nCriteria 1: " + str(sum(uncleaned_state.values())-sum(state1.values())))

    # Criteria 2: 'pattern' cannot be an empty value
    df = df[df['pattern'] != ""]
    state2 = get_current_state_df(set_h, df)
    print("\nCriteria 2: " + str(sum(state1.values())-sum(state2.values())))

    # Criteria 3: pattern can be encased in one pair of double quotes and no mismatched quotes
    df = df[~df['pattern'].str.count('"') % 2 == 1]
    state3 = get_current_state_df(set_h, df)
    print("\nCriteria 3:: " + str(sum(state2.values())-sum(state3.values())))

    # # Criteria 4: there should be no whitespace adjacent to commas
    # mask = df['pattern'].apply(lambda x: not bool(re.search(r'\s,', x) or re.search(r',\s', x)))
    # df = df[mask]
    # state4 = get_current_state_df(set_h, df)
    # print("\nCriteria 4: " + str(sum(state3.values())-sum(state4.values())))

    # Criteria 5: The tuple ('RequestType', 'RequestCategory', 'RequestSubCategory') should be present in set_h
    mask_tuple = df.apply(lambda row: (row['RequestType'], row['RequestCategory'], row['RequestSubCategory']) in set_h, axis=1)
    df = df[mask_tuple]
    state5 = get_current_state_df(set_h, df)
    print("\nCriteria 5: " + str(sum(state3.values())-sum(state5.values())))
    '''

    df, state = criteria_fourElements(df, set_h, uncleaned_state)
    df, state = criteria_emptyPatterns(df, set_h, state)
    df, state = criteria_quotes(df, set_h, state)
    df, state = criteria_inSet(df, set_h, state)

    cleaned_data = df.copy()

    
    if new_file:
        #optionally write the cleaned data as a new csv with variable 'extension' in its name
        base_dir = os.path.dirname(csv_path)
        file_name = os.path.basename(csv_path)
        file_base_name = os.path.splitext(file_name)[0]#getrfilename without the extension
        cleaned_file_name = file_base_name + extension + ".csv"
        cleaned_file_path = os.path.join(base_dir, cleaned_file_name)
    else:
        cleaned_file_path = csv_path
    
    cleaned_data.to_csv(cleaned_file_path, index=False)

    cleaned_state = get_current_state_df(set_h, cleaned_data)
    num_samples_cleaned = sum(cleaned_state.values())
    print('\nCleaned State: ')
    print(cleaned_state)
    print('Total Number of Samples: ' + str(num_samples_cleaned))
    print('Number of Samples Removed: '+ str(num_samples - num_samples_cleaned))

    return cleaned_data



def input_text(input_text, source_txt, key):
    '''
    Function that inputs a block of text 'input_text' into another block of text ('source_txt') 
    file at every place in the source where string 'key' is located, replacing 'key' and returning
    the augmented source text.
    params:
        input_text - the text to be inputted into the file
        txt_path - the absolute path of the .txt file
        key - the string that holds the location in 'txt_path' where 'input_text' will replace
    '''

    content_updated = source_txt.replace(key, input_text)
    return content_updated

def check_csv(csv_path):
    '''
    Check if the CSV file exists and is populated.
    '''
    if not os.path.exists(csv_path):
        return 0

    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row:  # Check if row is not empty
                return 1

    return 0

def set_descriptions(desc):
    output = []
    for row in desc:
        output.append(','.join(map(str, row)))
    return '\n'.join(output)


def get_current_state(set_h, csv_path):
    '''
    Count the current samples for each category in the dataset.\n
    Params:\n
        set_h: set of categories\n
        csv_path: path to the CSV file\n
    Returns:\n
        state: count of samples for each category\n
    '''
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        #if csv is empty (initial state), create empty dataframe but with the columns
        df = pd.DataFrame(columns=['RequestType', 'RequestCategory', 'RequestSubCategory'])
    except FileNotFoundError:
        # if the file does not exist, create an empty dataframe with the columns
        df = pd.DataFrame(columns=['RequestType', 'RequestCategory', 'RequestSubCategory'])
    
    #initialize an empty state dictionary
    state_dict = {}
    for category in set_h:
        # Extract the appropriate columns based on the hierarchy
        type_col, category_col, subcategory_col = category

        
        # Mask for each column
        mask_type = df['RequestType'] == type_col
        mask_category = df['RequestCategory'] == category_col
        mask_subcategory = (df['RequestSubCategory'] == subcategory_col) | (df['RequestSubCategory'].isna() & (subcategory_col == ''))

        # Combine all three masks using & operator
        combined_mask = mask_type & mask_category & mask_subcategory

        # # Debug prints
        # print(f"Filtering for {category}:")
        # print(f"- Entries matching RequestType: {df[mask_type].shape[0]}")
        # print(f"- Entries matching RequestCategory: {df[mask_category].shape[0]}")
        # print(f"- Entries matching RequestSubCategory: {df[mask_subcategory].shape[0]}")
        # print(f"- Entries matching all three conditions: {df[combined_mask].shape[0]}")


        # Count the number of rows for this category
        count = df[combined_mask].shape[0]
        state_dict[category] = count

    #convert the state dictionary to a set of tuples
    state = {(key, value) for key, value in state_dict.items()}
    
    return state_dict

def get_current_state_df(set_h, df):
    '''
    Count the current samples for each category in the dataset.\n
    Params:\n
        set_h: set of categories\n
        df: pandas DataFrame\n
    Returns:\n
        state: count of samples for each category\n
    '''
    # Check if the dataframe is empty
    if df.empty:
        # Create an empty dataframe with the columns
        df = pd.DataFrame(columns=['RequestType', 'RequestCategory', 'RequestSubCategory'])

    # Check if the required columns exist in the dataframe
    for column in ['RequestType', 'RequestCategory', 'RequestSubCategory']:
        if column not in df.columns:
            df[column] = ''

    # Initialize an empty state dictionary
    state_dict = {}
    for category in set_h:
        # Extract the appropriate columns based on the hierarchy
        type_col, category_col, subcategory_col = category

        # Mask for each column
        mask_type = df['RequestType'] == type_col
        mask_category = df['RequestCategory'] == category_col
        mask_subcategory = (df['RequestSubCategory'] == subcategory_col) | (df['RequestSubCategory'].isna() & (subcategory_col == ''))

        # Combine all three masks using & operator
        combined_mask = mask_type & mask_category & mask_subcategory

        # Count the number of rows for this category
        count = df[combined_mask].shape[0]
        state_dict[category] = count

    # Convert the state dictionary to a set of tuples
    state = {(key, value) for key, value in state_dict.items()}
    
    return state_dict

def get_remaining_counts(state, set_h_target):
    '''
    Calculate the difference between the target counts and current counts for each category.\n
    Params:\n
        state: current state of the dataset\n
        set_h_target: target count for each category\n
    Returns:\n
        remaining_counts: required samples to reach the target for each category
    '''
    #convert the sets to dictionaries for easier manipulation
    state_dict = dict(state)
    target_dict = dict(set_h_target)

    #calculate the remaining counts
    remaining_dict = {}
    for category in target_dict:
        remaining_dict[category] = target_dict[category] - state_dict.get(category, 0)

    #convert the remaining counts dictionary to a set of tuples
    remaining_counts = {(key, value) for key, value in remaining_dict.items()}
    
    return remaining_dict

def extract_label_set_from_csv(csv_path):
    """
    Extracts the unique set of hierarchical labels from the CSV file at the given path.\n
    Params:\n
        csv_path: Absolute path to the CSV containing data.\n
    Returns:\n
        A set of unique hierarchical labels, each represented as a tuple 
        (RequestType, RequestCategory, RequestSubCategory).
    """
    
    working_dir = os.path.dirname(os.path.abspath(__file__))

    data_dir = os.path.join(working_dir, '..', 'Data')
    data_dir = os.path.abspath(data_dir) #gets the correct data directory filepath from the filepath of the workingd directory

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)
    
    # Replace NaN values with empty strings
    df.fillna("", inplace=True)
    
    # Extract the unique tuples from the DataFrame
    labels_set = {(row['RequestType'], row['RequestCategory'], row['RequestSubCategory']) 
                  for _, row in df.iterrows()}
    
    heirachy = pd.read_csv(os.path.join(data_dir, 'ReqHeirachy.csv'))
    heirachy.fillna("", inplace=True)
    labels_set_checked = set()
    for idx, row in heirachy.iterrows():
        tup = (row['RequestType'], row['RequestCategory'], row['RequestSubCategory'])
        print(tup)
        if tup in labels_set:
            #insert into the set
            labels_set_checked.add(tup)

    
    return labels_set_checked