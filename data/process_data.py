import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    This function returns merged dataframe of input dataframes.

    Parameters:
        messages_filepath (str):The messages data filepath
        categories_filepath (str): The categories data filepath

    Returns:
        Dataframe: Input files are loaded and merged to form a single data frame.  
    '''
    #load in data set
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(messages, categories, how='inner', on='id')

    return df

def clean_data(df):
    '''
    This function takes in a dataframe and cleans it and return a new dataframe
    Parameters
    ----------
        df: pandas.DataFrame
            The dataFrame containing merged data to be cleaned
    Returns
    -------
        df: pandas.DataFrame
            A new dataFrame containing cleaned data
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', n=-1, expand=True)
    headers = categories.iloc[0]
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x:x[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    # convert column from string to numeric
        categories[column] = categories[column].astype(np.int)
    # drop the original categories column from `df`

    
    df = df.drop(labels='categories', axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis= 1)
    # converting categories to binary
    df.replace({'related':2}, 1, inplace=True)
    # drop duplicates
    df = df.drop_duplicates()
    return df

def save_data(df, database_filename):
    """Saves DataFrame (df) to database path"""
    name = 'sqlite:///' + database_filename
    engine = create_engine(name)
    df.to_sql('Disasters', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()