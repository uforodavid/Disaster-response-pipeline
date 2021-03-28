# import libraries

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# load messages dataset
def etl_pipeline(messages, categories):
    messages = pd.read_csv('disaster_messages.csv')
    # load categories dataset
    categories = pd.read_csv('disaster_categories.csv')
    # merge datasets
    df = pd.merge(messages, categories, how='inner', on='id')
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', n=-1, expand=True)
    headers = categories.iloc[0]
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
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
    # drop duplicates
    df = df.drop_duplicates()
    #load dataframe to SQL database
    engine = create_engine('sqlite:///DisasterResponses.db')
    return df.to_sql('table_name', engine, index=False)
