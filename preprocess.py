import pandas as pd
import re

def convert_lower(df,col):
  df[col] = df[col].apply(lambda x: x.lower())
  return df
# removing special chars
def remove_special(df,col):
  df[col] = df[col].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
  return df
