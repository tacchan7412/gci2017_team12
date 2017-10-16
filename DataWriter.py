import pandas as pd
import numpy as np
import datetime
from DataReader import DataReader

class DataWriter(object):
  def __init__(self):
    now = datetime.datetime.now()
    self.submission_csv = '../data/submission_' + now.strftime('%Y%m%d%H%M%S') + '.csv'

  def write_csv(self, df):
    dr = DataReader()
    _, test_df = dr.get_raw_data()
    submission_df = pd.concat([df.y, test_df], axis=1)
    submission_df['y'] = submission_df['y'] * (1 - submission_df['close']) 
    last_df = pd.concat([submission_df['datetime'], submission_df['y']], axis=1)
    last_df.to_csv(self.submission_csv, index=False, header=False)
    print(self.submission_csv + 'がdataフォルダに追加されました')
