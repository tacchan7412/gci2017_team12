import pandas as pd
import numpy as np

class DataReader(object):
  def __init__(self):
    self.train_csv = 'train.csv'
    self.test_csv = 'test.csv'

  def concat_csv(self):
    # csvを読み込む
    train_df = pd.read_csv(self.train_csv)
    test_df = pd.read_csv(self.test_csv)
  
    # y column だけ切り取る
    train_df_y = train_df[["y"]]
    train_df = train_df.drop(["y"], axis=1)
    train_df_y.reset_index()
  
    df = pd.concat([train_df,test_df])

    return df, train_df_y

  def reconst_data(self, df, df_y):
    train_df = df[0:2101].reset_index()
    train_df = pd.concat([train_df, df_y],axis=1)
    train_df = train_df.drop("index",axis=1)
    test_df = df[2101:]

    return train_df, test_df

  def transform_datetime(self, df):
    df['datetime'] = pd.to_datetime(df.datetime)
    #何月か
    df['Month'] = df['datetime'].dt.month
    #何日か
    df['DayofMonth'] = df['datetime'].dt.day
    #何曜日か
    df['DayofWeek'] = df['datetime'].dt.dayofweek
    #何年か
    df['year'] = df['datetime'].dt.year

    df = df.drop(['datetime'], axis=1)
    
    return df

  def get_dummies_of_datetime(self, df):
    # ダミー変数化
    month_df = pd.get_dummies(df['Month'], prefix='M', prefix_sep='_')
    dayofmonth_df = pd.get_dummies(df['DayofMonth'], prefix='DM', prefix_sep='_')
    dayofweek_df = pd.get_dummies(df['DayofWeek'], prefix='DW', prefix_sep='_')

    df = pd.concat([df,month_df,dayofmonth_df, dayofweek_df], axis=1)

    df = df.drop(['Month', 'DayofMonth', 'DayofWeek'], axis=1)
    
    return df

  def get_data(self):
    df, train_df_y = self.concat_csv()
    df = self.transform_datetime(df)
    train_df, test_df = self.reconst_data(df, train_df_y)
    print(train_df.columns)
    
    return train_df, test_df

  def get_dummied_data(self):
    df, train_df_y = self.concat_csv()
    df = self.transform_datetime(df)
    df = self.get_dummies_of_datetime(df)
    train_df, test_df = self.reconst_data(df, train_df_y)
    print(train_df.columns)
    
    return train_df, test_df

