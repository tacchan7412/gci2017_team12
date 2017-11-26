import numpy as np
import pandas as pd
from scipy import stats

from fbprophet import Prophet
from sklearn.svm import SVR
from sklearn import linear_model
import xgboost as xgb

import sys,os
sys.path.append(os.pardir)
from DataReader import DataReader
dr = DataReader()
train_df, test_df = dr.get_raw_data()

def score(y_pred, y):
    abs_list = [abs(y_pred_ - y_) for (y_pred_, y_) in zip(y_pred, y)]
    return sum(abs_list) / len(abs_list)

df = train_df[["datetime", "y"]]
df["ds"] = pd.to_datetime(df.datetime)
df = df.drop("datetime", axis=1)
df = df.reset_index()

event_df_train = pd.DataFrame({
  'holiday': 'client_train',
  'ds': train_df[train_df.client == 1].datetime,
})
event_df_test = pd.DataFrame({
  'holiday': 'client_test',
  'ds': test_df[test_df.client == 1].datetime,
})

event_df = pd.concat((event_df_train, event_df_test))

def march_weekend(ds):
    date = pd.to_datetime(ds)
    if date.weekday() >= 5 and date.month == 3:
        return 1
    else:
        return 0

df['march_weekend'] = df['ds'].apply(march_weekend)

m = Prophet(weekly_seasonality=True, yearly_seasonality=True, holidays=event_df)
m.add_seasonality(name='monthly', period=30.5, fourier_order=5, prior_scale=10)
m.add_regressor('march_weekend', prior_scale=50)

m.fit(df)  # df is a pandas.DataFrame with 'y' and 'ds' columns
future = m.make_future_dataframe(periods=365)
future['march_weekend'] = future['ds'].apply(march_weekend)
forecast = m.predict(future)

def linear_regressor(train_idx, test_idx):
    train, test = dr.get_dummied_data()
    X = train.drop("y", axis=1).as_matrix()
    y = train.y.as_matrix()
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    print(regr.predict(X_test).shape)
    return regr.predict(X_test), regr.predict(test.as_matrix())

import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,BatchNormalization

def neural(train_idx, test_idx):
    train, test = dr.get_dummied_data()
    df = pd.concat([train,test])

    year_df= pd.get_dummies(df['year'], prefix='y', prefix_sep='_')
    #priceはどうか 
    price_am_df = pd.get_dummies(df['price_am'], prefix='PA', prefix_sep='_')
    price_pm_df = pd.get_dummies(df['price_pm'], prefix='PP', prefix_sep='_')
    df = pd.concat([df,
                   price_am_df,
                   price_pm_df,
                   year_df], axis=1)
    df = df.drop(["year","price_am","price_pm",'y_2017'],axis=1)
    train = df[0:2101]
    test = df[2101:]
    test["y_2016"] = int(1)
    test = test.drop(["y"],axis=1)
    # train, test = dr.get_dummied_data_with_year()
    X = train.drop("y", axis=1).as_matrix()
    y = train.y.as_matrix()
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    model = Sequential()
    # 73, 62
    model.add(Dense(140,input_dim=73,activation=keras.layers.advanced_activations.LeakyReLU(alpha=0.3)))
    model.add(Dense(70,input_dim=73,activation=keras.layers.advanced_activations.LeakyReLU(alpha=0.3)))
    model.add(Dense(1))
    opt = keras.optimizers.Adam(lr = 0.05,beta_1=0.9,beta_2=0.999,epsilon=1e-08,decay=0.0003)
    model.compile(optimizer=opt,
                  loss = "mean_absolute_error",
                  metrics=["mae"])
    model.fit(X_train,y_train,nb_epoch=5)
    return model.predict(X_test)[:,0], model.predict(test.as_matrix())[:,0]

def xg_boost(train_idx, test_idx):
    train, test = dr.get_data()
    X = train.drop("y", axis=1).as_matrix()
    y = train.y.as_matrix()
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    model = xgb.XGBRegressor(max_depth =30,n_estimators=500)
    model.fit(X_train,y_train)
    return model.predict(X_test), model.predict(test.as_matrix())

from sklearn.cross_validation import StratifiedKFold

n_folds = 10
skf = list(StratifiedKFold(train_df["y"], n_folds))
# regressors = [linear_regressor, xg_boost, neural]
regressors = [neural]
dataset_blend_train = np.zeros((len(train_df), len(regressors)+1))
dataset_blend_test = np.zeros((len(test_df), len(regressors)+1))

# stage 1
for j, regressor in enumerate(regressors):
    print('regressor: ', j)
    dataset_blend_test_j = np.zeros((len(test_df), len(skf)))
    for i, (train, test) in enumerate(skf):
        print('fold: ', i)
        (dataset_blend_train[test, j], dataset_blend_test_j[:, i]) = regressor(train, test)
    dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)

# prophet の予測データをマージ
dataset_blend_train[:, len(regressors)] = forecast["yhat"][:-365]
dataset_blend_test[:, len(regressors)] = forecast["yhat"][-365:]

# stage 2
model = xgb.XGBRegressor(max_depth =30,n_estimators=500)
model.fit(dataset_blend_train, train_df["y"])
y_submission = model.predict(dataset_blend_test)
y_pred = model.predict(dataset_blend_train)

print(score(y_pred, train_df["y"]))

ans = pd.DataFrame(y_submission, columns=['y'])
from DataWriter import DataWriter
dw = DataWriter()
dw.write_csv(ans)
