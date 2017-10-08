# gci2017_team12

## Dataの取得方法
1. train.csv, test.csv を https://deepanalytics.jp/compe/47/download からダウンロードし、gci2017_team12 の直下に追加する
2. 以下を実行する
   ```
   from DataReader import DataReader
   dr = DataReader()
   train_df, test_df = dr.get_dummied_data()
   ```

`train_df`と`test_df`はpandas.DataFrameで返ってくる
`dr.get_dummied_data()`の代わりに`dr.get_data()`をすることでダミー変数化されていないデータ、`dr.get_raw_data`でcsvを読み込んだだけのデータ、を取得できる(使い方は全く同じ)
