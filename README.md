# gci2017_team12

## ipython notebookの共有方法
- `notebook` フォルダに各自`.ipynb`ファイルを追加する
- その際ファイル名の最後に`_名字`を追加する
  - ex: `sample_koga.ipynb`

## Dataの取得方法
- 以下を実行する
   ```
   from DataReader import DataReader
   dr = DataReader()
   train_df, test_df = dr.get_dummied_data()
   ```

`train_df`と`test_df`はpandas.DataFrameで返ってくる
`dr.get_dummied_data()`の代わりに`dr.get_data()`をすることでダミー変数化されていないデータ、`dr.get_raw_data()`でcsvを読み込んだだけのデータ、を取得できる(使い方は全く同じ)

## 外部データの共有方法
- `data`フォルダにcsvファイルを追加する
- PRの段階でどのようなデータかを説明する
  - その後READMEに古賀が説明をコピペして誰でもすぐ見れるようにする
- データは、`train.csv`と同じ形式の年月日のcolumnと一つ以上の外部データのcolumnsが含まれていることを前提とする
