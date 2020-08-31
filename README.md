# student-cup-2020-solution

## チームslab

- [まっち～～](https://twitter.com/syu_0952)
- [9shiki](https://twitter.com/9shikiOldtype)

## 解法概要

- RoBERTa
- トピックモデル
    - トピックIDをRoBERTaへの入力に使用
- アンダーサンプリング
- 分類タスクとトピック確率予測のマルチタスク学習

## おまけ

訓練データとテストデータのカテゴリ割合の違いを，トピックモデルを通して推察できるかも
`./notebooks/TrainTestDist.ipynb`
