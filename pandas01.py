import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("import success")

s = pd.Series([1, 3, 5, np.nan, 6, 8])
print(s)

val1 = 17
dates = pd.date_range('20170801', periods=val1)
print(dates)
df = pd.DataFrame(
    np.random.randn(val1, 4), index=dates,
    columns=list('ABCD'))  #需要和 columns 匹配上
print(df)

df2 = pd.DataFrame({
    'A': 1.,
    'B': pd.Timestamp('20170801'),
    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
    'D': np.array([3] * 4, dtype='int32'),
    'E': pd.Categorical(["test", "train", "test", "train"]),
    'F': 'foo'
})
print(df2)
print(df2.dtypes)

print("head\n", df.head(6))
print("tail\n", df.tail(3))
print("index\n", df.index)
print("columns\n", df.columns)
print("values\n", df.values)

print("describe\n", df.describe())
print("T\n", df.T)
print("sort_index\n", df.sort_index(axis=1, ascending=False))
print("sort\n", df.sort_values('B'))

print(df['A'])
print(df[0:3])
print(df['20170803':'20170810'])

print(df.loc[dates[0]])
print(df.loc[:, ['A', 'B']])
print(df.loc['20170803':'20170810', ['A', 'B']])
print(df.loc['20170810', ['A', 'B']])
print(df.loc[dates[0], 'A'])
print(df.at[dates[0], 'A'])

print(df.iloc[3])
print(df.iloc[3:5, 0:2])
print(df.iloc[[1, 2, 4], [0, 2]])
print(df.iloc[1:3, :])
print(df.iloc[:, 1:3])
print(df.iloc[1, 1])

print(df[df.A > 0])
print(df[df > 0])
df2 = df.copy()
df2['E'] = pd.date_range('20220801', periods=val1)
print(df2[df2['E'].isin(['20220802', '20220811'])])

s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20170823', periods=6))

df1 = df.reindex(index=dates[0:10], columns=list(df.columns[0:2]) + ['E'])
df1.loc[dates[0]:dates[5], 'E'] = 1
print(df1)

print(df1.dropna(how='any'))
print(df1.fillna(value=6))
print(pd.isnull(df1))

df1 = df1.reindex(index=df1.index, columns=list(df1.columns) + ['T', 'K', 99])
print(df1)

print(df.mean())
print(df.mean(1))

s = pd.Series([1, 3, 5, np.nan, 6, 8], index=[0, 1, 2, 3, 4, 5]).shift(2)
print(s)
print(df.sub(s, axis='index'))

print(df.apply(np.cumsum))
print(df.apply(lambda x: x.max() - x.min()))

s = pd.Series(np.random.randint(0, 7, size=10))
print(s.value_counts())

s = pd.Series(['A', 'b', 'abc', 'dd'])
print(s.str.lower())

df = pd.DataFrame(np.random.randn(10, 4))
pieces = [df[:3], df[3:7], df[7:]]
print(pd.concat(pieces))

left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})
print(pd.merge(left, right, on='key'))

df = pd.DataFrame(np.random.randn(8, 4), columns=['A', 'B', 'C', 'D'])
s = df.iloc[3]
print(df.append(s, ignore_index=True))

df = pd.DataFrame({
    'A': ['fo', 'ba', 'fo', 'ba', 'fo', 'ba', 'fo', 'ba'],
    'B': [1, 2, 3, 1, 2, 3, 1, 2],
    'C': np.random.randn(8),
    'D': np.random.randn(8)
})
print(df.groupby('A').sum())
print(df.groupby(['A', 'B']).sum())


ts=pd.Series(np.random.randn(1000),index=pd.date_range('1/1/2000',periods=1000))
ts=ts.cumsum()
ts.plot(kind='line')
