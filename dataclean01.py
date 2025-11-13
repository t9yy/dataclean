import pandas as pd
import numpy as np
import re


df = pd.read_csv(r'steps/step01/my_file (1).csv')
df.isna().sum()
null_percentage = (df.isnull().sum() / len(df)) * 100
df.columns = df.columns.str.replace(r'\xa0', ' ', regex=True)
df.columns = [re.sub(r'\s+', '_', col.strip()).lower() for col in df.columns]
df = df.rename(columns = {
    'adjusted_gross_(in_2022_dollars)':'adjusted_gross',
    'year(s)':'years',
    'ref.':'ref'
})
df = df.drop(['peak', 'all_time_peak'], axis=1)
df.iloc[7,0]=8

#非数字元素全部移除
cols = ['actual_gross', 'adjusted_gross', 'average_gross']
for col in cols:
    df[col] = df[col].str.replace(r'[^0-9]', '', regex=True)
print(df.head(20))

df['artist'] = df['artist'].str.replace(r'Beyoncé', 'Beyonce', regex=True)
df['tour_title'] = df['tour_title'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
df['tour_title'] = df['tour_title'].str.replace(r'[4a\21a]', '', regex=True).str.strip()
df['ref']=df['ref'].str.replace(r'[\[\]]', '', regex=True)
df['ref']=df['ref'].str.replace(r'd', '0', regex=True)
df.iloc[10,-1]=15 

#转换数据类型
cols = ['actual_gross', 'adjusted_gross', 'average_gross', 'ref']
df[cols]=df[cols].apply(pd.to_numeric, errors='coerce')
print(df.dtypes)

def parse_years(y):
    if pd.isna(y):
        return (np.nan, np.nan)
    y = str(y).strip()
    m = re.findall(r'(\d{4})', y)
    if len(m) == 0:
        return (np.nan, np.nan)
    if len(m) == 1:
        return (int(m[0]), int(m[0]))
    return (int(m[0]), int(m[1]))

df[['start_year', 'end_year']] = df['years'].apply(lambda x: pd.Series(parse_years(x)))
df = df.drop('years', axis=1)

#数据归一化
import pandas as pd
from sklearn.preprocessing import StandardScaler

num_cols = df.select_dtypes(include=['int64', 'float64']).columns

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

print(df.head(20))