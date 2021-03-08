
"""
@author: Tianxiao Yu, Zhounan Wang
"""

import pandas as pd
import numpy as np
from scipy import stats
import statistics 
from statistics import variance 
from scipy.stats import t
import researchpy
from bioinfokit.analys import stat

!pip.install('researchpy')

data = pd.read_csv("Prediction_new5w(KNN+countVec).csv")
data['timestamp'] = pd.to_datetime(data['timestamp'])
data['score'] = (data['Polarity']+1)/2
dataB = data.loc[data['Description_num'].isin(['1','2','5'])]
dataT = data.loc[data['Description_num'].isin(['3','4','6'])]

countB = dataB.groupby(['timestamp'])[['Description_num']].count()
score_sumB = dataB.groupby(['timestamp'])[['score']].sum()
joinB = countB.join(score_sumB)
joinB['Biden_score_1'] = joinB['score'].shift(1)
joinB['Biden_score_diff'] = joinB['score'] - joinB['Biden_score_1']

joinB['Biden_per'] = joinB['score']/joinB['Description_num']*100
joinB['Biden_percentage_1'] = joinB['Biden_per'].shift(1)
joinB['Biden_diff'] = joinB['Biden_per'] - joinB['Biden_percentage_1']

joinB['person'] = 'Biden'
joinB.rename(columns={'score':'Biden_score'}, inplace = True)

countT = dataT.groupby(['timestamp'])[['Description_num']].count()
score_sumT = dataT.groupby(['timestamp'])[['score']].sum()
joinT = countT.join(score_sumT)
joinT['Trump_score_1'] = joinT['score'].shift(1)
joinT['Trump_score_diff'] = joinT['score'] - joinT['Trump_score_1']

joinT['Trump_per'] = joinT['score']/joinT['Description_num']*100
joinT['Trump_percentage_1'] = joinT['Trump_per'].shift(1)
joinT['Trump_diff'] = joinT['Trump_per'] - joinT['Trump_percentage_1']

joinT['person'] = 'Trump'
joinT.rename(columns={'score':'Trump_score'}, inplace = True)

join_score = pd.concat([joinB.Biden_score, joinT.Trump_score], axis=1).dropna()
join_score_diff = pd.concat([joinB.Biden_score_diff, joinT.Trump_score_diff], axis=1).dropna()
join_per = pd.concat([joinB.Biden_per, joinT.Trump_per], axis=1).dropna()
join_diff = pd.concat([joinB.Biden_diff, joinT.Trump_diff], axis=1).dropna()

joinB

joinT

poll = pd.ExcelFile('Poll Result.xlsx')

df = pd.read_excel(poll, '538')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df['Trump_1'] = df['Trump'].shift(1)
df['Trump_diff538'] = df['Trump'] - df['Trump_1']
df['Biden_1'] = df['Biden'].shift(1)
df['Biden_diff538'] = df['Biden'] - df['Biden_1']
df_diff = df[['Biden_diff538' , 'Trump_diff538']].dropna()

df_RCP = pd.read_excel(poll, 'RealClearPolitics')
df_RCP['Date'] = pd.to_datetime(df_RCP['Date'])
df_RCP.set_index('Date', inplace=True)
df_RCP['Trump_1'] = df_RCP['Trump'].shift(1)
df_RCP['Trump_diffRCP'] = df_RCP['Trump'] - df_RCP['Trump_1']
df_RCP['Biden_1'] = df_RCP['Biden'].shift(1)
df_RCP['Biden_diffRCP'] = df_RCP['Biden'] - df_RCP['Biden_1']
df_RCP_diff = df_RCP[['Biden_diffRCP' , 'Trump_diffRCP']].dropna()

df_USC = pd.read_excel(poll, 'USC Dornsife_Los ')
df_USC['Date'] = pd.to_datetime(df_USC['Date'])
df_USC.set_index('Date', inplace=True)
df_USC['Trump_1'] = df_USC['Trump'].shift(1)
df_USC['Trump_diffUSC'] = df_USC['Trump'] - df_USC['Trump_1']
df_USC['Biden_1'] = df_USC['Biden'].shift(1)
df_USC['Biden_diffUSC'] = df_USC['Biden'] - df_USC['Biden_1']
df_USC_diff = df_USC[['Biden_diffUSC' , 'Trump_diffUSC']].dropna()

df_IBD = pd.read_excel(poll, 'IBD_TIPP')
df_IBD['Date'] = pd.to_datetime(df_IBD['Date'])
df_IBD.set_index('Date', inplace=True)
df_IBD['Trump_1'] = df_IBD['Trump'].shift(1)
df_IBD['Trump_diffIBD'] = df_IBD['Trump'] - df_IBD['Trump_1']
df_IBD['Biden_1'] = df_IBD['Biden'].shift(1)
df_IBD['Biden_diffIBD'] = df_IBD['Biden'] - df_IBD['Biden_1']
df_IBD_diff = df_IBD[['Biden_diffIBD' , 'Trump_diffIBD']].dropna()

df.plot()

df_diff.plot()

df_RCP.plot()

df_RCP_diff.plot()

df_USC.plot()

df_USC_diff.plot()

df_IBD.plot()

df_IBD_diff.plot()

diff_Biden538 = pd.concat([joinB.Biden_diff, df.Biden_diff538], axis=1).dropna()
diff_BidenRCP = pd.concat([joinB.Biden_diff, df_RCP.Biden_diffRCP], axis=1).dropna()
diff_BidenUSC = pd.concat([joinB.Biden_diff, df_USC.Biden_diffUSC], axis=1).dropna()
diff_BidenIBD = pd.concat([joinB.Biden_diff, df_IBD.Biden_diffIBD], axis=1).dropna()
diff_Trump538 = pd.concat([joinT.Trump_diff, df.Trump_diff538], axis=1).dropna()
diff_TrumpRCP = pd.concat([joinT.Trump_diff, df_RCP.Trump_diffRCP], axis=1).dropna()
diff_TrumpUSC = pd.concat([joinT.Trump_diff, df_USC.Trump_diffUSC], axis=1).dropna()
diff_TrumpIBD = pd.concat([joinT.Trump_diff, df_IBD.Trump_diffIBD], axis=1).dropna()

diff_Biden538 = diff_Biden538.rename(columns={"Biden_diff": "diff", "Biden_diff538": "poll"})
diff_BidenRCP = diff_BidenRCP.rename(columns={"Biden_diff": "diff", "Biden_diffRCP": "poll"})
diff_BidenUSC = diff_BidenUSC.rename(columns={"Biden_diff": "diff", "Biden_diffUSC": "poll"})
diff_BidenIBD = diff_BidenIBD.rename(columns={"Biden_diff": "diff", "Biden_diffIBD": "poll"})
diff_Trump538 = diff_Trump538.rename(columns={"Trump_diff": "diff", "Trump_diff538": "poll"})
diff_TrumpRCP = diff_TrumpRCP.rename(columns={"Trump_diff": "diff", "Trump_diffRCP": "poll"})
diff_TrumpUSC = diff_TrumpUSC.rename(columns={"Trump_diff": "diff", "Trump_diffUSC": "poll"})
diff_TrumpIBD = diff_TrumpIBD.rename(columns={"Trump_diff": "diff", "Trump_diffIBD": "poll"})

def cosine_similarity(x, y):
    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))

def euclidean_distance(x, y):   
    return np.sqrt(np.sum((x - y) ** 2))

print('Biden vs. 538: ', cosine_similarity(diff_Biden538['diff'], diff_Biden538['poll']))
print('Biden vs. RCP: ', cosine_similarity(diff_BidenRCP['diff'], diff_BidenRCP['poll']))
print('Biden vs. USC: ', cosine_similarity(diff_BidenUSC['diff'], diff_BidenUSC['poll']))
print('Biden vs. IBD: ', cosine_similarity(diff_BidenIBD['diff'], diff_BidenIBD['poll']))
print('Trump vs. 538: ', cosine_similarity(diff_Trump538['diff'], diff_Trump538['poll']))
print('Trump vs. RCP: ', cosine_similarity(diff_TrumpRCP['diff'], diff_TrumpRCP['poll']))
print('Trump vs. USC: ', cosine_similarity(diff_TrumpUSC['diff'], diff_TrumpUSC['poll']))
print('Trump vs. IBD: ', cosine_similarity(diff_TrumpIBD['diff'], diff_TrumpIBD['poll']))

print('Biden vs. 538: ', euclidean_distance(diff_Biden538['diff'], diff_Biden538['poll']))
print('Biden vs. RCP: ', euclidean_distance(diff_BidenRCP['diff'], diff_BidenRCP['poll']))
print('Biden vs. USC: ', euclidean_distance(diff_BidenUSC['diff'], diff_BidenUSC['poll']))
print('Biden vs. IBD: ', euclidean_distance(diff_BidenIBD['diff'], diff_BidenIBD['poll']))
print('Trump vs. 538: ', euclidean_distance(diff_Trump538['diff'], diff_Trump538['poll']))
print('Trump vs. RCP: ', euclidean_distance(diff_TrumpRCP['diff'], diff_TrumpRCP['poll']))
print('Trump vs. USC: ', euclidean_distance(diff_TrumpUSC['diff'], diff_TrumpUSC['poll']))
print('Trump vs. IBD: ', euclidean_distance(diff_TrumpIBD['diff'], diff_TrumpIBD['poll']))

def sign(df):
    des, res = researchpy.ttest(df['diff'], df['poll'])
    p = res.loc[res['Independent t-test']=='Two side test p value = '].iat[0, 1]
    if p < 0.05:
        print('The p value obtained from the t-test is significant(p < 0.05), and therefore, we conclude that there is a difference between the two variables.')
    else:
        print('The p value obtained from the t-test is not significant, and therefore, we conclude that There is no difference between the two variables.')
    print(researchpy.ttest(df['diff'], df['poll']))
    
def signi(df):
    res = stat()
    res.ttest(df=df, res=['diff', 'poll'], test_type=3)
    # output
    print(res.summary)

sign(diff_Biden538)
signi(diff_Biden538)
sign(diff_Trump538)
signi(diff_Trump538)


sign(diff_BidenRCP)
signi(diff_BidenRCP)
sign(diff_TrumpRCP)
signi(diff_TrumpRCP)


sign(diff_BidenUSC)
signi(diff_BidenUSC)
sign(diff_TrumpUSC)
signi(diff_TrumpUSC)


sign(diff_BidenIBD)
signi(diff_BidenIBD)
sign(diff_TrumpIBD)
signi(diff_TrumpIBD)











