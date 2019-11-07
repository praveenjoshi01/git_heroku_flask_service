import pandas as pd
df = pd.read_table('debug_log.log',header =0)
df.columns = ['log']
df['ErrorCode'] = int(1)
df.to_csv('dataset.csv',index= False)
print('voila')