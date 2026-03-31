from run_models import *
import config
import pandas as pd
import os

def combine_dfs_cumulative(dfs, start, end):
    selected = dfs[start:end]
    df_combine = pd.DataFrame(index=selected[0][selected[0].keys()[0]])
    cols=selected[0].keys()[1:]
    curr_col=0
    for col in cols:
        cumulative = [0] *len(df_combine.index)
        for i, df in enumerate(selected, start=0):
            cumulative = cumulative + np.asarray(df[col])
            df_combine[curr_col] = cumulative
            curr_col=curr_col+1
         
    return df_combine

def get_all_windows(window_size, dfs):
    rep_dict={}
    for window in range(0,22-window_size):
        rep_dict['len_'+str(window_size)+'_win_'+str(window)+'_'+str(window+window_size)]=combine_dfs_cumulative(dfs, window, window+window_size)
    return rep_dict

      
df_names=[f'zcurve_{str(i)}.csv' for i in range(0,21)]
dfs=[pd.read_csv(config.representation_dir+'/'+i) for i in df_names]
for window_size in [1,2,3,6,9,12,15,18,21]:
        reps= get_all_windows(window_size, dfs)
        for rep in reps:
            print(rep)
            print(reps[rep].head(2))
            reps[rep].to_csv(config.representation_dir+'/current_window.csv')
            print('Starting for: '+rep)
            rm=Run_Model(['current_window'],rep+'.json',huesken=True)
            rm.logo_run()