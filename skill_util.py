# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.7.1
# ---

import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import numpy as np
from sklearn.cluster import KMeans


def render_pca_plot(finalDf):
    fig = plt.figure(figsize = (12,12))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('Distribution Across Jobs', fontsize = 20)
    job_zones = finalDf['Job Zone'].unique()
    colors = ['r', 'g', 'b', 'gold', 'black']
    for job, color in zip( job_zones, colors):
        indicesToKeep = finalDf['Job Zone'] == job
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , s = 50,
                  c = color)
        
    finalDf= finalDf.reindex(finalDf['principal component 1'].abs().sort_values().index)
    largest_gap = finalDf.tail(20)
    
    for i, txt in enumerate(largest_gap['Title']):
        ax.annotate(txt, (largest_gap[largest_gap['Title']==txt]['principal component 1'], largest_gap[largest_gap['Title']==txt]['principal component 2']))

        
    ax.grid()
    ax.legend(job_zones)

def calc_rca(df, scale_id = 'IM'):
    df = df[df['Scale ID'] == scale_id]
    skills_df = df.groupby('O*NET-SOC Code')['Data Value'].sum().reset_index()
    jobs_df = df.groupby('Element Name')['Data Value'].sum().reset_index()
    all_val = df['Data Value'].sum()
    for job in tqdm_notebook(skills_df['O*NET-SOC Code'].unique().tolist()):
        for skill in jobs_df['Element Name'].unique().tolist():
            val = df[(df['Element Name'] == skill) & (df['O*NET-SOC Code']== job)]['Data Value'].values
            sum_val = skills_df[skills_df['O*NET-SOC Code'] == job]['Data Value'].values
            job_val = jobs_df[jobs_df['Element Name'] == skill]['Data Value'].values
            numerator = val/sum_val
            denominator = job_val / all_val
            rca = numerator/denominator
            df.loc[(df['Element Name'] == skill) & (df['O*NET-SOC Code']== job), 'rca'] = rca
    return df

def pca_df(df, scale='IM', num_dim = 2, value = 'rca'):
### matching each job to job zone.
    tmp = pd.merge(df, jobzones_23, on = ['O*NET-SOC Code'])
    #examine the level or importance
    temp = tmp[tmp['Scale ID']==scale]
    temp = temp.pivot_table(index = ['O*NET-SOC Code','Job Zone'], columns = 'Element Name', values = value).reset_index()
    columns = temp.columns.tolist()
    features =  [str(col) for col in columns if col not in ['Title','O*NET-SOC Code','Job Zone'] ]
    x = temp.loc[:, features].values
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=num_dim)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, temp[['O*NET-SOC Code','Job Zone']]], axis = 1)
    return finalDf

def mds(df, value = 'Data Value', n_dimension = 2):
    tmp = pd.merge(df, jobzones_23, on = ['O*NET-SOC Code'])
    #examine the level or importance
    temp = tmp[tmp['Scale ID']=='IM']
    temp = temp.pivot_table(index = ['O*NET-SOC Code','Job Zone'], columns = 'Element Name', values = value).reset_index()
    columns = temp.columns.tolist()
    features =  [str(col) for col in columns if col not in ['Title','O*NET-SOC Code','Job Zone'] ]
    x = temp.loc[:, features].values
    x = StandardScaler().fit_transform(x)
    #get the distance between jobs
    t = np.dot(x, np.transpose(x))
    mds = MDS(n_components= n_dimension, max_iter=3000, eps=1e-9, random_state=12345,
                       dissimilarity="precomputed", n_jobs=1)
    pos = mds.fit(t).embedding_
    # select the top 2 dimensions of data
    clf = PCA(n_components=2)
    pos = clf.fit_transform(pos)
    finalDf = pd.concat([pd.DataFrame(pos), temp[['O*NET-SOC Code','Job Zone']]], axis = 1)
    finalDf.rename(columns={0: 'PC1', 1: 'PC2'}, inplace=True)
    #plot the graphs
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111) 
    s = 100

    #plt.scatter(pos[:, 0], pos[:, 1], color='blue', s=s, lw=0)
    ax.scatter(finalDf['PC1'], finalDf['PC2'], color = 'blue', s= 50)
    finalDf= finalDf.reindex(finalDf.PC1.abs().sort_values().index)
    largest_gap = finalDf.tail(20)
    
    #for i, txt in enumerate(largest_gap['Title']):
    #   ax.annotate(txt, (largest_gap[largest_gap['Title']==txt]['PC1'], largest_gap[largest_gap['Title']==txt]['PC2']))

    plt.ylim(-8,8)
    plt.xlim(-30,30)
    plt.title('Polarization measured by Distance between Jobs')

    temp = tmp[tmp['Scale ID']=='IM']
    temp2 = temp.pivot_table(index = ['O*NET-SOC Code'], columns = 'Element Name', values = 'Data Value').reset_index()
    df = pd.merge(temp2, finalDf, on = ['O*NET-SOC Code'])
    df = df.corr()
    #df.rename(columns={0: 'principal component 1', 1: 'principal component 2'}, inplace=True)
    #df.columns = ['principal component 1', 'principal component 2','Title','O*NET-SOC Code','Job Zone' ]
    print df[['PC1']].sort_values('PC1', ascending = False).head(6)
    print df[['PC2']].sort_values('PC2', ascending = False).head(6)

def plot_important_skills(df, value, n = 10, low2high = False):
    df = df[df['Scale ID'] == 'IM']
    importance = df.groupby(['Element Name'])[value].median().sort_values(ascending = low2high).reset_index()
    if low2high == False:
        importance.head(n).plot.barh(x='Element Name', y=value, rot=1, title='Top {} Skills that increase in {}'.format(n, value))
    else:
        importance.head(n).plot.barh(x='Element Name', y=value, rot=1, title='Top {} Skills that decrease in {}'.format(n, value))



