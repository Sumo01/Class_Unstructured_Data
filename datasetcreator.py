import pandas as pd

import warnings
warnings.filterwarnings('ignore')

def add_ques(pred_data,q):
    #print(q)
    pred_data=pred_data.append({'question1':q,},ignore_index=True)
    return pred_data

def create_csv(pred_data,name):
    pred_data.to_csv(name)
    
def run(name):
    df=pd.read_csv('train.csv')
    df = df.iloc[:3000]

    #global pred_data
    pred_data = pd.DataFrame(columns=['question1'])
    for index,row in df.iterrows():
       pred_data=add_ques(pred_data,row['question1'])
       pred_data=add_ques(pred_data,row['question2'])

    #pred_data = pred_data.drop(pred_data.columns[0], axis=1)
    #print(pred_data.head())
    create_csv(pred_data,name)

name="quesdataset.csv"
run(name)

