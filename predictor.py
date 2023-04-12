import pickle
import pandas as pd
import numpy as np
import train


def run(namecsv,ques):
    df=pd.read_csv(namecsv)
    df = df.drop(df.columns[0], axis=1)
    print("Dataset Loaded...")
    #global nqt,nqw,nqs
    df['question2'] = ques
    processed_df=train.main_code(df)
    #processed_df = processed_df.drop(processed_df.columns[0], axis=1)

    print("Processed...")
    model_df=processed_df.values
    
    #Load the model
    model = pickle.load(open("rf.pkl", "rb"))
    predict=model.predict(model_df)
    indices = np.where(predict == 1)[0]
    #print(indices)
    print("Model Loaded...")

    if len(indices)==0:
        return "NULL"
    else:
        return df.loc[df.index[indices],'question1']
    
name="quesdataset.csv"
print("Sent...")
similar=run(name,"What is the purpose of life?")

print(similar)