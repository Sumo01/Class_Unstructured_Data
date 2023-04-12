import re
from bs4 import BeautifulSoup
import pandas as pd
import nltk
from nltk.corpus import stopwords
import distance 
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

import warnings
warnings.filterwarnings('ignore')

#preprocessing of the dataset
def preprocess(q):
    
    q = str(q).lower().strip()
    
    # Replace certain special characters with their string equivalents
    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')
    
    # The pattern '[math]' appears around 900 times in the whole dataset.
    q = q.replace('[math]', '')
    
    # Replacing some numbers with string equivalents (not perfect, can be done better to account for more cases)
    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)
    
    # Decontracting words
    # https://en.wikipedia.org/wiki/Wikipedia%3aList_of_English_contractions
    # https://stackoverflow.com/a/19794953
    contractions = { 
    "ain't": "am not",
    "aren't": "are not",
    "can't": "can not",
    "can't've": "can not have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
    }

    q_decontracted = []

    for word in q.split():
        if word in contractions:
            word = contractions[word]

        q_decontracted.append(word)

    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'ll", " will")
    
    # Removing HTML tags
    q = BeautifulSoup(q)
    q = q.get_text()
    
    # Remove punctuations
    pattern = re.compile('\W')
    q = re.sub(pattern, ' ', q).strip()

    
    return q

#function to get the token features    
def fetch_tokens_features(q1_tokens, q1_words, q1_stops,q2_tokens, q2_words, q2_stops):
    SAFE_DIV = 0.0001 
    token_features=[0.0]*8
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features
    common_word_count = len(q1_words.intersection(q2_words))
    
    # Get the common stopwords from Question pair
    common_stop_count = len(q1_stops.intersection(q2_stops))
    
    # Get the common Tokens from Question pair
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
    
    
    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    
    # Last word of both question is same or not
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    
    # First word of both question is same or not
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])
    
    return token_features

#function to get the length features
def fetch_length_features(q1,q2,q1_tokens,q2_tokens):  
    length_features = [0.0]*3
    
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return length_features
    
    # Absolute length features
    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))
    
    #Average Token Length of both Questions
    length_features[1] = (len(q1_tokens) + len(q2_tokens))/2
    
    strs = list(distance.lcsubstrings(q1, q2))
    length_features[2] = len(strs[0]) / (min(len(q1), len(q2)) + 1)
    
    return length_features

#function to get the fuzzy features 
def fetch_fuzzy_features(q1,q2):  
    fuzzy_features = [0.0]*4
    
    if q1 and q2:
        # fuzz_ratio
        fuzzy_features[0] = fuzz.QRatio(q1, q2)

        # fuzz_partial_ratio
        fuzzy_features[1] = fuzz.partial_ratio(q1, q2)

        # token_sort_ratio
        fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)

        # token_set_ratio
        fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)

    return fuzzy_features

#get the tokens,words, stops
def getting_tokens(row,val):
    q1 = row['question1']
    q2 = row['question2']
   

    STOP_WORDS = stopwords.words("english")
    
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    # Get the non-stopwords in Questions
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
    
    #Get the stopwords in Questions
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])

    if val==1:  
        return fetch_tokens_features(q1_tokens, q1_words, q1_stops,q2_tokens, q2_words, q2_stops)
    elif val==2:
        return fetch_length_features(q1,q2,q1_tokens,q2_tokens)
    elif val==3:
        return fetch_fuzzy_features(q1,q2)

#function to create the dataset (utilizes all above functions)
def creating_dataset(data_create):
    
    #getting token features
    token_features = data_create.apply(lambda row: getting_tokens(row,1), axis=1)
    data_create["cwc_min"]       = list(map(lambda x: x[0], token_features))
    data_create["cwc_max"]       = list(map(lambda x: x[1], token_features))
    data_create["csc_min"]       = list(map(lambda x: x[2], token_features))
    data_create["csc_max"]       = list(map(lambda x: x[3], token_features))
    data_create["ctc_min"]       = list(map(lambda x: x[4], token_features))
    data_create["ctc_max"]       = list(map(lambda x: x[5], token_features))
    data_create["last_word_eq"]  = list(map(lambda x: x[6], token_features))
    data_create["first_word_eq"] = list(map(lambda x: x[7], token_features))
    #print("token_features")
    
    #getting length features
    length_features = data_create.apply(lambda row: getting_tokens(row,2), axis=1)
    data_create['abs_len_diff'] = list(map(lambda x: x[0], length_features))
    data_create['mean_len'] = list(map(lambda x: x[1], length_features))
    data_create['longest_substr_ratio'] = list(map(lambda x: x[2], length_features))
    #print("length_features")
    
    #getting fuzzy features
    fuzzy_features = data_create.apply(lambda row: getting_tokens(row,3), axis=1)
    #print(fuzzy_features)
    # Creating new feature columns for fuzzy features
    data_create['fuzz_ratio'] = list(map(lambda x: x[0], fuzzy_features))
    data_create['fuzz_partial_ratio'] = list(map(lambda x: x[1], fuzzy_features))
    data_create['token_sort_ratio'] = list(map(lambda x: x[2], fuzzy_features))
    data_create['token_set_ratio'] = list(map(lambda x: x[3], fuzzy_features))
    #print(data_create.head())
    
    return data_create

def main_code(new_df):
    #preprocessing the dataset
    new_df['question1'] = new_df['question1'].apply(preprocess)
    new_df['question2'] = new_df['question2'].apply(preprocess)
    #print("new_df")
    
    #creating and processing the dataset
    df_temp=creating_dataset(new_df)
    #print(df_temp.head())
    #print("df_temp")
    
    #questions df
    ques_df=df_temp[['question1','question2']]
    final_df = df_temp.drop(columns=['question1','question2'])
    #print("final_df")
    # merge texts
    questions = list(ques_df['question1']) + list(ques_df['question2'])
    cv = CountVectorizer(max_features=3000)
    q1_arr, q2_arr = np.vsplit(cv.fit_transform(questions).toarray(),2)
    temp_df1 = pd.DataFrame(q1_arr, index= ques_df.index)
    temp_df2 = pd.DataFrame(q2_arr, index= ques_df.index)
    temp_df = pd.concat([temp_df1, temp_df2], axis=1)
    #print("temp_df")
    final_df = pd.concat([final_df, temp_df], axis=1)
    #print("final")
    return final_df

def model_creation(df):
    X_train,X_test,y_train,y_test = train_test_split(df.iloc[:,1:].values,df.iloc[:,0].values,test_size=0.2,random_state=1)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    rf = RandomForestClassifier()
    rf.fit(X_train,y_train)
    return rf

#run function
def run():
    df = pd.read_csv('train.csv')
    new_df = df.sample(30000,random_state=2)
    new_df = new_df.drop(columns=['id','qid1','qid2'])
    final_df=main_code(new_df)
    #print(final_df.head)
    rf=model_creation(final_df)
    #dumping the model for the predictor to use
    pickle.dump(rf, open("rf.pkl", "wb"))
    
    #print("Done")

run()