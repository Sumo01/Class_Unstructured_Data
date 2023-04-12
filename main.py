print("Sample Question: ")
import predictor
import train
import datasetcreator

def createtesterdata(name):
    datasetcreator.run(name)

def train_model():
    train.run()

def predict(name,ques):
    predictor(name,ques)
choice=''
while(choice!="exit"):
    print("Enter what you wish to do(train,predict,createdataset,exit)")
    name="quesdataset.csv"
    choice=input()
    if choice=="train":
        train_model()
        print("Model Trained")
    elif choice=="createdataset":
        createtesterdata(name)
        print("Dataset Created")
    elif choice=="predict":
        print("Enter the question: ")
        ques=input()
        print("Question Asked: ",ques)
        result=predictor.run(name,ques)
        print("Existing Questions: ")
        print(result)
    
