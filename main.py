import predictor
import train
import datasetcreator

def createtesterdata(name):
    datasetcreator.run(name)

def train_model():
    train.run()

def predict(name,ques):
    predictor(name,ques)

print("Enter what you wish to do(train,predict,createdataset)")
name="questiondataset"
choice=input()
if choice=="train":
    train_model()
    print("Model Trained")
elif choice=="createdataset":
    createtesterdata(name)
    print("Dataset Created")
else:
    print("Enter the question: ")
    ques=input()
    print(predict(name,ques))

