from Eigenfaces import *

# Some global variables and basic hyperparameter information are defined here
Path="./att_faces"

def myreconstruct(face,model,path):
    efaces = readInModel(model)                      # create the Eigenfaces object with the data dir
    faceDir=os.path.join(path,face)
    efaces.classify(faceDir,True)         # evaluate our model
if __name__ == "__main__":
    faceName=str(input("Please input the name(dir) of the picture that you want to classify\nFor example, you can input 's1/1.pgm' to import the picture:\n"))
    mytest(faceName,'model',Path)