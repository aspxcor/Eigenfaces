from Eigenfaces import *

# Some global variables and basic hyperparameter information are defined here
Path="./att_faces"

def mytest(face,model,path):
    # if os.path.exists('results'):                             # create a folder where to store the results
    #     shutil.rmtree('results')                                 # clear everything in the results folder
    # os.makedirs('results')
    efaces = readInModel(model)                      # create the Eigenfaces object with the data dir
    faceDir=os.path.join(path,face)
    efaces.classify(faceDir,True)         # evaluate our model
if __name__ == "__main__":
    faceName=str(input("Please input the name(dir) of the picture that you want to classify\nFor example, you can input 's1/1.pgm' to import the picture:\n"))
    mytest(faceName,'model',Path)