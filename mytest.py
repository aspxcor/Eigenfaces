from Eigenfaces import *

# Some global variables and basic hyperparameter information are defined here
Path="./att_faces"

# [Function name] mytest
# [Function Usage] This function is used to test whether the model we trained can correctly recognize new pictures
# [Parameter]
    # face: Image path to be detected
    # model: The name of the model to be loaded
    # path: The storage path of the image to be tested
# [Return value] None
# [Developer and date] Zhi DING 2020/12/28
# [Change Record] None
def mytest(face,model,path):
    efaces = readInModel(model)                      # create the Eigenfaces object with the data dir
    faceDir=os.path.join(path,face)
    efaces.classify(faceDir,True)

if __name__ == "__main__":
    faceName=str(input("Please input the name(dir) of the picture that you want to classify\nFor example, you can input 's1/1.pgm' to import the picture:\n"))
    mytest(faceName,'model',Path)