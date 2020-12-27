from Eigenfaces import *

# Some global variables and basic hyperparameter information are defined here
Path="./att_faces"

# [Function name] mytrain
# [Function Usage] This function is used to train the model according to the given energy value
# [Parameter]
    # energy: Energy value used to train the model
    # model: The name of the model to be loaded
    # path: The storage path of the image to be trained
# [Return value] None
# [Developer and date] Zhi DING 2020/12/28
# [Change Record] None
def mytrain(energy,model,path):
    efaces = Eigenfaces(path,energy,model,True)                # create the Eigenfaces object with the data dir
    efaces.write()                                              # write our model

if __name__ == "__main__":
    mytrain(1,'model',Path)