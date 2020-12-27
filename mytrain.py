from Eigenfaces import *

# Some global variables and basic hyperparameter information are defined here
Path="./att_faces"

def mytrain(energy,model,path):
    efaces = Eigenfaces(path,energy,model,True)                      # create the Eigenfaces object with the data dir
    efaces.write()                                              # write our model
if __name__ == "__main__":
    mytrain(1,'model',Path)