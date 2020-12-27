from Eigenfaces import *

# Some global variables and basic hyperparameter information are defined here
Path="./att_faces"

def mytrain(energy,model,path):
    if os.path.exists('results'):                             # create a folder where to store the results
        shutil.rmtree('results')                                 # clear everything in the results folder
    os.makedirs('results')
    efaces = Eigenfaces(path,energy,model)                      # create the Eigenfaces object with the data dir
    efaces.write()                                                           # evaluate our model
if __name__ == "__main__":
    mytrain(1,'model',Path)