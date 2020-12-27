from Eigenfaces import *

# Some global variables and basic hyperparameter information are defined here
Path="./att_faces"
Save_Path = "./Output"

def myreconstruct(face,model,path):
    efaces = readInModel(model)                      # create the Eigenfaces object with the data dir
    faceDir=os.path.join(path,face)
    numOfPCs=[10, 25, 50, 100, 150, 200]
    for numOfPC in numOfPCs:
        efaces.reconstruct(faceDir,numOfPC)         # evaluate our model
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(cv2.imread(Save_Path+"/Reconstruct_%s.jpg"%numOfPCs[i]), cmap='Greys_r')
    plt.show()
if __name__ == "__main__":
    faceName=str(input("Please input the name(dir) of the picture that you want to classify\nFor example, you can input 's1/1.pgm' to import the picture:\n"))
    myreconstruct(faceName,'model',Path)