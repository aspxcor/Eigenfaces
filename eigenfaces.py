# Homework 4 Name: Eigenface
# Program description: Write code to implement the training, recognition and reconstruction process of Eigenface face recognition:
    # 1. Assuming that there is only one face in each face image, and the positions of the two eyes are known (that is, they can be manually labeled). For example: the eye position of each image is stored in a text file with the same name as the image file but with the suffix txt under the corresponding directory. The text file is represented by a line of 4 numbers separated by spaces, corresponding to two eyes respectively The position of the center in the image;
    # 2. Realize three program processes, corresponding to training, recognition and reconstruction respectively
    # 3. Build a face database by yourself (at least 40 people, must contain their own face images), the course provides an AT&T face database to choose from.
    # 4. The functions related to Eigenface in OpenCV cannot be called directly, and the SDK can be called to solve the eigenvalue and eigenvector
    # 5. The training process is roughly: mytrain energy percentage model file name other parameters …)”…)”, use energy percentage to determine how many eigenfaces are taken, and save the training result output to the model file. The demo program combines the first 10 characteristic faces into an image at the same time, and then displays it.
    # 6. The recognition process is roughly as follows: mytest input face image file name model file name other parameters After the model file is loaded in, the input face image is recognized. The demonstration program superimposes the recognition result on the input face image and displays it, and at the same time displays the image most similar to the face image in the face library.
    # 7. Reconstruction process: roughly as follows: myreconstruct input face image file name model file name and other parameters, after loading the model file, transform the input face image into eigenface space, and then use the transformed result to rebuild Construct back to the original face image. The demo program can display the results of reconstruction with 10 PCs, 25 PCs, 50 PCs, and 100 PCs at the same time, and realize the reconstruction of own images.
    # 8. The experimental report must include the following experimental results: a. Average face and at least the top 10 eigenfaces; b. 10 PCs, 25 PCs, 50 PCs, and 100 PCs reconstructed from your own face image Results; c. Half of the data of each person in the AT&T face database is trained, and the other half is tested, showing the change curve of Rank 1 recognition rate as the PC increases (that is, the abscissa is the number of PCs used, and the ordinate is Rank 1 a curve of rate)
# File Name: Eigenfaces.py

import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Some global variables and basic hyperparameter information are defined here
Save_Path = "./Output"     # Path to save picture which is after detected

# [Class name] Eigenfaces
# [Class Usage] A Python class that implements the Eigenfaces algorithm for face recognition, using eigenvalue decomposition and principle component analysis. We use the AT&T data set, with 50% of the images as train and the rest 50% as a test set. Additionally, we use a small set of celebrity images to find the best AT&T matches to them.
# [Class Interface]
    # write(self): Write data into the file
    # classify(self, pathToImg,isShow=False):Classify an image to one of the eigenfaces.
    # reconstruct(self, pathToImg,numOfPC): Reconstruct the image by using the eigenfaces.
    # evaluate(self):Evaluate the model using the 50% test faces left from every different face in the AT&T set.
# [Developer and date] Anonymous
# [Change Record] None
class Eigenfaces(object):
    facesCount = 41        #40 faces from AT&T dataset and the last face is my face data
    trainFacesCount = 5                                                   # number of faces used for training
    testFacesCount = 5                                                    # number of faces used for testing
    totalNumOfTrain = trainFacesCount * facesCount                                     # training images count
    m = 92                                                                  # number of columns of the image
    n = 112                                                                 # number of rows of the image
    mn = m * n                                                              # length of the column vector

    # [Function name] __init__
    # [Function Usage] This function is used to Initialize the Eigenfaces class
    # [Parameter]
        # _facesDir: The path of the face dataset
        # _energy: Percentage of energy during training of Eigenface algorithm
        # _model: Location of model output
        # isShow: Whether to show the top ten Eigenfaces
    # [Return value] None
    # [Developer and date] Anonymous
    # [Change Record] None
    def __init__(self, _facesDir = '.', _energy = 0.8,_model='trainModel',isShow=False):
        print ('> Eigenfaces initializing ...')
        self.facesDir = _facesDir
        self.energy = _energy
        self.model=_model
        self.trainingIDs = []                                      # train image id's for every at&t face
        L = np.empty(shape=(self.mn, self.totalNumOfTrain), dtype='float64')      # each row of L represents one train image
        curImg = 0
        for faceID in range(1, self.facesCount + 1):
            trainingIDs = random.sample(range(1, 11), self.trainFacesCount)  # the id's of the 5 random training images
            self.trainingIDs.append(trainingIDs)                   # remembering the training id's for later
            for trainingID in trainingIDs:
                pathToImg = os.path.join(self.facesDir,
                        's' + str(faceID), str(trainingID) + '.pgm')          # relative path
                img = cv2.imread(pathToImg, 0)                                # read a grayscale image
                imgCol = np.array(img, dtype='float64').flatten()              # flatten the 2d image into 1d
                L[:, curImg] = imgCol[:]         # set the curImg-th column to the current training image
                curImg += 1
        self.meanImgCol = np.sum(L, axis=1) / self.totalNumOfTrain     # get the mean of all images / over the rows of L
        cv2.imwrite("mean.jpg",self.meanImgCol.reshape(self.n,self.m))
        for j in range(0, self.totalNumOfTrain):                       # subtract from all training images
            L[:, j] -= self.meanImgCol[:]
        # Instead of computing the covariance matrix as L*L^T, we set C = L^T*L, and end up with way smaller and computentionally inexpensive one. we also need to divide by the number of training images
        C = np.matrix(L.transpose()) * np.matrix(L)
        C /= self.totalNumOfTrain
        # Calculate the eigenvectors and eigenvalues of AT *A, evalues are eigenvalues, evectors are eigenvectors
        self.evalues, self.evectors = np.linalg.eig(C)
        sortIndices = self.evalues.argsort()[::-1]         # getting their correct order - decreasing
        self.evalues = self.evalues[sortIndices]           # puttin the evalues in that order
        self.evectors = self.evectors[:,sortIndices]       # same for the evectors
        evaluesSum = sum(self.evalues[:])                  # include only the first k evectors/values so
        self.evaluesCount = 0                                   # that they include approx. 85% of the energy
        evaluesEnergy = 0.0
        for evalue in self.evalues:
            self.evaluesCount += 1
            evaluesEnergy += evalue / evaluesSum
            if evaluesEnergy >= self.energy:
                break
        self.evalues = self.evalues[0:self.evaluesCount]# reduce the number of eigenvectors/values to consider
        self.evectors = self.evectors[:,0:self.evaluesCount]
        self.evectors = L * self.evectors               # left multiply to get the correct evectors
        path=os.path.join(Save_Path,str(int(100*self.energy)))
        if not os.path.isdir(path):
            os.makedirs(path)
        for i in range(self.evectors.shape[1]):
            cv2.imwrite(path+"/Eigenface%s.jpg"%(i+1),self.evectors[:,i].reshape(self.n,self.m))
        image=[]
        if isShow:
            try:
                for i in range(10):
                    image.append(plt.imread(path+"/Eigenface%s.jpg"%(i+1)))
                    plt.subplot(2, 5, i+1)
                    plt.imshow(image[i],cmap='Greys_r')        #
                plt.show()
            except:
                pass
        norms = np.linalg.norm(self.evectors, axis=0)         # find the norm of each eigenvector
        self.evectors = self.evectors / norms                 # normalize all eigenvectors
        self.W = self.evectors.transpose() * L                # computing the weights
        print ('> Eigenfaces initializing ended')

    # [Function name] write
    # [Function Usage] This function is used to Write data into the file
    # [Parameter] None
    # [Return value] None
    # [Developer and date] Anonymous
    # [Change Record] None
    def write(self):
        fileOut=open(self.model,"wb")
        pickle.dump(self,fileOut,0)
        fileOut.close()

    # [Function name] classify
    # [Function Usage] This function is used to Classify an image to one of the eigenfaces.
    # [Parameter]
        # pathToImg: The path of the face to be classified
        # isShow: Whether to show the Match result
    # [Return value]
        # matchID: Matched face ID
        # closestFaceID: The closest matching face dataset ID for training
    # [Developer and date] Anonymous
    # [Change Record] None
    def classify(self, pathToImg,isShow=False):
        img = cv2.imread(pathToImg, 0)                                        # read as a grayscale image
        imgCol = np.array(img, dtype='float64').flatten()                      # flatten the image
        imgCol -= self.meanImgCol                                            # subract the mean column
        imgCol = np.reshape(imgCol, (self.mn, 1))                             # from row vector to col vector
        S = self.evectors.transpose() * imgCol  # projecting the normalized probe onto the Eigenspace, to find out the weights
        diff = self.W - S                                                       # finding the min ||W_j - S||
        norms = np.linalg.norm(diff, axis=0)
        closestFaceID = np.argmin(norms)               # the id of the minerror face to the sample
        matchID=int(closestFaceID / self.trainFacesCount) + 1
        if isShow:
            closestFace = os.path.join(self.facesDir,
                                       's' + str(matchID), str(self.trainingIDs[matchID-1][closestFaceID-(matchID-1)*self.trainFacesCount]) + '.pgm')
            closestFaceImg = cv2.imread(closestFace, 0)  # read a grayscale image
            closestFaceImgResize=cv2.resize(closestFaceImg, (5 * img.shape[1], 5 * img.shape[0]))
            cv2.imshow('ClosestFaceID:%s' % str(closestFaceID+1), closestFaceImgResize)
            imgResize = cv2.resize(img, (5 * img.shape[1], 5 * img.shape[0]))
            cv2.putText(imgResize, 'Match:%s' % matchID, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow('Match:%s' % matchID, imgResize)
            cv2.waitKey()
            print("Match ID:%s" % matchID)
            print("Closest Face ID:%s" % str(closestFaceID+1))
        return matchID,closestFaceID

    # [Function name] reconstruct
    # [Function Usage] This function is used to reconstruct the image by using the eigenfaces.
    # [Parameter]
        # pathToImg: The path of the face to be reconstructed
        # numOfPC: number of PCs to reconstruct the faces
    # [Return value] None
    # [Developer and date] Anonymous
    # [Change Record] None
    def reconstruct(self, pathToImg,numOfPC):
        img = cv2.imread(pathToImg, 0)                                        # read as a grayscale image
        imgCol = np.array(img, dtype='float64').flatten()                      # flatten the image
        imgCol -= self.meanImgCol                                            # subract the mean column
        imgCol = np.reshape(imgCol, (self.mn, 1))                             # from row vector to col vector
        output=self.meanImgCol.reshape(self.n,self.m)
        for i in range(numOfPC):
            weight=self.evectors[:,i].reshape(1,self.mn).dot(imgCol)
            output+=self.evectors[:,i].reshape(self.n,self.m)*int(weight)
        cv2.imwrite(Save_Path+"/Reconstruct_%s.jpg"%numOfPC,output)

    # [Function name] evaluate
    # [Function Usage] This function is used to Evaluate the model using the 50% test faces left from every different face in the AT&T set.
    # [Parameter] None
    # [Return value] None
    # [Developer and date] Anonymous
    # [Change Record] None
    def evaluate(self):
        print ('> Evaluating Models started')
        results_file = os.path.join(Save_Path,'results_%s.txt'%str(int(100*self.energy)))               # filename for writing the evaluating results in
        f = open(results_file, 'w')                                             # the actual file
        test_count = self.testFacesCount * self.facesCount       # number of all AT&T test images/faces
        test_correct = 0
        for faceID in range(1, self.facesCount + 1):
            for test_id in range(1, 11):
                if (test_id in self.trainingIDs[faceID-1]) == False:          # we skip the image if it is part of the training set
                    pathToImg = os.path.join(self.facesDir,
                            's' + str(faceID), str(test_id) + '.pgm')          # relative path
                    result_id,_ = self.classify(pathToImg)
                    result = (result_id == faceID)
                    if result == True:
                        test_correct += 1
                        f.write('Image: %s\nResult: Correct\n\n' % pathToImg)
                    else:
                        f.write('Image: %s\nResult: Wrong. The correct answer should be %2d while got %2d\n\n' %(pathToImg, faceID, result_id))
        print ('> Evaluating Models ended')
        self.accuracy = float(100. * test_correct / test_count)
        print ('Correct: ' + str(self.accuracy) + '%')
        f.write('Correct: %.2f\n' % (self.accuracy))
        f.close()                                                               # closing the file

# [Function name] readInModel
# [Function Usage] This function is used to read In the Model
# [Parameter] model:the model to be read in.
# [Return value] modelIn: The Eigenfaces model.
# [Developer and date] Anonymous
# [Change Record] None
def readInModel(model):
    fileOut = open(model, "rb")
    modelIn = pickle.load(fileOut)
    fileOut.close()
    return modelIn