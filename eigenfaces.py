import os
import cv2
import sys
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Some global variables and basic hyperparameter information are defined here
Save_Path = "./Output"     # Path to save picture which is after detected
# Path="./att_faces"

"""
A Python class that implements the Eigenfaces algorithm
for face recognition, using eigenvalue decomposition and
principle component analysis.

We use the AT&T data set, with 60% of the images as train
and the rest 40% as a test set, including 85% of the energy.

Additionally, we use a small set of celebrity images to
find the best AT&T matches to them.

Example Call:
    $> python eigenfaces.py att_faces celebrity_faces

Algorithm Reference:
    http://docs.opencv.org/modules/contrib/doc/facerec/facerec_tutorial.html
"""
class Eigenfaces(object):
    faces_count = 41
    faces_dir = '.'                                                         # directory path to the AT&T faces
    train_faces_count = 5                                                   # number of faces used for training
    test_faces_count = 5                                                    # number of faces used for testing
    l = train_faces_count * faces_count                                     # training images count
    m = 92                                                                  # number of columns of the image
    n = 112                                                                 # number of rows of the image
    mn = m * n                                                              # length of the column vector
    """
    Initializing the Eigenfaces model.
    """
    def __init__(self, _faces_dir = '.', _energy = 0.85,_model='trainModel',isShow=False):
        print ('> Initializing started')
        self.faces_dir = _faces_dir
        self.energy = _energy
        self.model=_model
        self.training_ids = []                                          # train image id's for every at&t face
        L = np.empty(shape=(self.mn, self.l), dtype='float64')      # each row of L represents one train image
        cur_img = 0
        for face_id in range(1, self.faces_count + 1):
            training_ids = random.sample(range(1, 11), self.train_faces_count)  # the id's of the 6 random training images
            self.training_ids.append(training_ids)                   # remembering the training id's for later
            for training_id in training_ids:
                path_to_img = os.path.join(self.faces_dir,
                        's' + str(face_id), str(training_id) + '.pgm')          # relative path
                img = cv2.imread(path_to_img, 0)                                # read a grayscale image
                img_col = np.array(img, dtype='float64').flatten()              # flatten the 2d image into 1d
                L[:, cur_img] = img_col[:]         # set the cur_img-th column to the current training image
                cur_img += 1
        self.mean_img_col = np.sum(L, axis=1) / self.l     # get the mean of all images / over the rows of L
        cv2.imwrite("mean.jpg",self.mean_img_col.reshape(self.n,self.m))
        for j in range(0, self.l):                                         # subtract from all training images
            L[:, j] -= self.mean_img_col[:]
        C = np.matrix(L.transpose()) * np.matrix(L)             # instead of computing the covariance matrix as
        C /= self.l                                         # L*L^T, we set C = L^T*L, and end up with way
                                                            # smaller and computentionally inexpensive one
                                                            # we also need to divide by the number of training
                                                            # images
        # 计算AT *A的 特征向量和特征值evalues是特征值，evectors是特征向量
        self.evalues, self.evectors = np.linalg.eig(C)      # eigenvectors/values of the covariance matrix
        sort_indices = self.evalues.argsort()[::-1]         # getting their correct order - decreasing
        self.evalues = self.evalues[sort_indices]           # puttin the evalues in that order
        self.evectors = self.evectors[:,sort_indices]       # same for the evectors
        evalues_sum = sum(self.evalues[:])                  # include only the first k evectors/values so
        self.evalues_count = 0                                   # that they include approx. 85% of the energy
        evalues_energy = 0.0
        for evalue in self.evalues:
            self.evalues_count += 1
            evalues_energy += evalue / evalues_sum
            if evalues_energy >= self.energy:
                break
        self.evalues = self.evalues[0:self.evalues_count]# reduce the number of eigenvectors/values to consider
        self.evectors = self.evectors[:,0:self.evalues_count]
        #self.evectors = self.evectors.transpose()      # change eigenvectors from rows to columns (Should not transpose)
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
        print ('> Initializing ended')
    """
    Write data into the file
    """
    def write(self):
        file_out=open(self.model,"wb")
        pickle.dump(self,file_out,0)
        file_out.close()
    """
    Classify an image to one of the eigenfaces.
    """
    def classify(self, path_to_img,isShow=False):
        img = cv2.imread(path_to_img, 0)                                        # read as a grayscale image
        img_col = np.array(img, dtype='float64').flatten()                      # flatten the image
        img_col -= self.mean_img_col                                            # subract the mean column
        img_col = np.reshape(img_col, (self.mn, 1))                             # from row vector to col vector
        S = self.evectors.transpose() * img_col                 # projecting the normalized probe onto the
                                                                # Eigenspace, to find out the weights
        diff = self.W - S                                                       # finding the min ||W_j - S||
        norms = np.linalg.norm(diff, axis=0)
        closest_face_id = np.argmin(norms)               # the id [0..240) of the minerror face to the sample
        matchID=int(closest_face_id / self.train_faces_count) + 1
        if isShow:
            closest_face = os.path.join(self.faces_dir,
                                       's' + str(matchID), str(self.training_ids[matchID-1][closest_face_id-(matchID-1)*self.train_faces_count]) + '.pgm')  # relative path
            closest_face_img = cv2.imread(closest_face, 0)  # read a grayscale image
            closest_face_img_resize=cv2.resize(closest_face_img, (5 * img.shape[1], 5 * img.shape[0]))
            cv2.imshow('ClosestFaceID:%s' % str(closest_face_id+1), closest_face_img_resize)
            imgResize = cv2.resize(img, (5 * img.shape[1], 5 * img.shape[0]))
            cv2.putText(imgResize, 'Match:%s' % matchID, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow('Match:%s' % matchID, imgResize)
            cv2.waitKey()
            print("Match ID:%s" % matchID)
            print("Closest Face ID:%s" % str(closest_face_id+1))
        return matchID,closest_face_id                   # return the faceid (1..40)
    """
    Evaluate the model using the 4 test faces left
    from every different face in the AT&T set.
    """
    def evaluate(self):
        print ('> Evaluating AT&T faces started')
        results_file = os.path.join('results', 'att_results.txt')               # filename for writing the evaluating results in
        f = open(results_file, 'w')                                             # the actual file
        test_count = self.test_faces_count * self.faces_count       # number of all AT&T test images/faces
        test_correct = 0
        for face_id in range(1, self.faces_count + 1):
            for test_id in range(1, 11):
                if (test_id in self.training_ids[face_id-1]) == False:          # we skip the image if it is part of the training set
                    path_to_img = os.path.join(self.faces_dir,
                            's' + str(face_id), str(test_id) + '.pgm')          # relative path
                    result_id,_ = self.classify(path_to_img)
                    result = (result_id == face_id)
                    if result == True:
                        test_correct += 1
                        f.write('image: %s\nresult: correct\n\n' % path_to_img)
                    else:
                        f.write('image: %s\nresult: wrong, got %2d\n\n' %
                                (path_to_img, result_id))

        print ('> Evaluating AT&T faces ended')
        self.accuracy = float(100. * test_correct / test_count)
        print ('Correct: ' + str(self.accuracy) + '%')
        f.write('Correct: %.2f\n' % (self.accuracy))
        f.close()                                                               # closing the file

def readInModel(model):
    file_out = open(model, "rb")
    modelIn = pickle.load(file_out)
    file_out.close()
    return modelIn