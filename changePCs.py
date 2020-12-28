from Eigenfaces import *
from scipy.interpolate import make_interp_spline

# Some global variables and basic hyperparameter information are defined here
Path="./Dataset"

# [Function name] changePCs
# [Function Usage] This function is used to test and plot the change in recognition accuracy as the number of PCs increases
# [Parameter] None
# [Return value] None
# [Developer and date] Zhi DING 2020/12/28
# [Change Record] None
if __name__ == '__main__':
    energyValues = range(0,101)
    f = open('energyPCsAccuracy.dat', 'w')
    evaluesCount=[]
    accuracy=[]
    for energyValue in energyValues:
        efaces = Eigenfaces(Path, float(energyValue / 100.0))
        efaces.evaluate()
        print('> Evaluating Energy:%s%%. Number of PCs:%s' % (energyValue,efaces.evaluesCount))
        f.write('%d %d %.6lf\n' % (energyValue, efaces.evaluesCount, efaces.accuracy))
        if len(evaluesCount) and evaluesCount[len(evaluesCount)-1]==efaces.evaluesCount:
            accuracy[len(evaluesCount)-1]=max(efaces.accuracy,accuracy[len(evaluesCount)-1])
        else:
            evaluesCount.append(efaces.evaluesCount)
            accuracy.append(efaces.accuracy)
    evaluesCountSmooth = np.linspace(np.array(evaluesCount).min(), np.array(evaluesCount).max(), 1000)
    accuracySmooth = make_interp_spline(np.array(evaluesCount), np.array(accuracy))(evaluesCountSmooth)
    plt.plot(evaluesCountSmooth, accuracySmooth)
    plt.scatter(evaluesCount, accuracy, marker='o')
    plt.xlabel('Number Of PCs')
    plt.ylabel('Rank-1 Rate/%')
    plt.xlim((0,205))
    plt.ylim((0,100))
    plt.xticks(np.arange(0,206,20))
    plt.yticks(np.arange(0,101,10))
    plt.show()
    f.close()