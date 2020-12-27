from Eigenfaces import *
from scipy.interpolate import make_interp_spline

Path="./att_faces"
"""
Iteratively setting different energy threshold to be used for the Eigenfaces
and recording the accuracy in a data text file
"""
if __name__ == '__main__':
    energy_values = range(0,101)
    f = open('energy.dat', 'w')
    evalues_count=[]
    accuracy=[]
    for energy_value in energy_values:
        print('> Evaluating Energy:%s%%'%energy_value)
        efaces = Eigenfaces(Path, float(energy_value / 100.0))
        efaces.evaluate()
        f.write('%d %d %.6lf\n' % (energy_value, efaces.evalues_count, efaces.accuracy))
        if len(evalues_count) and evalues_count[len(evalues_count)-1]==efaces.evalues_count:
            accuracy[len(evalues_count)-1]=max(efaces.accuracy,accuracy[len(evalues_count)-1])
        else:
            evalues_count.append(efaces.evalues_count)
            accuracy.append(efaces.accuracy)
    evalues_count_smooth = np.linspace(np.array(evalues_count).min(), np.array(evalues_count).max(), 1000)
    accuracy_smooth = make_interp_spline(np.array(evalues_count), np.array(accuracy))(evalues_count_smooth)

    plt.plot(evalues_count_smooth, accuracy_smooth)
    plt.scatter(evalues_count, accuracy, marker='o')
    plt.xlabel('Number Of PCs')
    plt.ylabel('Rank-1 Rate/%')
    plt.xlim((0,205))
    plt.ylim((0,100))
    plt.xticks(np.arange(0,206,20))
    plt.yticks(np.arange(0,101,10))
    plt.show()

    f.close()
