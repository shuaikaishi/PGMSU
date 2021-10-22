import numpy as np
import scipy.io as scio


def loadhsi(case):
    '''
    :input: case: for different datasets,
                 'toy' and 'usgs' are synthetic datasets
    :return: Y : HSI data of size [Bands,N]
             A_ture : Ground Truth of abundance map of size [P,N]
             P : nums of endmembers signature
    '''

    if case == 'ridge':
        file = './dataset/JasperRidge2_R198.mat'
        data = scio.loadmat(file)
        Y = data['Y']  # (C,w*h)
        Y = np.reshape(Y,[198,100,100])

        for i,y in enumerate(Y):
            Y[i]=y.T
        Y = np.reshape(Y, [198, 10000])
        GT_file = './dataset/JasperRidge2_end4.mat'
        A_true = scio.loadmat(GT_file)['A']
        A_true = np.reshape(A_true, (4, 100, 100))
        for i,A in enumerate(A_true):
            A_true[i]=A.T
        if np.max(Y) > 1:
            Y = Y / np.max(Y)



    elif case == 'urban':
        file = './dataset/Urban_R162.mat'
        data = scio.loadmat(file)
        Y = data['Y']  # (C,w*h)

        GT_file = './dataset/Urban_end4.mat'
        A_true = scio.loadmat(GT_file)['A']
        if np.max(Y) > 1:
            Y = Y / np.max(Y)

    elif case == 'ex2':
        file = './dataset/data_ex2.mat'
        data = scio.loadmat(file)
        Y = data['r']
        # Y = Y.T
        A_true = data['alphas']
        A_true=A_true.reshape(3,50,50)




    P = A_true.shape[0]

    Y = Y.astype(np.float32)
    A_true = A_true.astype(np.float32)
    return Y, A_true, P

if __name__=='__main__':
    cases = ['ex2','ridge', 'urban']
    case = cases[0]

    Y, A_true, P = loadhsi(case)
    Channel = Y.shape[0]
    N = Y.shape[1]
    print(case)
    print('nums of EM:',P)
    print('Channel :',Channel, ' pixels :',N)

    GT_file = './dataset/JasperRidge2_end4.mat'
    M = scio.loadmat(GT_file)['M']

    from matplotlib import pyplot as plt
    plt.plot(M)
    plt.show()