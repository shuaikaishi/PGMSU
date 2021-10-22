import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from loadhsi import loadhsi
from model.model import PGMSU
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cases = ['ex2', 'ridge']
case = cases[1]

model_weights = './PGMSU_weight/'
model_weights += 'PGMSU.pt'

output_path = './PGMSU_out/'
Y, A_true, P = loadhsi(case)

if case == 'ridge':
    nCol = 100
    nRow = 100
elif case == 'urban':
    nCol = 307
    nRow = 307
elif case == 'ex2':
    nCol = 50
    nRow = 50

nband = Y.shape[0]
N = Y.shape[1]
Channel = Y.shape[0]
z_dim = 4
##########################################################
# plot 1: trainging loss
loss = scio.loadmat(output_path + 'loss.mat')['loss']
plt.loglog(loss[0])
# plt.savefig(output_path+'loss.png')
plt.show()
##########################################################
EM_hat = scio.loadmat(output_path + 'results.mat')['EM']
A_hat = scio.loadmat(output_path + 'results.mat')['A']
Y_hat = scio.loadmat(output_path + 'results.mat')['Y_hat']


A_hat = np.reshape(A_hat, (nRow, nCol, P))
B = np.zeros((P, nRow, nCol))
for i in range(P):
    B[i] = A_hat[:, :, i]
A_hat = B
A_true = A_true.reshape([P, -1])
A_hat = A_hat.reshape([P, -1])

A_true = A_true.reshape([P, nCol, nRow])
A_hat = A_hat.reshape([P, nCol, nRow])

# plot 2 : Abundance maps
fig = plt.figure()
for i in range(1, P + 1):
    plt.subplot(2, P, i + P)
    aaa = plt.imshow(A_true[i - 1], cmap='jet', interpolation='none')
    # plt.axis('off')

    aaa.set_clim(vmin=0, vmax=1)
    plt.subplot(2, P, i)
    aaa = plt.imshow(A_hat[i - 1], cmap='jet',
                     interpolation='none')
    aaa.set_clim(vmin=0, vmax=1)
    # plt.axis('off')
plt.subplot(2, P, 1)
plt.ylabel('PGMSU')
plt.subplot(2, P, 1 + P)
plt.ylabel('reference GT')
plt.show()

plt.figure()
for i in range(P):
    plt.subplot(2, (P + 1) // 2, i + 1)
    plt.plot(EM_hat[0:EM_hat.shape[0]: 100, i, :].T, 'c', linewidth=0.5)
    plt.xlabel('$\it{Bands}$', fontdict={'fontsize': 16})
    plt.ylabel('$\it{Reflectance}$', fontdict={'fontsize': 16})
    plt.axis([0, len(EM_hat[0, i, :]), 0, 1])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # plt.axis('off')

armse = np.mean(np.sqrt(np.mean((A_hat - A_true) ** 2, axis=0)))
print('aRMSE: ', armse)



from sklearn.decomposition import PCA

pca = PCA(2)
y2d = pca.fit_transform(Y.T)
plt.figure()
plt.scatter(y2d[:, 0], y2d[:, 1], 5, label='Pixel data')
P = EM_hat.shape[1]

for i in range(P):
    em2d = pca.transform(np.squeeze(EM_hat[:, i, :]))
    plt.scatter(em2d[:, 0], em2d[:, 1], 5, label='EM #' + str(i + 1))

plt.legend()
plt.title('Scatter plot of mixed pixels and EMs')
# plt.savefig('em2d')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()




if case =='ridge':
    # generate new EMs
    model = PGMSU(P, Channel, z_dim).to(device)
    model.load_state_dict(torch.load(model_weights))
    model.eval()

    data = scio.loadmat('./dataset/' + 'road4.mat')['road4']
    data = data.T.astype(np.float32)
    data = torch.from_numpy(data)

    num_pixels = 4
    repeat = 800
    Z = []
    for i in range(repeat):
        with torch.no_grad():
            y_hat, mu, log_var, a, em_tensor = model(data.to(device))
            z = model.reparameterize(mu, log_var)
        Z.append(z.cpu().numpy())
    Z = np.concatenate(Z, axis=0)
    # print(np.max(Z), np.min(Z))
    scio.savemat(output_path+'./Z.mat', {'z2d': Z})
    ################################
    # generate new endmembers
    EM_interp = []
    z_interp_ = []
    for t in np.arange(0, 8):
        z_interp_.append(t / 8 * mu[2, :] + (1 - t) / 8 * mu[3, :])
    for t in np.arange(0, 4):
        z_interp_.append(t / 4 * mu[0, :] + (1 - t) / 4 * mu[2, :])

    for t in np.arange(0, 8):
        z_interp_.append(t / 8 * mu[1, :] + (1 - t) / 8 * mu[0, :])

    for z_interp in z_interp_:
        with torch.no_grad():
            em_interp = model.decoder(z_interp.reshape(1, -1))
            em_interp = torch.reshape(em_interp, [-1, P, Channel])
            EM_interp.append(em_interp.cpu().numpy())
    EM_interp = np.concatenate(EM_interp, axis=0)

    scio.savemat(output_path+'./EM_interp.mat', {'EM_interp': EM_interp})
    ################################
    from sklearn.decomposition import PCA
    pca = PCA(2)
    z2d = pca.fit_transform(Z)
    mu = pca.transform(mu.cpu().numpy())
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(4):
        plt.scatter(z2d[i:repeat * num_pixels:num_pixels, 0], z2d[i:repeat * num_pixels:num_pixels, 1], 10)
    ax.arrow(mu[3, 0], mu[3, 1], mu[0, 0] - mu[3, 0], mu[0, 1] - mu[3, 1],
             width=0.1,
             length_includes_head=True,
             head_width=0.25,
             head_length=1,
             fc='k',
             ec='k')
    plt.show()

    scio.savemat(output_path+'./z2d.mat', {'z2d': z2d})


    plt.plot(data.numpy().T)
    plt.grid('on')
    plt.legend(['pixel 1', 'pixel 2', 'pixel 3', 'pixel 4'])
    plt.show()
