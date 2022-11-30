import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

def read_mat_file(name_file:str='linreg.mat') -> tuple[np.ndarray]:
    import scipy.io
    mat = scipy.io.loadmat(name_file)
    return np.reshape(mat['x'], (-1,)), np.reshape(mat['y'], (-1,))

def sto(a:np.ndarray, k:float) -> np.ndarray:
    a[abs(a) <= k] = 0
    a[a > k] = a[a > k] - k
    a[a < -k] = a[a < -k] + k
    return a

def lav(t, y, iter_max:int=500, epsilon:float=10e-5, r:int=100) -> tuple[np.ndarray]:

    iter = 0
    err = 1
    usr = 1/r # Un sur r

    A = np.transpose(np.array([[1 for _ in t],t]))
    B = np.transpose(A) @ A

    b = np.reshape(y, (-1,1))

    x, xp = np.zeros((2,1)), np.zeros((2,1))
    u, up = np.zeros(b.shape), np.zeros(b.shape)
    z, zp = np.zeros(b.shape), np.zeros(b.shape)

    while iter <= iter_max and err > epsilon:
        x = np.linalg.solve(B, np.transpose(A) @ (b + z - u))
        z = sto(A @ x - b + u, usr)
        u = u + A @ x - z - b
        err = (norm(x-xp) + norm(u-up) + norm(z-zp))/(norm(x) + norm(u) + norm(z))
        xp, zp, up = x, z, u
        iter += 1

    return x, A, b, iter

def main() -> None:

    ### Définir t et y sous forme de liste ou de array 1D ###
    # Ajouter ou commenter et décommenter les valeurs pour changer

    # t = [i/10 for i in range(11)]
    # y = [2.06, 2.12, 2.32, 2.02, 2.76, 3.04, 2.83, 3.15, 3.36, 3.68, 3.96]

    t, y = read_mat_file()

    ### ADMM ###
    x, A, b, iter = lav(t, y)

    ### Moindres carrés classiques ###
    lstsq = np.linalg.lstsq(A, b, rcond=None)[0]
    
    ### Affiche les résultats
    plt.plot(t, y, linestyle='', marker='.')
    plt.plot(t,A @ x, label='ADMM')
    plt.plot(t,A @ lstsq, label='LSTSQ')
    plt.legend()
    plt.grid()
    plt.title(f'En {iter} itérations')
    plt.show()

if __name__ == '__main__':
    main()