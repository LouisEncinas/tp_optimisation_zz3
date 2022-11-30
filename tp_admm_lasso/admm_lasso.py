import numpy as np
from numpy.linalg import norm

from scipy import sparse
import scipy.sparse.linalg as splinalg

import matplotlib.pyplot as plt

def read_mat_file(name_file:str='data.mat'):

    """
    Fonction permettant de lire un fichier .mat avec le format donné par le code generator.m
    -> Changer m et n pour changer de taille
    """

    import scipy.io
    mat = scipy.io.loadmat(name_file)
    return mat['A'], mat['b'], mat['x0'], mat['lambda_max'][0,0], mat['lambda'][0,0]

def sto(a:np.ndarray, k:float) -> np.ndarray:
    a[abs(a) <= k] = 0
    a[a > k] = a[a > k] - k
    a[a < -k] = a[a < -k] + k
    return a

def lav(A, b, x0, z0, u0, lam, r,epsilon:float=10e-6, iter_max:int=1000) -> tuple[np.ndarray]:

    #-------------------------- Initialisation
    iter = 0
    err = 1
    err_diff = 1
    usr = lam/r # Un sur r
    cost = 0
    lst_cost = [] # liste contenant les couts à chaque iter pour affichage

    x = x0
    u = u0
    z = z0

    B = np.transpose(A) @ A
    eye = sparse.eye(B.shape[0])

    #-------------------------- Décomposition LU
    lu = splinalg.splu(B + r*eye)
    # Il n'existe pas de fonction prédifini dans numpy et scipy pour faire une
    # décomposition de Cholesky


    while iter <= iter_max and (err > epsilon or err_diff > 10e-4):

        #-------------------------- Affichage iter et fonction cout
        print(f'itération {iter} : {cost}\n')

        #-------------------------- Resolve sans décomposition LU
        # x = np.reshape(np.linalg.solve(B + r*eye, np.transpose(A) @ b + r*(z - u)),(-1,1))

        #-------------------------- Resolve avec décomposition LU
        x = np.reshape(lu.solve(np.transpose(A) @ b + r*(z - u)),(-1,1))
        z = sto(x + u, usr)
        u = u + x - z

        #-------------------------- Calcul des erreurs
        err = (norm(x-x0) + norm(z-z0))/(norm(x) + norm(z))
        err_diff = norm(x0-z0)

        #-------------------------- Iter et fonction cout
        x0, z0, u0 = x, z, u
        iter += 1
        cost = 0.5*norm(A @ x - b)**2 + lam*norm(x,ord=1)

        lst_cost.append(cost)

    return x, iter, lst_cost

def main() -> None:

    #-------------------------- Lecture depuis fichier .mat (cf. fonction read_mat_file)
    A, b, x0, lam_max, lam = read_mat_file()

    z0 = np.zeros(x0.shape)
    u0 = np.zeros(x0.shape)
    r = 20 # r = 0.9 donne le moins d'iter pour le fichier data.mat

    #-------------------------- Appel fonction lav
    x, iter, lst = lav(A, b, x0, z0, u0, lam, r)

    #-------------------------- Affichage courbe des couts
    plt.plot(lst)
    plt.show()

if __name__ == '__main__':
    main()