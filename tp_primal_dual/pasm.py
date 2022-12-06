import numpy as np
from random import random
from copy import deepcopy
import unittest as ut

def pasm(Q: np.ndarray, c: np.ndarray, Ai: np.ndarray, bi: np.ndarray, maxit: int, eps: float, x0: np.ndarray, verbose:int=2) -> tuple[np.ndarray, int, np.ndarray, int]:

    if type(Ai) != np.ndarray:
        raise TypeError("Ai is supposed to be numpy array")
    if type(Q) != np.ndarray:
        raise TypeError("Q is supposed to be numpy array")
    
    primal_to_dual = False
    coef = 1
    flag = 1 # Le nombre d'iter atteint maxit

    if Ai.shape != Q.shape or (Ai != np.eye(Q.shape[0])).all():
        if verbose == 2 : print("Switch from Primal to Dual")
        prim_Q, prim_c = deepcopy(Q), deepcopy(c)
        primal_to_dual = True
        coef = -1
        # x0 = np.linalg.solve(Ai @ Ai.T, Ai @ (-c - Q @ x0))
        Q, c = Ai @ np.linalg.inv(Q) @ Ai.T, -np.transpose(np.transpose(c) @ np.linalg.inv(Q) @ Ai.T)
        bi = np.zeros((Q.shape[0],1))
        x0 = np.zeros((Q.shape[0],1))

    x = x0
    mu = np.zeros((Q.shape[0], 1))
    iter = 0

    while iter < maxit:

        ens_Ak = [i for i in range(x.shape[0]) if coef*x[i] > bi[i] or (x[i] == bi[i] and mu[i] >= 0)]
        ens_Ik = [i for i in range(x.shape[0]) if coef*x[i] < bi[i] or (x[i] == bi[i] and mu[i] < 0)]

        x[ens_Ak] = bi[ens_Ak]

        x[ens_Ik] = np.linalg.solve(Q[ens_Ik][:, ens_Ik], -c[ens_Ik] - Q[ens_Ik][:, ens_Ak] @ bi[ens_Ak])
        mu[ens_Ak] = -c[ens_Ak] - Q[ens_Ak][:] @ x

        iter += 1

        if (x <= bi).all() and (mu[ens_Ak] >= 0).all():
            flag = 0
            break

    if primal_to_dual:
        mu = x
        x = np.linalg.solve(prim_Q, -prim_c - Ai.T @ x)

    if verbose in [1,2]:
        print(iter)
        print(x)

    return x, flag, mu, iter

def premiere_condition(Q:np.ndarray, x:np.ndarray, Ai:np.ndarray, mu:np.ndarray, c:np.ndarray):
    return Q @ x + Ai.T @ mu + c

class VerificationOptimilate(ut.TestCase):

    def test_pb_2(self):
        #--------------------------------------------------------- Définition variables
        itermax = 10
        Q = np.array([[2.3546, 1.6361, 1.8427, 2.1537],
                    [1.6361, 1.6617, 1.5320, 1.4873],
                    [1.8427, 1.5320, 2.4314, 2.2958],
                    [2.1537, 1.4873, 2.2958, 2.8471]])
        c = np.array([[-0.7583],
                    [-0.2899],
                    [-1.0962],
                    [-1.2270]])
        Ai = np.array([[0.202700, 0.272100, 0.746700, 0.465900],
                    [0.198700, 0.198800, 0.445000, 0.418600],
                    [0.603700, 0.015200, 0.931800, 0.846200]])
        bi = np.array([[0.5251],
                    [0.2026],
                    [0.6721]])
        x = np.zeros((Q.shape[0],1))

        #--------------------------------------------------------- Résolution
        x, flag, mu, iter = pasm(Q, c, Ai, bi, itermax, 10e-6, x, verbose=0)

        #--------------------------------------------------------- Vérification
        np.testing.assert_almost_equal(np.zeros((Q.shape[0],1)), premiere_condition(Q, x, Ai, mu, c))
        for index, value in enumerate(Ai @ x):
            self.assertLessEqual(value[0], bi[index][0])

    def test_pb_3(self):
        #--------------------------------------------------------- Définition variables
        itermax = 100
        Q = np.array([
            [14, 2, 6, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 6, 2],
            [2, 14, 2, 6, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 6],
            [6, 2, 14, 2, 6, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
            [0, 3, 6, 2, 14, 2, 6, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [3, 6, 2, 14, 2, 6, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 3, 6, 2, 14, 2, 6, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 3, 6, 2, 14, 2, 6, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 3, 6, 2, 14, 2, 6, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 3, 6, 2, 14, 2, 6, 3, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 3, 6, 2, 14, 2, 6, 3, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 3, 6, 2, 14, 2, 6, 3, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 3, 6, 2, 14, 2, 6, 3, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 6, 2, 14, 2, 6, 3, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 6, 2, 14, 2, 6, 3, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 6, 2, 14, 2, 6, 3, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 6, 2, 14, 2, 6, 3, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 6, 2, 14, 2, 6, 3],
            [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 6, 2, 14, 2, 6],
            [6, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 6, 2, 14, 2],
            [2, 6, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 6, 2, 14]])
        c = np.reshape(np.array([-i for i in range(1, 21)]), (-1, 1))
        Ai = np.ones((1,Q.shape[0]))
        bi = np.array([[0]])
        x = np.zeros((Q.shape[0],1))

        #--------------------------------------------------------- Résolution
        x, flag, mu, iter = pasm(Q, c, Ai, bi, itermax, 10e-6, x, verbose=0)

        #--------------------------------------------------------- Vérification
        np.testing.assert_almost_equal(np.zeros((Q.shape[0],1)), premiere_condition(Q, x, Ai, mu, c))
        for index, value in enumerate(Ai @ x):
            self.assertLessEqual(value[0], bi[index][0])

    def test_pb_4(self):
        #--------------------------------------------------------- Définition variables
        itermax = 10
        Q = np.array([
            [2., -1., 0., 0., 0.],
            [-1., 2., -1., 0., 0.],
            [0., -1., 2., -1., 0.],
            [0., 0., -1., 2., -1.],
            [0., 0., 0., -1., 2.]])
        c = np.array([
            [-1.],
            [0.],
            [0.],
            [0.],
            [-1.]
        ])
        bi = np.array([
            [1/2.],
            [0.],
            [0.],
            [0.],
            [1/2.]
        ])
        x = np.array([
            [-1.],
            [-1.],
            [-1.],
            [-1.],
            [-1.]
        ])
        Ai = np.eye(Q.shape[0])

        #--------------------------------------------------------- Résolution
        x, flag, mu, iter = pasm(Q, c, Ai, bi, itermax, 10e-5, x, verbose=0)

        #--------------------------------------------------------- Vérification
        np.testing.assert_almost_equal(np.zeros((Q.shape[0],1)), premiere_condition(Q, x, Ai, mu, c))
        for index, value in enumerate(Ai @ x):
            self.assertLessEqual(value[0], bi[index][0])

if __name__ == '__main__':
    ut.main()
