import numpy as np
import unittest as ut

VAR = 10

def laplacien(fk, lambdak, hk) -> np.ndarray:
    lapl = fk
    for index in range(len(hk)): lapl += lambdak[index,0] * hk[index]
    return lapl

def pqsl(x0:np.ndarray, lambda0:np.ndarray, f, h, gradf, gradh, grad2f, grad2h, ineq:list=[], epsilon:float=10e-6, maxit:int=100):

    #------------------------------------------------------------ Initialisation
    iter = 0
    nb_comp = len(x0)
    xk, lambdak = x0, lambda0

    while iter < maxit:
        hk, gradfk, gradhk, grad2fk, grad2hk = h(xk), gradf(xk), gradh(xk), grad2f(xk), grad2h(xk)

        #------------------------------------------------------------ Correction de la var d'écart z (Marche pas)
        for index in range(len(xk)):
            if xk[index,0] < 0 and index in ineq: hk[index] -= xk[index,0]
        nb_cond = len(hk)

        #------------------------------------------------------------ étape 1
        grad2Lapl = laplacien(grad2fk, lambdak, grad2hk)

        #------------------------------------------------------------ étape 2
        Dhk = np.concatenate(gradhk, axis=1)
        umatrix = np.concatenate((grad2Lapl, Dhk), axis=1)
        dmatrix = np.concatenate((Dhk.T, np.zeros((nb_cond,nb_cond))), axis=1)
        lmatrix = np.concatenate((umatrix, dmatrix), axis=0)
        rmatrix = -np.concatenate((gradfk, np.array([[el] for el in hk])), axis=0)
        sol = np.linalg.solve(lmatrix, rmatrix)

        xk = xk + sol[0:nb_comp,:]
        lambdak = sol[nb_comp:,:]

        #------------------------------------------------------------ vérification
        gradLapl = laplacien(gradfk, lambdak, gradhk)

        iter += 1
        if np.linalg.norm(gradLapl) <= epsilon:
            break

    return xk, lambdak, iter

class VerificationResultat(ut.TestCase):

    def test_problem_1(self):
        #------------------------------------------------------------ initialisation x0
        x01 = np.array([[1.],[-1.]])
        x02 = np.array([[-3/2],[2.]])
        x03 = np.array([[-.1],[1.]])
        lambda0 = np.array([[1.]])

        #------------------------------------------------------------ initialisation f et h
        f = lambda x : x[0,0] + x[1,0]
        gradf = lambda x : np.ones((2,1))
        grad2f = lambda x : np.zeros((2,2))
        h = lambda x : [x[0,0]**2 + (x[1,0] - 1)**2 - 1,]
        gradh = lambda x : [np.array([[2*x[0,0]],[2*x[1,0]-2]]),]
        grad2h = lambda x : [2*np.eye(2),]

        #------------------------------------------------------------ solve
        x1, lam1, iter = pqsl(x01, lambda0, f, h, gradf, gradh, grad2f, grad2h)
        x2, lam2, iter = pqsl(x02, lambda0, f, h, gradf, gradh, grad2f, grad2h)
        x3, lam3, iter = pqsl(x03, lambda0, f, h, gradf, gradh, grad2f, grad2h)

        #------------------------------------------------------------ Test
        np.testing.assert_array_almost_equal(x1, np.array([[-7.07107e-1],[2.92893e-1]]))
        np.testing.assert_array_almost_equal(x2, np.array([[-7.07107e-1],[2.92893e-1]]))
        np.testing.assert_array_almost_equal(x3, np.array([[-7.07107e-1],[2.92893e-1]]))
        np.testing.assert_array_almost_equal(lam1, np.array([[7.07107e-1]]))
        np.testing.assert_array_almost_equal(lam2, np.array([[7.07107e-1]]))
        np.testing.assert_array_almost_equal(lam3, np.array([[7.07107e-1]]))

    def test_problem_2(self):
        #------------------------------------------------------------ initialisation x0
        x0 = np.array([[-1.],[0.]])
        lambda0 = np.array([[1.]])

        #------------------------------------------------------------ initialisation f et h
        f = lambda x : 100*(x[1,0] - x[0,0]**2)**2 + (1. - x[0,0])**2
        gradf = lambda x : np.array([
            [-400*x[0,0]*x[1,0] + 400*x[0,0]**3 + 2*x[0,0] - 2],
            [200*x[1,0] - 200*x[0,0]**2                       ]
        ])
        grad2f = lambda x : np.array([
            [-400*x[1,0] + 1200*x[0,0]**2 + 2, -400*x[0,0]],
            [-400*x[0,0]                     , 200.       ]
        ])
        h = lambda x : [x[0,0] - x[1,0]**2 - 1/2,]
        gradh = lambda x : [np.array([
            [1.       ],
            [-2*x[1,0]]
        ]),]
        grad2h = lambda x : [np.array([
            [0., 0. ],
            [0., -2.]
        ]),]

        #------------------------------------------------------------ solve
        x, lam, iter = pqsl(x0, lambda0, f, h, gradf, gradh, grad2f, grad2h)

        #------------------------------------------------------------ Test
        np.testing.assert_array_almost_equal(x, np.array([[6.64029e-1],[4.05006e-1]]))
        np.testing.assert_array_almost_equal(lam, np.array([[-8.87139]]))

    def test_problem_3(self):
        #------------------------------------------------------------ initialisation x0
        x0 = 2*np.ones((3,1))
        lambda0 = np.ones((5,1))
        ineq = [2,3,4]

        #------------------------------------------------------------ initialisation f et h
        f = lambda x : 1000 - x[0,0]**2 - 2*x[1,0]**2 - x[2,0]**2 - x[0,0]*x[1,0] - x[0,0]*x[2,0]
        gradf = lambda x : np.array([
            [-2*x[0,0] - x[1,0] - x[2,0]],
            [-4*x[1,0] - x[0,0]         ],
            [-2*x[2,0] - x[0,0]         ]
        ])
        grad2f = lambda x : np.array([
            [-2., -1., -1.],
            [-1., -4., 0. ],
            [-1., 0. , -2.]
        ])
        h = lambda x : [
            8*x[0,0] + 14*x[1,0] + 7*x[2,0] - 56,
            x[0,0]**2 + x[1,0]**2 + x[2,0]**2 - 25,
            x[0,0],
            x[1,0],
            x[2,0]
        ]
        gradh = lambda x : [np.array([
            [8. ],
            [14.],
            [7. ]       
        ]), np.array([
            [2*x[0,0]],
            [2*x[1,0]],
            [2*x[2,0]]
        ]), np.array([[1.],[0.],[0.]]),
        np.array([[0.],[1.],[0.]]),
        np.array([[0.],[0.],[1.]])]
        grad2h = lambda x : [np.zeros((3,3)), 2*np.eye(3), np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3))]

        #------------------------------------------------------------ solve
        x, lam, iter = pqsl(x0, lambda0, f, h, gradf, gradh, grad2f, grad2h, ineq)

        print(iter)
        print(lam)
        print(x)

        # #------------------------------------------------------------ Test
        # np.testing.assert_array_almost_equal(x, np.array([[6.64029e-1],[4.05006e-1]]))
        # np.testing.assert_array_almost_equal(lam, np.array([[-8.87139]]))

if __name__ == '__main__':
    ut.main()