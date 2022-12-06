import numpy as np
import unittest as ut

def pqsl(x0:np.ndarray, lambda0:np.ndarray, f, h, gradf, gradh, grad2f, grad2h, epsilon:float=10e-5):
    xk, lambdak = x0, lambda0
    gradfk, gradhk, grad2fk, grad2hk = gradf(xk), gradh(xk), grad2f(xk), grad2h(xk)
    Lk = grad2fk + lambdak @ grad2hk
    print(Lk)
    print(gradhk)
    print(np.concatenate((Lk,gradhk), axis=1))

def test():
    x0 = np.array([[1.],[-1.]])
    lambda0 = np.array([[1.,1.]])

    f = lambda x : x[0,0] + x[1,0]
    gradf = lambda x : np.ones((2,1))
    grad2f = lambda x : np.zeros((2,2))
    h = lambda x : x[0,0]**2 + (x[1,0] + 1)**2 - 1
    gradh = lambda x : np.array([[2*x[0,0]],[2*x[1,0]+2]])
    grad2h = lambda x : 2*np.eye(2)

    pqsl(x0, lambda0, f, h, gradf, gradh, grad2f, grad2h)

class VerificationResultat(ut.TestCase):

    def test_problem_1(self):
        f = lambda x : x[0] + x[1]
        gradf = lambda x : np.ones((2,1))
        grad2f = lambda x : np.zeros((2,2))
        h = lambda x : x[0]**2 + (x[1] + 1)**2 - 1
        gradh = lambda x : np.array([[2*x[0]],[2*x[1]+2]])
        grad2h = lambda x : 2*np.eye(2)

if __name__ == '__main__':
    test()