import numpy as np
import unittest as ut

def pqsl(x0:np.ndarray, lambda0:np.ndarray, gradf, gradh, grad2f, grad2h, epsilon:float=10e-5):
    pass

class VerificationResultat(ut.TestCase):

    def test_problem_1(self):
        gradf = lambda x : np.ones((2,1))
        grad2f = lambda x : np.zeros((2,2))
        gradh = lambda x : np.array([[2*x[0]],[2*x[1]+2]])
        grad2h = lambda x : 2*np.eye(2)

if __name__ == '__main__':
    pass