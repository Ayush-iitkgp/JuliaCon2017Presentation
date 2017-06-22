import picos as pic
import cvxopt as cvx
#P = pic.Problem()
#Z = P.add_variable('Z',(3,2),'complex')
P = cvx.matrix([ [1-1j , 2+2j  , 1    ],
                 [3j   , -2j   , -1-1j],
                 [1+2j, -0.5+1j, 1.5  ]
                ])
P = P * P.H

Q = cvx.matrix([ [-1-2j , 2j   , 1.5   ],
                 [1+2j  ,-2j   , 2.-3j ],
                 [1+2j  ,-1+1j , 1+4j  ]
                ])
Q = Q * Q.H

n=P.size[0]
P = pic.new_param('P',P)
Q = pic.new_param('Q',Q)

#create the problem in picos

F = pic.Problem()
Z = F.add_variable('Z',(n,n),'complex')

F.set_objective('max','I'|0.5*(Z+Z.H))       #('I' | Z.real) works as well
F.add_constraint(((P & Z) // (Z.H & Q))>>0 )

print F

F.solve(verbose = 0)

print 'fidelity: F(P,Q) = {0:.4f}'.format(F.obj_value())

print 'optimal matrix Z:'
print Z

#verify that we get the same value with numpy
import numpy as np
PP = np.matrix(P.value)
QQ = np.matrix(Q.value)

S,U = np.linalg.eig(PP)
sqP = U * np.diag([s**0.5 for s in S]) * U.H #square root of P
S,U = np.linalg.eig(QQ)
sqQ = U * np.diag([s**0.5 for s in S]) * U.H #square root of P

fidelity = sum(np.linalg.svd(sqP * sqQ)[1])  #trace-norm of P**0.5 *  Q**0.5

print 'fidelity computed by trace-norm: F(P,Q) = {0:.4f}'.format(fidelity)