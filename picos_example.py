import picos as pic
import cvxopt as cvx
#P = pic.Problem()
#Z = P.add_variable('Z',(3,2),'complex')
import numpy
#numpy.random.seed(200)
n = 60
s = (n,n)
P = numpy.ones(s) + numpy.ones(s)*1j
P = numpy.matrix(P)
P = cvx.matrix(P)
s = (n,n)
#P = P*numpy.matrix.getH(P)
Q = numpy.ones(s) + numpy.ones(s)*1j
Q = numpy.matrix(Q)
Q = cvx.matrix(Q)
P = P * P.H
Q = Q * Q.H

n=P.size[0]
P = pic.new_param('P',P)
Q = pic.new_param('Q',Q)

#create the problem in picos

F = pic.Problem()
Z = F.add_variable('Z',(n,n),'complex')

F.set_objective('max','I'|0.5*(Z+Z.H))       #('I' | Z.real) works as well
F.add_constraint(((P & Z) // (Z.H & Q))>>0 )
#print F

%time F.solve(verbose = 0)