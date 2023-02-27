import numpy as np
import pandas as pd

def forward_substitution(A = "matrix",b = "vector"):

    # AM = np.c_[A, b]   Augmented Matrix
    n = len(A[:,0])
    U = np.matrix.copy(A)   # will be upper triagle matrix
    y = np.matrix.copy(b)
    
    for p in range(0,n):
        
        #partial pivoting    
        pivot = p
        for i in range(p+1,n):
            if np.abs(U[pivot,p]) < np.abs(U[i,p]):
                pivot = i
        U[[p,pivot],:] = U[[pivot,p],:]
        y[[p,pivot]] = y[[pivot,p]]
        # y[p], y[pivot] = y[pivot], y[p] # 이거 안먹힘

        s = 1 / U[p,p]

        y[p] = s * y[p]

        for c in range(p, n):         # 피벗 항 scaling
            U[p,c] = s * U[p,c]

        for r in range(p+1, n):       # 행 연산
            s = -U[r,p]
            y[r] = y[r]+s*y[p]

            for c in range(p, n):
                U[r,c] = U[r,c] + s * U[p,c]

    return U, y

def back_substitution(U="matrix", y="vector"):
    
    x = np.matrix.copy(y)
    n = len(U[:,0])

    for p in range(n-1, -1, -1):
        for r in range(0, p):
            x[r] = x[r] - ((U[r,p]*x[p]) / (U[p,p]))
    return x

if __name__ == "__main__":
    
    A = np.matrix([[1,1,1],[2,1,3],[3,4,-2]])
    b = np.matrix([4,7,9])
    b = np.transpose(b)

    print(np.c_[A,b],"\n")
    
    U, y = forward_substitution(A, b)

    x = back_substitution(U,y)

    print("after forward substitution...\n", np.c_[U,y],"\n")

    print("solution:\n",x)
    