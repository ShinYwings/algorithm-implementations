import numpy as np

def isPossibleToFormLU(A):

    # TODO pivoting없이 REF 되는지 확인
    return 

def LU(A):

    M = np.copy(A)
    n = len(M[:,0])

    for p in range(0, n):
        for r in range(p+1, n):
            s = (-M[r,p]) / (M[p,p])
            M[r,p] = -s #하삼각행렬 E^-1 에 저장
            for c in range(p+1, n):
                M[r,c] = M[r,c] + s * M[p,c]

    #LDU로 바꿔 유일하게 인수분해 될 수 있게 바꿔줌
    for p in range(0,n):
        for r in range(p+1, n):
            M[p,r] = M[p,r] / M[p,p]

    return M

if __name__ == "__main__":

    A = np.matrix([[2,6,2],[-3,-8,0],[4,9,2]], dtype=float)
    
    print(A)
    
    result = LU(A)

    print("result: \n", result)

    print("A det: ",np.linalg.det(A))
    
    ans =1
    for i in range(0,3):
        ans = ans * result[i,i]
    print("\nresult det: \n", ans)


