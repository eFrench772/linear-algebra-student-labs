import numpy as np
import pandas as pd

def classical_gram_schmidt(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    for j in range(n):
        v = A[:, j].copy()
        
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]

        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    
    return Q, R

def compute_errors(A, Q, R):
    n = A.shape[1]
    
    #\| A - QR \|_2 \\
    error1 = np.linalg.norm(A - Q @ R, 2)
    
    #\| Q^T Q - I_{n} \|_2 \\
    error2 = np.linalg.norm(Q.T @ Q - np.eye(n), 2)
    
    #\| R - \text{toupper}(R) \|_2
    error3 = np.linalg.norm(R - np.triu(R), 2)
    
    return error1, error2, error3

epsilon_values = [10**(-i) for i in range(6, 17)]
results = []

for eps in epsilon_values:
    A_eps = np.array([
        [1, 1 + eps],
        [1 + eps, 1]
    ])
    
    Q, R = classical_gram_schmidt(A_eps)
    
    error1, error2, error3 = compute_errors(A_eps, Q, R)
    
    results.append({
        'epsilon': eps,
        'error_1': error1,
        'error_2': error2,
        'error_3': error3
    })

#create table
df = pd.DataFrame(results)
df['epsilon'] = df['epsilon'].apply(lambda x: f'{x:.0e}')


print("Table:")
print(df.to_string(index=False))