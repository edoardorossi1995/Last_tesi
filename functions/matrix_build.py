import numpy as np

def build_T(df_mdp):

    SA_col = df_mdp[['S', 'At']]
    
    S = SA_col['S'].max()+1
    A = SA_col['At'].max()+1
    T = np.zeros((S, S, A), dtype=float)

    for i in range(S):
        for j in range(A):
            idx = SA_col.loc[(SA_col['S'] == i) & (SA_col['At'] == j)]
            T[i, idx, j] += 1
            
    
    # Normalize T to be a stochastic matrix
    T = T.astype('float64')

    none_index = []
    for a in range(A):
        for s in range(S):
            if T[s, :, a].sum() > 0:
                T[s, :, a] /= T[s, :, a].sum()
            if T[s,:,a].sum() == 0:
                none_index.append((s,a))
    for index in none_index:
        T[index[0],index[0],index[1]] = 1

    return T, none_index


def is_ergodic(T):
    n_states = T.shape[0]
    P_reach = np.zeros((n_states, n_states))
    P_reach[0, :] = T[0, :, 0]
    for i in range(1, n_states):
        P_reach[i, :] = np.dot(P_reach[i-1, :], T[i, :, :].max(axis=1))
    return np.all(P_reach > 0)

def is_markovian(T):
    n_states = T.shape[0]
    for s in range(n_states):
        for a in range(T.shape[2]):
            if not np.isclose(T[s, :, a].sum(), 1.0):
                return False
    return True
