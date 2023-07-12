import numpy as np

def policy_evaluation(T, R, gamma, pi, v):
    S = T.shape[0]
    Ppi = np.zeros((S, S))
    Rpi = np.zeros((S,))

    for s in range(S):
        Ppi[s, :] = T[s, :, pi[s].astype(int)]
        Rpi[s] = R[s, pi[s].astype(int)]
    
    while True:
        vprec = v
        v = Rpi + gamma * np.dot(Ppi, vprec)
        if np.linalg.norm(v - vprec, np.inf) < 1e-6:
            print("p_eval convergenza raggiunta")
            break
    
    return v

def policy_improvement(T, R, gamma, value):
    S = T.shape[0]
    A = T.shape[2]
    
    pip = np.zeros((S, 1))

    for s in range(S):
        q = np.zeros((A, 1))
        for a in range(A):
            trans = T[s, :, a]
    
            q[a] = R[s,a] + gamma*np.dot(trans, value)
        pip[s] = np.argmax(q)
    
    return pip


def policy_iteration(T, R, gamma):
    S = T.shape[0]
    A = T.shape[2]
    
    policy = np.random.randint(A, size=S)
    value = np.zeros(S)
    oldpolicy = policy
    
    while True:
        value = policy_evaluation(T, R, gamma, policy, value)
        policy = policy_improvement(T, R, gamma, value)
        if np.linalg.norm(policy - oldpolicy, np.inf) < 1e-3:
            print("p_iteration convergenza raggiunta")
            break
        oldpolicy = policy
    
    return policy, value


def policy_optim(T, R, gamma, value):
    S = T.shape[0]
    A = T.shape[2]

    new_value = np.zeros(S)
    policy = np.zeros(S)

    for s in range(S):
        q = np.zeros((A, 1))
        for a in range(A):
            trans = T[s, :, a]
            # ho modificato il - alla riga seguente, era +
            q[a] = R[s,a] - gamma* np.dot(trans,value)
        new_value[s] = np.max(q)
        policy[s] = np.argmax(q)
    
    return new_value, policy


def value_iteration(T, R, gamma):
    S = T.shape[0]
    A = T.shape[2]

    value = np.random.randint(A, size=(S, 1))
    print("value" , value)
    prevpolicy = np.random.rand(S, 1)

    while True:
        value, policy = policy_optim(T, R, gamma, value)
        if np.linalg.norm(policy - prevpolicy, np.inf) < 1e-4:
            print("v_iteration convergenza raggiunta")
            break
        prevpolicy = policy
    
    return policy, value