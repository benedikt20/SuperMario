# Yale University
# S&DS665: IML
# Benedikt Farag, Dec 18th 2024
# ----------------
#
# Q-learning for Super Mario
#
import numpy as np

N=5 # Number of horizontal locations
H=2 # Number of vertical locations

# Actions
l, r, j = 0, 1, 2 # Move left, right, jump


# Q-learning parameters
g=0.5 # Discount factor
Iter=1000 # Iterations 






# ----------------------------------------
# Program - do not change unless to fix
# ----------------------------------------

n=N-1 # Max state # (for python index 0,...,N-1)
h=H-1 # Max state # (for python index 0,...,N-1)

# Function to get turtle location 0<=x<=n
def get_t_loc(t_loc):
    if t_loc == 0:
        t=n
    else:
        t = t_loc - 1
    return t

# Function to get Mario's location [0,0]<=[x,y]<=[n,1]

def get_m_loc(m_loc, a):
    if a == l:  # Move left
        new_x = max(0, m_loc[0] - 1)
    elif a == r:  # Move right
        new_x = min(n, m_loc[0] + 1)
    else:  # Jump
        new_x = m_loc[0]

    new_y = 1 if a == j and m_loc[1] == 0 else 0  # Vertical location
    return np.array([new_x, new_y])

# Define state array (nx2n matrix)
states=np.arange(0,2*N**2)
states=np.reshape(states,(2*N,N)).T

# Initialize reward and next arrays
reward=np.zeros((2*N**2,3))
next_state=np.zeros((2*N**2,3))

# Loop over all possible states and actions to assemble next_state and reward matrices
for mh in range(0,N):
    for mv in range(0,2):
        M=np.array([mh,mv])
        for t in range(0,N):
            for a in range(0,3):
                t_loc = get_t_loc(t)
                m_loc = get_m_loc(M,a)
                next_state[states[M[0],N*M[1]+t],a]=states[m_loc[0],N*m_loc[1]+t_loc]

                if t_loc == m_loc[0] and m_loc[1]==0:
                    reward[states[M[0],N*M[1]+t],a]=-100
                if M[0]==t-1 and M[1]==0 and a==1:
                    reward[states[M[0],N*M[1]+t],a]=-100
                if m_loc[0]==n and m_loc[1]==1: #and t_loc != n 
                    reward[states[M[0],N*M[1]+t],a]=20


# Q-learning              
Q=np.zeros(next_state.shape)
for _ in range(0,Iter):
    Q_old=Q.copy()
    for i in range(0,Q.shape[0]):
        for j in range(0,Q.shape[1]):
            if i==states[n,1]: 
                Q[i,:]=0
            else:
                Q[i,j]=reward[i,j]+g*max(Q_old[int(next_state[i,j]),])

Pi=np.argmax(Q,axis=1)

# ---------------------
# Outputs:
# ---------------------
#print(reward,"\n")
#print(next_state,"\n")

#print("Q: \n",Q,"\n") # Q-vals
#print("V: \n",np.max(Q,axis=1).T,"\n") # Value matrix
print("Pi: \n",Pi,"\n") # Policy




