# SuperMario
Q-learning implementation for Super Mario

Super Mario can move left, right or jump.
The turtle moves left consistently and starts over on the right end if reaches the left boundary, i.e. it moves left mod n. 

The environment is Nx2 grid and there are $2N^2$ number of states. 

The output is the converged Q-matrix, value matrix and the optimal policy matrix Pi.
