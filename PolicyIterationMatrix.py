import numpy as np

Q_Table = np.zeros((4,4))

T_Matrix = np.array(
     [
        [
            [0.66667, 0.00000, 0.33333, 0.00000, 0.00000],
            [0.33333, 0.33333, 0.33333, 0.00000, 0.00000],
            [0.33333, 0.33333, 0.33333, 0.00000, 0.00000],
            [0.66667, 0.33333, 0.00000, 0.00000, 0.00000]
        ],
        [
            [0.33333, 0.33333, 0.00000, 0.33333, 0.33333],
            [0.33333, 0.33333, 0.00000, 0.33333, 0.33333],
            [0.00000, 0.66667, 0.00000, 0.33333, 0.33333],
            [0.33333, 0.66667, 0.00000, 0.00000, 0.00000]
        ],
        [
            [0.00000, 0.00000, 1.00000, 0.00000, 0.00000],
            [0.00000, 0.00000, 1.00000, 0.00000, 0.00000],
            [0.00000, 0.00000, 1.00000, 0.00000, 0.00000],
            [0.00000, 0.00000, 1.00000, 0.00000, 0.00000]
        ],
        [
            [0.00000, 0.00000, 1.00000, 1.00000, 0.00000],
            [0.00000, 0.00000, 1.00000, 1.00000, 0.00000],
            [0.00000, 0.00000, 1.00000, 1.00000, 0.00000],
            [0.00000, 0.00000, 1.00000, 1.00000, 0.00000]
        ]
    ] 
)

actions = ["left", "down", "right", "up"]
policy = [3,2,2,2] # which way to try to move given state
reward_index = 4 # reward is the 5th entry in the table after all the state probabilities
n_iterations = 20
n_states = 4

for iteration in range(n_iterations):
    print(Q_Table)
    Q_Table_New = np.copy(Q_Table)
    for state in range(n_states):
        for action in range(len(actions)):
            r = T_Matrix[state, action, reward_index]
            q = 0
            q = q + r
            for next_state in range(n_states):
                p_next = T_Matrix[state, policy[state], next_state]
                q_next = Q_Table[next_state, policy[next_state]]
                q = q + p_next*q_next

            Q_Table_New[state, action] = q
    
    Q_Table = Q_Table_New

    # wait to press enter
    input()