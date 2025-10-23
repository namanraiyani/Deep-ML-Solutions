# https://www.deep-ml.com/problems/142

def gridworld_policy_evaluation(policy: dict, gamma: float, threshold: float) -> list[list[float]]:
    grid_size = 5
    states = [(i,j) for i in range(grid_size) for j in range(grid_size)]
    V = {state: 0.0 for state in states}
    terminal_states = {(0,0), (0,grid_size-1), (grid_size-1,0), (grid_size-1,grid_size-1)}

    def get_next_state_reward(s, a):
        i, j = s
        if a == 'up':
            i = max(i-1, 0)
        elif a == 'down':
            i = min(i+1, grid_size-1)
        elif a == 'left':
            j = max(j-1, 0)
        else:
            j = min(j+1, grid_size-1)
        next_state = (i,j)
        reward = -1
        return next_state, reward

    # bellman equation implementation-> 
    while True:
        delta = 0
        new_V = V.copy()
        
        for state in V:
            if state in terminal_states:
                new_V[state] = 0.0
                continue
            val = 0
            for action, action_prob in policy[state].items():
                next_state, reward = get_next_state_reward(state, action)
                val += action_prob * (reward + gamma * V[next_state])
            delta = max(delta, abs(val-V[state]))
            new_V[state] = val
        V = new_V
        if delta < threshold:
            break
    result = [[V[(i,j)] for j in range(grid_size)] for i in range(grid_size)]    
    return result
