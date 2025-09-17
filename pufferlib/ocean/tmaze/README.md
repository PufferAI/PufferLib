The T-maze is designed to test the memory capacity of RL algorithms under partial observability.

* The maze consists of a corridor of length N, ending in a T-junction with two final states (left and right). 
* Start condition: At the first tile of the corridor, the observation contains a special marker (3 or 4). This marker determines which final state (left or right) will yield reward +1 and which yields -1.
* Termination: The episode terminates immediately when the agent reaches a final state.
* Observations: local observation: obs = [current, front, left, right] with values: 0=wall, 1=open, 2&3 being the two possible states of the starting tile.
* Actions: The agent can either go forward, left or right. It can not go back or turn inside the corridor
* Rewards: Rewards are 0 everywhere except on the final states. If the starting state was a 2, the reward of 1 is on the left and -1 on the right. If the starting state was a 3, the reward of 1 is on the right and -1 on the left. 

Examples of observations:

* Middle of corridor: [1,1,0,0]
* At T-junction: [1,0,1,1]
* Starting state: [3,1,0,0]
* At a final/terminal state: [1,0,0,0]
