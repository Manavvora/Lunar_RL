# ae504
This is a code for comparing different algorithms to solve the lunar lander problem. The algorithms simulated are:
1. Monte Carlo
2. Q-Learning
3. DQN
4. SARSA


To run the code `python autopilot.py 10000 models/*/qnetwork_2000.pt`

To track the lunar lander and the final path traversed `python main.py`

# Outputs:

Simulation            |  Final Trajectory
:-------------------------:|:-------------------------:
![random](outputs/random.gif)*Random policy* |  ![random_plot](trajectories/random.png)
![monte_carlo](outputs/monte_carlo.gif)*Monte Carlo* | ![monte_carlo](trajectories/monte_carlo.png)
![q_learning](outputs/qlearning.gif)*QLearning* | ![q_learning](trajectories/q_learning.png)
![sarsa](outputs/sarsa.gif)*SARSA* | ![sarsa](trajectories/sarsa.png)
![dqn](outputs/dqn.gif)*DQN* | ![dqn](trajectories/DQN.png)





