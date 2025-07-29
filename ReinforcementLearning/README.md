**YOU MUST BE IN THE "ReinforcementLearning" FOLDER WHEN YOU RUN THESE COMMANDS**

Run this command to test Approximate Q-learning with ApproximateQAgent in pacmanQAgents.py:
```python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 2000 -n 2005 -l smallGrid```

Run this command for regular Q-learning with QLearningAgent in qLearningAgents.py:
```python pacman.py -p PacmanQAgent -x 2000 -n 2005 -l smallClassic```

Run this command to run test with GridWorld for QLearningAgent in qLearningAgents.py:
```python gridworld.py -a q -k 100```
- can add GridWorld maps at the end w/ ```-g MazeGrid``` etc.

Also there are different Pacman grids as well, such as smallClassic, mediumClassic, trickyClassic, etc.