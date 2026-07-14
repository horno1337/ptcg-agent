"""Run one game and write an HTML replay to replay.html."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from kaggle_environments import make
from kaggle_environments.envs.cabt.cabt import random_agent
from agent import agent as my_agent

env = make("cabt", debug=True)
env.run([my_agent, random_agent])
with open("replay.html", "w") as f:
    f.write(env.render(mode="html"))
print("wrote replay.html; final:", [(s.status, s.reward) for s in env.steps[-1]])
