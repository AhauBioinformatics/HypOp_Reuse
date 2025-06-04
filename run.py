from src.run_exp import exp_centralized
import json

with open('configs/MIS_hypop.json') as f:
   params = json.load(f)
exp_centralized(params)