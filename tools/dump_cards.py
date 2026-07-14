"""Regenerate data/cards.json and data/attacks.json from the installed engine."""
import ctypes, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from kaggle_environments.envs.cabt.cg.sim import lib

lib.AllCard.restype = ctypes.c_char_p
lib.AllAttack.restype = ctypes.c_char_p
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
json.dump(json.loads(lib.AllCard().decode()), open(os.path.join(out, "cards.json"), "w"), ensure_ascii=False)
json.dump(json.loads(lib.AllAttack().decode()), open(os.path.join(out, "attacks.json"), "w"), ensure_ascii=False)
print("dumped cards.json / attacks.json")
