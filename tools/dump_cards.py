"""Regenerate data/cards.json and data/attacks.json from the local engine build."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cabt import lib

L = lib()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
json.dump(json.loads(L.AllCard().decode()), open(os.path.join(out, "cards.json"), "w"), ensure_ascii=False)
json.dump(json.loads(L.AllAttack().decode()), open(os.path.join(out, "attacks.json"), "w"), ensure_ascii=False)
print("dumped cards.json / attacks.json")
