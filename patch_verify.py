import re

with open("verification/verify_heckman.py", "r") as f:
    text = f.read()

text = text.replace("family_size: family_size", "family_size: family_size.astype(float)")
text = text.replace("employed: participation", "employed: participation.astype(float)")
text = text.replace("'family_size': family_size,", "'family_size': family_size.astype(float),")
text = text.replace("'employed': participation", "'employed': participation.astype(float)")

with open("verification/verify_heckman.py", "w") as f:
    f.write(text)
