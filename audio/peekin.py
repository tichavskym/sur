import io
from glob import glob

from gmm import classify
from contextlib import redirect_stdout

FILENAME = "gmm_24*.npz"
AUGMENTED = True
TARGETS = 5*10 if AUGMENTED else 10
TOTAL = 5*70 if AUGMENTED else 70

for file in glob("models/" + FILENAME):
    f = io.StringIO()
    with redirect_stdout(f):
        classify(f"{file}", "data/target_dev", augmentation=AUGMENTED)
        classify(f"{file}", "data/non_target_dev", augmentation=AUGMENTED)
    results = f.getvalue()

    i = 0
    loss = 0
    for line in results.split("\n"):
        if i == TOTAL:
            continue

        score = line.split(" ")[1]
        if i < TARGETS:
            loss += 1 - float(score)
        else:
            loss += float(score) - 0
        i += 1

    print(f"{file} {loss}")
