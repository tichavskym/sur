import io
from glob import glob

from gmm import evaluate_model
from contextlib import redirect_stdout

for file in glob("models/" + "gmm_24*.npz"):
    f = io.StringIO()
    with redirect_stdout(f):
        evaluate_model(f"{file}", "data/target_dev", "data/non_target_dev")
    results = f.getvalue()

    i = 0
    loss = 0
    for line in results.split("\n"):
        if i == 70:
            continue

        score = line.split(" ")[1]
        if i < 10:
            loss += 1 - float(score)
        else:
            loss += float(score) - 0
        i += 1

    print(f"{file} {loss}")
