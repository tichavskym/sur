import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 12})


def load_scores(filename):
    score_errors = []
    with open(filename, "r") as file:
        for line in file:
            error = line.split()[1]
            score_errors.append(float(error))
    return score_errors


def load_training(filename):
    errors = []
    with open(filename, "r") as file:
        for line in file:
            error = -(float(line.split()[0]) + float(line.split()[1]))
            errors.append(float(error))
    return errors


if __name__ == "__main__":
    scores_16 = load_scores("models/gmm_16.results")
    scores_24 = load_scores("models/gmm_24.results")
    scores_32 = load_scores("models/gmm_32.results")

    training_16 = load_training("models/gmm_16.training")
    training_24 = load_training("models/gmm_24.training")
    training_32 = load_training("models/gmm_32.training")


    x_axis = list(range(1, 100 + 1))

    fig, ax1 = plt.subplots(figsize=(10, 8))

    ax1.plot(
        x_axis, scores_16, label="Validation error, 16 components", color="firebrick"
    )
    ax1.plot(
        x_axis, scores_24, label="Validation error, 24 components", color="mediumblue"
    )
    ax1.plot(
        x_axis, scores_32, label="Validation error, 32 components", color="darkgreen"
    )
    ax1.legend()

    ax2 = ax1.twinx()
    ax2.set(ylabel="Training Dataset Error")
    ax2.plot(
        x_axis,
        training_16,
        label="Training error, 16 components",
        linestyle=":",
        color="firebrick",
    )
    ax2.plot(
        x_axis,
        training_24,
        label="Training error, 24 components",
        linestyle=":",
        color="mediumblue",
    )
    ax2.plot(
        x_axis,
        training_32,
        label="Training error, 32 components",
        linestyle=":",
        color="darkgreen",
    )

    ax1.set(xlabel="# of iterations", ylabel="Validation Dataset Error")
    ax1.set_title("GMM errors with respect to the number of iterations & components")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.grid(True)
    plt.savefig("gmm_errors.png")
