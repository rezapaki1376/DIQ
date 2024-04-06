import matplotlib.pyplot as plt

your_path = "/Users/camillasancricca/Desktop/" # example of path

def mean(results_all):
    list_mean = []
    for res in results_all:
        list_mean.append(res["mean_perf"])
    return list_mean

def distance(results_all):
    list_over = []
    for res in results_all:
        list_over.append(res["distance"])
    return list_over

def speed(results_all):
    list_speed = []
    for res in results_all:
        list_speed.append(res["speed"])
    return list_speed

def generateFigurePerformance(x_axis, xlabel, results_all, title, legend, score):

    plt.title(title)
    for i in range(0,len(results_all)):

        mean_perf = mean(results_all[i])

        plt.plot(x_axis, mean_perf, marker='o', label=legend[i], markersize=3)

    plt.xlabel(xlabel)
    plt.ylabel(score)
    plt.legend()
    #plt.ylim(0.1, 2)  # if you want to fix a limit for the y_axis
    #plt.savefig(your_path + title + ".pdf", bbox_inches='tight') # if you want to save the figure
    plt.show()

def generateFigureDistance(x_axis, xlabel, results_all, title, legend, score):

    plt.title(title)
    for i in range(0,len(results_all)):

        distance_perf = distance(results_all[i])

        plt.plot(x_axis, distance_perf, marker='o', label=legend[i], markersize=3)

    plt.xlabel(xlabel)
    plt.ylabel(score)
    plt.legend()
    #plt.ylim(0.1, 2) # if you want to fix a limit for the y_axis
    #plt.savefig(your_path + title + ".pdf", bbox_inches='tight') # if you want to save the figure
    plt.show()

def generateFigureSpeed(x_axis, xlabel, results_all, title, legend, score):

    plt.title(title)
    for i in range(0,len(results_all)):

        speed_perf = speed(results_all[i])

        plt.plot(x_axis, speed_perf, marker='o', label=legend[i], markersize=3)

    plt.xlabel(xlabel)
    plt.ylabel(score)
    plt.legend()
    #plt.ylim(0.1, 2)  # if you want to fix a limit for the y_axis
    #plt.savefig(your_path + title + ".pdf", bbox_inches='tight') # if you want to save the figure
    plt.show()

def plot(x_axis_values, x_label, results, title, algorithms, plot_type):

    title = str(title)

    if plot_type == "performance":
        if algorithms[0] == "DecisionTree":
            generateFigurePerformance(x_axis_values, x_label, results, title, algorithms, "f1 weighted")
        elif algorithms[0] == "LinearRegressor":
            generateFigurePerformance(x_axis_values, x_label, results, title, algorithms, "RMSE")
        else:
            generateFigurePerformance(x_axis_values, x_label, results, title, algorithms, "silhouette")

    elif plot_type == "distance train-test": # only for classification & regression
        if algorithms[0] == "DecisionTree":
            generateFigureDistance(x_axis_values, x_label, results, title, algorithms, "f1_train - f1_test")
        else:
            generateFigureDistance(x_axis_values, x_label, results, title, algorithms, "RMSE_test - RMSE_train")

    else:
        generateFigureSpeed(x_axis_values, x_label, results, title, algorithms, "speed")

