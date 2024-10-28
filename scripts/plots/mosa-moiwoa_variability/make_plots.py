import csv
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":

    # read data
    with open('time.csv', 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

    rows = data[1:]
    time = np.array([float(row[4]) for row in rows])

    with open('NOS.csv', 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

    rows = data[1:]
    NOS = np.array([float(row[1]) for row in rows])

    with open('MID.csv', 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

    rows = data[1:]
    MID = np.array([float(row[1]) for row in rows])

    with open('distance.csv', 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

    rows = data[1:]
    distance = np.array([float(row[1]) for row in rows])

    data = [NOS, MID, distance]

    # plot
    plt.boxplot(time, vert=True, patch_artist=True,
                boxprops=dict(facecolor='skyblue', color='black'),
                medianprops=dict(color='red'))

    plt.xticks([1], ["Esecution time (s)"])
    plt.title("Execution time variability for MOSA-MOIWOA")

    plt.savefig('time_variability.png')
    plt.close()

    plt.boxplot(data, vert=True, patch_artist=True,
                boxprops=dict(facecolor='skyblue', color='black'),
                medianprops=dict(color='red'))

    plt.xticks([1, 2, 3], ["NOS", "MID", "Distance"])
    plt.title("Metrics variability for MOSA-MOIWOA")

    plt.savefig('metrics_variability.png')
    plt.close()
