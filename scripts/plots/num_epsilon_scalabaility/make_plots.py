import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


if __name__ == '__main__':

    # read data
    with open('num_epsilon_scalability.csv', 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

    # compute data to be plotted
    rows = data[1:]
    num_epsilon = [float(row[0]) for row in rows]
    total_time = [float(row[1]) for row in rows]
    NOS = [float(row[2]) for row in rows]
    MID = [float(row[3]) for row in rows]
    distance = [float(row[4]) for row in rows]

    # plot time
    plt.plot(num_epsilon, total_time, marker='o')
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.xlabel('Number of epsilon values')
    plt.ylabel('Total execution time (s)')
    plt.title('Execution time of epsilon-constraint')

    plt.savefig('time.png')
    plt.close()

    # plot NOS
    plt.plot(num_epsilon, NOS, marker='o')
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.xlabel('Number of epsilon values')
    plt.ylabel('Number of Pareto solutions')
    plt.title('NOS of epsilon-constraint')

    plt.savefig('NOS.png')
    plt.close()

    # plot MID
    plt.plot(num_epsilon, MID, marker='o')
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.xlabel('Number of epsilon values')
    plt.ylabel('Mean of ideal distance')
    plt.title('MID of epsilon-constraint')

    plt.savefig('MID.png')
    plt.close()

    # plot distance
    plt.plot(num_epsilon, distance, marker='o')
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.xlabel('Number of epsilon values')
    plt.ylabel('Distance of Pareto solutions')
    plt.title('Distance of epsilon-constraint')

    plt.savefig('distance.png')
    plt.close()
