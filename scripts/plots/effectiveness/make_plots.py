import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


if __name__ == '__main__':

    # read data
    with open('metrics_mm.csv', 'r') as f:
        reader = csv.reader(f)
        data_mm = list(reader)

    with open('metrics_ec.csv', 'r') as f:
        reader = csv.reader(f)
        data_ec = list(reader)

    # compute data to be plotted
    rows = data_ec[1:]
    problem_ids = [float(row[0]) for row in rows]
    NOS_ec = [float(row[1]) for row in rows]
    MID_ec = [float(row[2]) for row in rows]
    distance_ec = [float(row[3]) for row in rows]

    rows = data_mm[1:]
    NOS_mm = [float(row[1]) for row in rows]
    MID_mm = [float(row[2]) for row in rows]
    distance_mm = [float(row[3]) for row in rows]

    NOS_avgs_mm = [sum(NOS_mm[i:i + 5]) / 5 for i in range(0, len(NOS_ec) * 5, 5)]
    MID_avgs_mm = [sum(MID_mm[i:i + 5]) / 5 for i in range(0, len(MID_ec) * 5, 5)]
    distance_avgs_mm = [sum(distance_mm[i:i + 5]) / 5 for i in range(0, len(distance_ec) * 5, 5)]

    # plot NOS
    plt.plot(problem_ids, NOS_ec, label='epsilon-constraint', color='blue', marker='o')
    plt.plot(problem_ids, NOS_avgs_mm, label='MOSA-MOIWOA', color='red', marker='x')
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.xlabel('Problem ID')
    plt.ylabel('Number of Pareto solutions')
    plt.title('NOS of epsilon-constraint and MOSA-MOIWOA')

    plt.legend()

    plt.savefig('NOS.png')
    plt.close()

    # plot MID
    plt.plot(problem_ids, MID_ec, label='epsilon-constraint', color='blue', marker='o')
    plt.plot(problem_ids, MID_avgs_mm, label='MOSA-MOIWOA', color='red', marker='x')
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.xlabel('Problem ID')
    plt.ylabel('Mean of ideal distance')
    plt.title('MID of epsilon-constraint and MOSA-MOIWOA')

    plt.legend()

    plt.savefig('MID.png')
    plt.close()

    # plot distance
    plt.plot(problem_ids, distance_ec, label='epsilon-constraint', color='blue', marker='o')
    plt.plot(problem_ids, distance_avgs_mm, label='MOSA-MOIWOA', color='red', marker='x')
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.xlabel('Problem ID')
    plt.ylabel('Distance of Pareto solutions')
    plt.title('Distance of solutions of epsilon-constraint and MOSA-MOIWOA')

    plt.legend()

    plt.savefig('distance.png')
    plt.close()
