import csv
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # read data
    with open('profiling_mm.csv', 'r') as f:
        reader = csv.reader(f)
        data_mm = list(reader)

    with open('profiling_ec.csv', 'r') as f:
        reader = csv.reader(f)
        data_ec = list(reader)

    # compute data to be plotted
    rows = data_ec[1:]
    problem_ids = [float(row[0]) for row in rows]
    time_totals_ec = [float(row[4]) for row in rows]

    rows = data_mm[1:]
    time_totals_mm = [float(row[4]) for row in rows]

    time_avgs_mm = [sum(time_totals_mm[i:i + 8]) / 8 for i in range(0, len(time_totals_mm), 8)]

    # plot
    plt.plot(problem_ids, time_totals_ec, label='epsilon-constraint', color='blue', marker='o')
    plt.plot(problem_ids, time_avgs_mm, label='MOSA-MOIWOA', color='red', marker='x')

    plt.xlabel('Problem ID')
    plt.ylabel('Total Time to solution (s)')
    plt.title('Scalability of epsilon-constraint and MOSA-MOIWOA')

    plt.legend()

    plt.savefig('scalability.png')
    plt.close()
