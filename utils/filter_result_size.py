from itertools import combinations


def main():
    print ('Filter result size to a single distribution')

    distributions = ['diagonal', 'gaussian', 'parcel', 'uniform']
    # distributions = [['diagonal', 'uniform'], ['gaussian', 'uniform'], ['parcel', 'uniform']]

    for r in range(1, len(distributions) + 1):
        groups = combinations(distributions, r)

        for dists in groups:
            input_f = open('../data/result_size.csv')
            output_f = open('../data/result_size_{}.csv'.format('_'.join(dists)), 'w')

            line = input_f.readline()

            while line:
                data = line.strip().split(',')
                dist1 = data[0].split('_')[0]
                dist2 = data[1].split('_')[0]
                if dist1 in dists and dist2 in dists:
                    output_f.writelines(line)

                line = input_f.readline()

            output_f.close()
            input_f.close()


if __name__ == '__main__':
    main()
