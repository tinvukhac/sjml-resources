from itertools import combinations


def item_in_string(g, str):
    for item in g:
        if item in str:
            return True
    return False


def main():
    print ('Filter data')

    distributions = ['uniform', 'diagonal', 'gauss', 'parcel', 'bit']

    for r in range(1, len(distributions) + 1):
        groups = combinations(distributions, r)
        for g in groups:
            print(g)
            name = '_'.join(g)
            output_name = '{}distribution.{}'.format(r, name)
            input_f = open('../data/train_and_test_all_features_split/train_join_results_combined_data.csv')
            output_f = open(
                '../data/train_and_test_all_features_split/train_join_results_combined_data.{}.csv'.format(output_name),
                'w')

            line = input_f.readline()

            output_f.writelines(line)
            line = input_f.readline()

            while line:
                data = line.strip().split(',')
                # result_size = int(data[2])
                write = False

                write = item_in_string(g, data[0].lower()) and item_in_string(g, data[1].lower())

                # if 'diagonal' in data[0].lower() and 'gaussian' in data[1].lower():
                #     write = True
                # if 'gaussian' in data[0].lower() and 'gaussian' in data[1].lower():
                #     write = True
                # if 'uniform' in data[0].lower() and 'diagonal' in data[1].lower():
                #     write = True
                # if 'uniform' in data[0].lower() and 'uniform' in data[1].lower():
                #     write = True

                # join_sel = float(data[36])
                # min_sel = pow(10, -6)
                # max_sel = pow(10, -4)
                # if min_sel < join_sel < max_sel:
                #     write = True

                if write:
                    output_f.writelines(line)

                line = input_f.readline()

            output_f.close()
            input_f.close()


if __name__ == '__main__':
    main()
