def main():
    print ('Extract largexlarge result')
    input_f = open('../data/join_results/join_results_large_x_large.csv')
    output_f = open('../data/join_results/join_results_large_x_large_formatted.csv', 'w')

    lines = input_f.readlines()
    for line in lines:
        data = line.strip().split(',')
        join_names = data[0]
        join_names_data = join_names.split('_')
        if join_names_data[1] == 'enx':
            left = '{}_{}_{}_'.format(join_names_data[0], join_names_data[1], join_names_data[2])
            right = join_names.replace(left, '')
            left = left[0:-1]
            print ('left:{} , right:{}'.format(left, right))
            output_f.writelines('{},large_aws/{}.csv,large_aws/{}.csv\n'.format(line.strip(), left, right))
        else:
            left = '{}_{}_'.format(join_names_data[0], join_names_data[1])
            right = join_names.replace(left, '')
            left = left[0:-1]
            print ('left:{} , right:{}'.format(left, right))
            output_f.writelines('{},large_aws/{}.csv,large_aws/{}.csv\n'.format(line.strip(), left, right))


if __name__ == '__main__':
    main()
