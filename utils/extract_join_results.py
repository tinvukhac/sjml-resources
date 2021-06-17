def main():
    print ('Extract join result')
    f = open('../data/join_results/join_results_small_x_small_bnlj.log')
    output_f = open('../data/join_results/join_results_small_x_small_bnlj.log.csv', 'w')
    # algorithms = ['bnlj', 'pbsm', 'dj', 'repj']
    algorithms = ['bnlj']
    header = 'dataset1,dataset2,'
    for algo in algorithms:
        header += '{}_result_size,'.format(algo)
        header += '{}_mbr_tests,'.format(algo)
        # header += '{}_split_count,'.format(algo)
        # if algo == 'repj':
        #     header += '{}_mbr_tests_join,'.format(algo)
        #     header += '{}_mbr_tests_index,'.format(algo)
        header += '{}_duration,'.format(algo)
    header = header[0: -1]
    print (header)
    output_f.writelines('{}\n'.format(header))

    lines = f.readlines()

    for line in lines:
        data = line.strip().split('bnlj_result')
        output_f.writelines(data[1][1:] + '\n')

    output_f.close()
    f.close()


if __name__ == '__main__':
    main()
