def main():
    print ('Filter data')
    input_f = open('../data/result_size_large_files.tsv')
    output_f = open('../data/result_size_large.csv', 'w')

    lines = input_f.readlines()

    distributions = ['Combo', 'Diagonal', 'DiagonalRot', 'Gauss', 'Parcel', 'Uniform']

    for line in lines:
        print (line)
        data = line.strip().split('\t')
        filenames = data[0].split('_')
        print (filenames)
        filenames_array = []
        a = []
        for s in filenames:
            if s in distributions:
                filenames_array.append(a)
                a = [s]
            else:
                a.append(s)
        filenames_array.append(a)
        filenames_array = filenames_array[1:]
        datasets = []
        for a in filenames_array:
            datasets.append('_'.join(a))
        print (len(datasets))

        output_f.writelines('{}.csv,{}.csv,{}\n'.format(datasets[0], datasets[1], data[1]))

    output_f.close()
    input_f.close()


if __name__ == '__main__':
    main()
