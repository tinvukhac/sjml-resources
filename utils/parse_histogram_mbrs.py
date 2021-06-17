from geomet import wkt
import json


def main():
    print('Parse histogram mbrs')
    num_rows = 16
    num_columns = 16

    input_f = open('../data/histograms/{}x{}/diagonal_001.csv'.format(num_rows, num_columns))
    output_f = open('../data/histogram_{}_{}_mbrs.csv'.format(num_rows, num_columns), 'w')

    lines = input_f.readlines()
    lines = lines[1:]

    for line in lines:
        data = line.split('\t')
        polygon = data[2]
        mbr = wkt.loads(polygon)
        output_f.writelines('{},{},{},{}\n'.format(mbr['coordinates'][0][0][0], mbr['coordinates'][0][0][1], mbr['coordinates'][0][2][0], mbr['coordinates'][0][2][1]))

        # print (mbr['coordinates'][0][0])
        # print (mbr['coordinates'][0][2])

    output_f.close()
    input_f.close()


if __name__ == '__main__':
    main()
