import operator

def main():
    print ('Compute algorithm rank')
    f = open('../data/join_results/sj.12_30.log.csv')
    output_f = open('../data/join_results/sj.12_30.log.ranked.csv', 'w')

    header = f.readline()
    header = header.strip()
    header += ',1st time,2nd time,3rd time,4th time, 1st #splits,2nd #splits,3rd #splits,4th #splits\n'
    output_f.writelines(header)
    line = f.readline()

    while line:
        data = line.strip().split(',')
        duration = {}
        duration['pbsm'] = float(data[9]) if float(data[9]) > 0 else 10000
        duration['dj'] = float(data[13]) if float(data[13]) > 0 else 10000
        duration['repj'] = float(data[17]) if float(data[17]) > 0 else 10000
        duration['bnlj'] = float(data[5]) if float(data[5]) > 0 else 10000
        # print (duration)
        sorted_duration = sorted(duration.items(), key=operator.itemgetter(1))
        # print (sorted_duration)
        line = line.strip()
        for sorted_entry in sorted_duration:
            print (sorted_entry[0])
            line += ',{}'.format(sorted_entry[0])

        split_counts = {}
        split_counts['pbsm'] = float(data[8]) if float(data[8]) > 0 else 10000
        split_counts['dj'] = float(data[12]) if float(data[12]) > 0 else 10000
        split_counts['repj'] = float(data[16]) if float(data[16]) > 0 else 10000
        split_counts['bnlj'] = float(data[4]) if float(data[4]) > 0 else 10000
        print (duration)
        sorted_split_counts = sorted(split_counts.items(), key=operator.itemgetter(1))
        # print (sorted_duration)
        for sorted_entry in sorted_split_counts:
            print (sorted_entry[0])
            line += ',{}'.format(sorted_entry[0])

        output_f.writelines('{}\n'.format(line))

        line = f.readline()

    output_f.close()
    f.close()


if __name__ == '__main__':
    main()
