import spatial_quality_extractor


def main():
    print ('Extract partitioning feature')
    block_size = 128
    master_path = '../data/master_files/small_datasets/'
    input_f = open('../data/spatial_descriptors/spatial_descriptors_small_datasets.csv')
    output_f = open('../data/spatial_descriptors/spatial_descriptors_small_datasets_v2.csv', 'w')

    line = input_f.readline()
    line = line.strip() + ',area,margin,overlap,util,balance'
    output_f.writelines(line + '\n')

    line = input_f.readline()
    while line:
        # master_file = master_path + line.strip().split(',')[0] + '/_master.rsgrove'
        master_file = master_path + 'bit_001.csv' + '/_master.rsgrove'
        partitions = spatial_quality_extractor.get_partitions(master_file, block_size)
        total_area = spatial_quality_extractor.get_total_area(partitions)
        total_margin = spatial_quality_extractor.get_total_margin(partitions)
        total_overlap = spatial_quality_extractor.get_total_overlap(partitions)
        disk_util = spatial_quality_extractor.get_disk_util(partitions, block_size)
        load_balance = spatial_quality_extractor.get_size_std(partitions)
        line = line.strip() + '{},{},{},{},{}\n'.format(total_area, total_margin, total_overlap, disk_util, load_balance)

        output_f.writelines(line)

        line = input_f.readline()

    output_f.close()
    input_f.close()


if __name__ == '__main__':
    main()
