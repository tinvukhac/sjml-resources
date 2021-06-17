import os


def main():
    print('Copy master files from HDFS')

    # Create directories to store master files
    master_paths = ['masters/large_aws', 'masters/medium', 'masters/small', 'masters/real']
    os.system('rm -r masters')
    for master_path in master_paths:
        os.system('mkdir -p {}'.format(master_path))

    hdfs_index_paths = ['datasets/deep_join/large_aws_indexed',
                        'datasets/deep_join/medium_datasets_indexed_1MB',
                        'datasets/deep_join/small_datasets_indexed',
                        'datasets/deep_join/real_datasets_indexed_zcurve']
    dataset_filenames_paths = ['dataset_filenames/large_aws_filenames.csv',
                               'dataset_filenames/medium_filenames.csv',
                               'dataset_filenames/small_filenames.csv',
                               'dataset_filenames/real_filenames.csv']
    for hdfs_index_path, dataset_filenames_path, master_path in zip(hdfs_index_paths, dataset_filenames_paths, master_paths):
        f = open(dataset_filenames_path)
        filenames = f.readlines()
        filenames = [f.strip() for f in filenames]

        for filename in filenames:
            mkdir_cmd = 'mkdir -p {}/{}'.format(master_path, filename)
            download_cmd = 'hdfs dfs -get {}/{}/_master.* {}/{}/'.format(hdfs_index_path, filename, master_path, filename)
            print(mkdir_cmd)
            print(download_cmd)
            os.system(mkdir_cmd)
            os.system(download_cmd)


if __name__ == '__main__':
    main()
