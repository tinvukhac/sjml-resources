import os


def main():
    print ('Execute batch indexing')
    f = open('large_datasets_filenames.csv')
    filenames = [line.strip() for line in f.readlines()]

    for filename in filenames:
        os.system ('spark-submit --master local[*] beast-uber-spark-0.5.0-SNAPSHOT.jar index file:///home/tvu032/sj_estimator/spatial_data_generators/large_datasets/{} file:///home/tvu032/sj_estimator/spatial_data_generators/indexed_large_datasets/{} gindex:rsgrove \'iformat:envelope(0,1,2,3)\' separator:, \'pcriterion:Size(1280k)\' -overwrite'.format(filename, filename))

    f.close()


if __name__ == '__main__':
    main()
