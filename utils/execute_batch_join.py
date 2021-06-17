import os


def main():
    print ('Execute batch join with list of mbrs')

    # input_f = open('deep_join/data/histogram_16_16_mbrs.csv')
    # lines = input_f.readlines()
    #
    # dataset1 = 'diagonal_001.csv'
    # dataset2 = 'gaussian_001.csv'
    #
    # for line in lines:
    #     filter_mbr = line.strip()
    #     # print(
    #     #     'spark-submit --master local[*] beast-tv/target/beast-uber-spark-0.2.3-RC2-SNAPSHOT.jar sj small_datasets/{} small_datasets/{} output.csv filtermbr:{} \'iformat:envelope(0,1,2,3)\' separator:, -overwrite >> result_count.txt'.format(
    #     #         dataset1, dataset2, filter_mbr))
    #     print(
    #         'spark-submit --master local[*] beast-tv/target/beast-uber-spark-0.2.3-RC2-SNAPSHOT.jar sj small_datasets/{} small_datasets/{} output.csv filtermbr:{} \'iformat:envelope(0,1,2,3)\' separator:, -overwrite >> result_count.txt'.format(
    #             dataset1, dataset2, filter_mbr))

    input_f = open('large_datasets.csv')

    datasets = [line.strip() for line in input_f.readlines()]

    for i in range(len(datasets)):
        for j in range(i + 1, len(datasets)):
            dataset1 = datasets[i]
            dataset2 = datasets[j]
            if dataset1 != dataset2:
                os.system (
                    'hadoop jar spatialhadoop-2.4.3-SNAPSHOT-uber.jar dj sj_estimator/large_datasets_all/{} sj_estimator/large_datasets_all/{} repartition:no direct-join:no heuristic-repartition:no shape:rect dj/dj_{}_{}_noindex -no-output -overwrite > dj_logs/dj_{}_{}_noindex.log 2>&1 '.format(
                        dataset1, dataset2, dataset1.split('.')[0], dataset2.split('.')[0], dataset1.split('.')[0],
                        dataset2.split('.')[0]))


if __name__ == '__main__':
    main()
