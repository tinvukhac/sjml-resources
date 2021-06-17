import glob
import os
import pandas as pd


def main():
    print ('Concatenate tabular data')
    # os.system('rm ../data/tabular/tabular_all.csv')
    # tabular_data_files = glob.glob("../data/tabular/*.csv")
    # df = pd.concat((pd.read_csv(f, header=0) for f in tabular_data_files))
    # df.to_csv('../data/tabular/tabular_all.csv', index=0)

    os.system('rm ../data/spatial_quality_metrics/quality_metrics_all.csv')
    quality_metrics_data_files = glob.glob("../data/spatial_quality_metrics/quality_metrics*.csv")
    df = pd.concat((pd.read_csv(f, header=0) for f in quality_metrics_data_files))
    df.to_csv('../data/spatial_quality_metrics/quality_metrics_all.csv', index=0)

    tabular_df = pd.read_csv('../data/tabular/tabular_all.csv', delimiter='\\s*,\\s*', header=0)
    quality_metrics_df = pd.read_csv('../data/spatial_quality_metrics/quality_metrics_all.csv', delimiter='\\s*,\\s*', header=0)
    tabular_quality_metrics_df = pd.merge(tabular_df, quality_metrics_df, on='dataset_name')
    tabular_quality_metrics_df.to_csv('../data/tabular/tabular_all_v2.csv', index=False)

    os.system('rm ../data/spatial_quality_metrics/non_indexed_quality_metrics_all.csv')
    quality_metrics_data_files = glob.glob("../data/spatial_quality_metrics/non_indexed*.csv")
    df = pd.concat((pd.read_csv(f, header=0) for f in quality_metrics_data_files))
    df.to_csv('../data/spatial_quality_metrics/non_indexed_quality_metrics_all.csv', index=0)

    tabular_df = pd.read_csv('../data/tabular/tabular_all.csv', delimiter='\\s*,\\s*', header=0)
    quality_metrics_df = pd.read_csv('../data/spatial_quality_metrics/non_indexed_quality_metrics_all.csv', delimiter='\\s*,\\s*',
                                     header=0)
    tabular_quality_metrics_df = pd.merge(tabular_df, quality_metrics_df, on='dataset_name')
    tabular_quality_metrics_df.to_csv('../data/tabular/tabular_all_v2_non_indexed.csv', index=False)


if __name__ == '__main__':
    main()
