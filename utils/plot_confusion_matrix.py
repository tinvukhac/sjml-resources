import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import itertools
from mpl_toolkits.axes_grid1 import ImageGrid


def plot_two_matrices(confusion_matrix_values, titles):
    classes = ['BNLJ', 'PBSM', 'DJ', 'RepJ']

    fig = plt.figure()
    grid = ImageGrid(fig, 111,  # as in plt.subplot(111)
                     nrows_ncols=(1, 2),
                     axes_pad=0.15,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="7%",
                     cbar_pad=0.15,
                     )

    for n, ax in enumerate(grid[:2]):
        # cm = np.random.random((2, 2))
        cm = confusion_matrix_values[n]
        im = ax.imshow(cm, vmin=0, vmax=1, cmap=plt.cm.Blues)
        ax.set_title("{}".format(titles[n]))  # ax.___ instead of plt.___
        tick_marks = np.arange(4)
        ax.set_xticks(tick_marks)  # Warning: different signature for [x|y]ticks in pyplot and OO interface
        ax.set_xticklabels(classes, rotation=0)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes)

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], '.3f'),
                    horizontalalignment="center",
                    color="black")
            if confusion_matrix_values[n][i][j] > 0.7:
                ax.text(j, i, format(cm[i, j], '.3f'),
                        horizontalalignment="center",
                        color="white")

        ax.set_ylabel('Actual best algorithm')
        ax.set_xlabel('Predicted algorithm')

    # fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    fig.colorbar(im, cax=ax.cax)
    fig.savefig('../figures/confusion_matrix.png', bbox_inches='tight')

    plt.show()


def main():
    print('Plot confusion matrix')

    # Note: somehow you need to run this file on terminal.
    # I always get FileNotFoundError exception even the file path is correct

    # Remove empty lines from Alberto's data
    # f = open('../data/temp/algorithm_selection_b3_updated_5_31.alberto.csv')
    # output_f = open('../data/temp/algorithm_selection_b3_updated_5_31.csv', 'w')
    #
    # lines = f.readlines()
    #
    # for line in lines:
    #     if len(line.strip()) > 0:
    #         output_f.writelines('{}\n'.format(line.strip()))
    #
    # output_f.close()
    # f.close()


    # Plot confusion matrix
    # df = pd.read_csv('../data/temp/algorithm_selection_b3_updated_5_31.csv', header=0)
    # y_test = df['y_test']
    # y_pred = df['y_pred']
    # cm = confusion_matrix(y_test, y_pred)
    #
    # class_names = ['BNLJ', 'PBSM', 'DJ', 'RepJ']
    # cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3, 4], normalize='true')
    # print(cm)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    # disp.plot(cmap=plt.cm.Blues)
    # plt.xlabel('Predicted algorithm', fontsize=16)
    # plt.ylabel('Actual best algorithm', fontsize=16)
    # plt.savefig('../figures/confusion_matrix_with_normalization_b3.png')

    confusion_matrix_values = []

    # Compute fist confusion matrix
    df = pd.read_csv('../data/temp/algorithm_selection_b3_updated_5_31.csv', header=0)
    y_test = df['y_test']
    y_pred = df['y_pred']
    confusion_matrix_values.append(confusion_matrix(y_test, y_pred, labels=[1, 2, 3, 4], normalize='true'))

    # Compute second confusion matrix
    df = pd.read_csv('../data/temp/algorithm_selection_m3_fs3_v3.csv', header=0)
    y_test = df['y_test']
    y_pred = df['y_pred']
    confusion_matrix_values.append(confusion_matrix(y_test, y_pred, labels=[1, 2, 3, 4], normalize='true'))

    SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    titles = ['B3{}'.format('2'.translate(SUB)), 'M3']

    plot_two_matrices(confusion_matrix_values, titles)


if __name__ == '__main__':
    main()
