# https://github.com/fossasia/visdom
from collections import defaultdict
from os import listdir, stat
from os.path import join, isdir, isfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import dill as pickle
import argparse
import seaborn as sns


scans_dir = "/media/ron/Seagate Backup #3 Drive AI AMD-T/output/"
annotation_file = "/media/ron/Seagate Backup #3 Drive AI AMD-T/labels.csv"
pkl_dir = "/home/ron/workspace/AMD/utils/pkls"
colors = ["orange", "blue", "red", "purple", "yellow", "pink", "gold", "magenta", "black", "green"]

def plot_bar_chart(labels, values, title, x_label, y_label, bar_label=""):
    # Plot bar chart such that the labels are the x axis and values are the y axis.
    x = np.arange(len(labels))  # the label locations
    width = 0.5  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x, values, width, label=bar_label, color=colors[:len(labels)])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    if bar_label:
        ax.legend()

    ax.bar_label(rects1, padding=3)
    fig.tight_layout()
    plt.show()


def plot_graph(x, y, title, x_label, y_label, cfg, legends=None):
    # Plot graph.
    # if type(x[0]) is not list:
    #     x, y = [x], [y]

    # Add all functions to the graph.
    fig = plt.figure()
    # y = [y]
    for y1, color in zip(y, colors):
        x = np.arange(y1.shape[0])
        plt.plot(x, y1, color)

    if legends:
        plt.legend(legends)

    # Add titles and headlines
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
    file_name = cfg.MODEL.CHECKPOINT_NAME + '.png'
    plt.savefig(file_name)


def count_slices(patients_dir, pkl_dir, force=False):
    # Count amount of appearence for each slice number.
    slices_count = defaultdict(lambda: 0)
    fundus_count = 0

    patients = [join(patients_dir, d) for d in listdir(patients_dir) if isdir(join(patients_dir, d)) and d.startswith("AIA")]

    slices_count_pkl = join(pkl_dir, "slices_count.pkl")
    fundus_count_pkl = join(pkl_dir, "fundus_count.pkl")

    # Check if the we already saved the data and load it.
    if isfile(slices_count_pkl) and isfile(fundus_count_pkl) and (not force):
        slices_count = pickle.load(open(slices_count_pkl, "rb" ))
        fundus_count = pickle.load(open(fundus_count_pkl, "rb" ))
    else:
        # Go over all patients oct scans and count slices and fundus images in each scan.
        for patient in patients:
            oct_scans_dirs = [join(patient, d) for d in listdir(patient) if isdir(join(patient, d))]
            for oct_scan in oct_scans_dirs:
                print(oct_scan)
                slices = len([f for f in listdir(oct_scan) if isfile(join(oct_scan, f)) and f.startswith("slice")])
                fundus_count += len(listdir(oct_scan)) - slices
                slices_count[slices] += 1

        # Save the counting
        pickle.dump(slices_count, open(slices_count_pkl, "wb"))
        pickle.dump(fundus_count, open(fundus_count_pkl, "wb"))


    slices_count = list(sorted(slices_count.items(), key=lambda pair: pair[0]))
    slices_num = [slices_num for (slices_num, count) in slices_count]
    slices_amount = [count for (slices_num, count) in slices_count]

    # Plot bar chart for the slices.
    for i in range(0, len(slices_num), 10):
        plot_bar_chart(slices_num[i: i+10], slices_amount[i: i+10], "Slices appearences", "Slices amount", "Total appearances")
    print("Number of fundus images:", fundus_count)

    return slices_count, fundus_count


def get_features():
    # Get all the features in the annotation file.
    return [
        "P.I.D",
        "EYE",
        "DATE",
        "AMD STAGE",
        "DRUSEN",
        "SDD",
        "PED",
        "SCAR",
        "SRHEM",
        "IRF",
        "SRF",
        "IRORA",
        "IRORA location",
        "CRORA",
        "cRORA location"
    ]


def get_features_statistics_from_file(annotation_file):
    # Get statistics of feature from given annotation file.
    scans_labels = pd.read_csv(annotation_file)
    get_features_statistics(scans_labels)


def static_vars(**kwargs):
    # Decoratore for function static variables.
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


@static_vars(binary=["DRUSEN", "SDD", "PED", "SCAR", "SRHEM", "IRF", "SRF", "IRORA", "CRORA"])
@static_vars(multi=["AMD STAGE", "IRORA location", "cRORA location"])
@static_vars(binary_range=[0, 1])
@static_vars(multi_range=[i for i in range(13)])
def check_feature_mislabeld(feature, feature_value):
    # Check if the given feature value is whithin the valid range.
    if feature in check_feature_mislabeld.binary:
        return feature_value in check_feature_mislabeld.binary_range
    if feature in check_feature_mislabeld.multi:
        return feature_value in check_feature_mislabeld.multi_range
    return True


def check_label(feature_value, patient_id, feature, ind):
    # Check for missing data.
    if feature_value != feature_value:
        print("Missing data:", patient_id, feature, ind, feature_value)
    # Check for mislabeled data.
    elif not check_feature_mislabeld(feature, feature_value):
        print("Mislabeled data:", patient_id, feature, ind, feature_value)


def get_features_statistics(scans_labels):

    # Get statistics for each feature.
    features = get_features()
    print("#"*20, "\nFeatures statistics:")
    for feature in features:
        print("#"*20)
        print(feature)
        print(scans_labels[feature].describe())
    print("#"*30, "\n")

    features_count = defaultdict(lambda: defaultdict(lambda: 0))
    # Count for each feature number of appearences.
    for ind in scans_labels.index:
        patient_id = scans_labels["P.I.D"][ind]
        for feature in features:  # Go over all features of a patient
            feature_value = scans_labels[feature][ind]
            check_label(feature_value, patient_id, feature, ind)
            features_count[feature][feature_value] += 1

    # Plot bar chart for each feature.
    print("\nFeatures counts:")
    for feature,counts in features_count.items():
        if feature  == "P.I.D" or feature == "DATE":
            continue
        print(feature, list(counts.items()))
        feature_counts = list(sorted(counts.items(), key=lambda pair: pair[0]))
        features_classes = [label_num for (label_num, count) in feature_counts]
        features_amount = [count for (label_num, count) in feature_counts]
        plot_bar_chart(features_classes, features_amount, feature, "Label", "Amount")
        # sns.countplot(x=k, data=scans_labels)
        # plt.show()

    # Create correlation matrix.
    corr_mat = scans_labels.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr_mat, vmax=.8, square=True)
    plt.show()


def main(args):
    if args.count_slices:
        count_slices(args.scans_dir, args.pkl_dir, args.force)

    if args.count_features:
        get_features_statistics_from_file(args.annotation_file)


class VisualizerParser(argparse.ArgumentParser):

    def __init__(self, **kwargs):
        super(VisualizerParser, self).__init__(**kwargs)
        self.add_argument(
            "--scans_dir",
            default=scans_dir,
            help="Directory that contains all the OCT scans"
        )
        self.add_argument(
            "--count_slices",
            action="store_true",
            help="Get slices amount statistics"
        )
        self.add_argument(
            "--annotation_file",
            default=annotation_file,
            help="Directory that contains all the OCT scans"
        )
        self.add_argument(
            "--count_features",
            action="store_true",
            help="Get labels statistics"
        )
        self.add_argument(
            "-f", "--force",
            action="store_true",
            help="Force creation of new pickle"
        )
        self.add_argument(
            "--pkl_dir",
            default=pkl_dir,
            help="Directory that contains all saved data"
        )


    def parse_args(self, args=None, namespace=None):
        """ Parse the input arguments """
        args = super(VisualizerParser, self).parse_args(args, namespace)
        return args


def parse_args():
    parser = VisualizerParser()
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

