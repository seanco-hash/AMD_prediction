import argparse
from operator import mul
import os
from os import listdir, makedirs
from os.path import join, isdir, isfile, basename
import cv2
from cv2 import KeyPoint
import numpy as np
# import matplotlib.pyplot as plt
from scipy import ndimage
import sys
from skimage import feature, img_as_ubyte
import multiprocessing
import random
import pickle

from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)

# np.set_printoptions(threshold=sys.maxsize)
# plt.rcParams["figure.figsize"] = (20,20)

PROJ_DIR = '/cs/labs/dina/seanco/hadassah/dl_project/AMD/'
scans_dir = "/cs/labs/dina/seanco/hadassah/OCT_output"
output_dir = "/cs/labs/dina/seanco/hadassah/OCT_output/pre_process"


#
# def plot_comparison(original, rotated):
#     # Plot two images side by side
#     f = plt.figure()
#     f.add_subplot(1,2, 1)
#     plt.imshow(original)
#     f.add_subplot(1,2, 2)
#     plt.imshow(rotated)
#     plt.show()
#
#
# def plot_image(img, gray=False):
#     if gray:
#         img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     plt.imshow(img)
#     plt.show()
#
#
# def draw_kp(gray, kp):
#     # Draw key points on given image
#     img=cv2.drawKeypoints(gray,kp,gray,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#     plt.imshow(img)
#     plt.show()
#
#
# def compare_orig_and_rot(oct_scan, rot, poly, x, y):
#     # Plot original image with key points and the rotated image
#     kp = list()
#     for row, col in zip(x, y):
#         kp.append(KeyPoint(int(row), int(col), 1))
#     kp = np.array(kp)
#
#     oct_scan = cv2.drawKeypoints(oct_scan, kp, oct_scan)
#     red = (255, 0, 0)
#     thickness = 3
#     for i in range(oct_scan.shape[1]):
#         cv2.line(oct_scan, (i, int(poly(i))), (i+1, int(poly(i))), red, thickness)
#
#     plot_comparison(oct_scan, rot)


def fetch_and_add(counter):
	# Update the counter atomically
	with counter.get_lock():
		old_value = counter.value
		counter.value += 1
	return old_value


def get_patients(patients_dir):
    # Get all patient directories in the data directory
	return sorted([join(patients_dir, d) for d in listdir(patients_dir) if isdir(join(patients_dir, d)) and d.startswith("AIA")])


def crop_white_areas_sides(oct_scan):
    # Remove white areas from sides of oct scans
    gray_oct = cv2.cvtColor(oct_scan, cv2.COLOR_BGR2GRAY)
    left_col = 0
    right_col = gray_oct.shape[1] - 1
    white_pixel = 255

    while np.all(oct_scan[:, left_col] <= 15):
        left_col += 1

    while np.all(oct_scan[:, right_col] <= 15):
        right_col -= 1

    # Check for dashed white line on the sides
    low_row = 0
    top_row = gray_oct.shape[0] - 1
    while (gray_oct[low_row][left_col] == white_pixel) and (gray_oct[low_row+1][left_col] == white_pixel) and \
          (gray_oct[top_row][left_col] == white_pixel) and (gray_oct[top_row - 1][left_col] == white_pixel):
        left_col += 1

    while (gray_oct[low_row][right_col] == white_pixel) and (gray_oct[low_row+1][right_col] == white_pixel) and \
          (gray_oct[top_row][right_col] == white_pixel) and (gray_oct[top_row - 1][right_col] == white_pixel):
        right_col -= 1

    return oct_scan[:, left_col:right_col]


def get_canny_edge_detector_sigma(img, iter):
    mean = np.mean(img)

    if mean < 1.5:
        return 0.1 - (iter * 0.01)
    elif mean < 3:
        return 0.7 - (iter * 0.1)
    elif mean < 5:
        return 2 - (iter * 0.2)
    elif mean < 10:
        return 2.1 - (iter * 0.2)
    elif mean < 15:
        return 2.3 - (iter * 0.2)
    return 2.7 - (iter * 0.2)


def get_edge_image(gray_oct, iter):
    # Get the edge image of oct scan
    sigma = get_canny_edge_detector_sigma(gray_oct, iter)
    kernel = np.ones((5, 5), 'uint8')
    edge_img = cv2.dilate(gray_oct, kernel, iterations=3)
    edge_img = feature.canny(edge_img, sigma=sigma, low_threshold=60)
    edge_img = img_as_ubyte(edge_img)
    return edge_img


def create_key_points(edge_image):

    # x, y = list(), list()
    # Find key points in the oct scan edge image
    edge_pixels = np.nonzero(edge_image[:, :].T)
    x = edge_pixels[0][edge_pixels[0] % 20 == 0]
    y = edge_pixels[1][edge_pixels[0] % 20 == 0]
    # for col in range(0, edge_image.shape[1], 20):
    #     edge_points = np.nonzero(edge_image[:, col])[0]
    #     for row in edge_points:
    #             x.append(col)
    #             y.append(row)

    return x, y, len(x)


def get_folds_mid_values(x, y, kp_num, cols_num):
    # Divde the image cols space to folds
    folds_num = int(cols_num // 100) + 1
    mid_values = [[] for i in range(folds_num)]

    # Add for each fold the row values that related to it
    for i in range(kp_num):
        ind = int(x[i] // 100)
        mid_values[ind].append(y[i])
        if (x[i] % 100 >= 90 and ind < folds_num - 1):
            mid_values[ind+1].append(y[i])
        elif(x[i] % 100 <= 10) and ind > 0:
            mid_values[ind-1].append(y[i])

    # Calculate the mid value in the fold
    for i in range(folds_num):
        if mid_values[i]:
            mid_values[i] = (max(mid_values[i]) + (sum(mid_values[i]) // len(mid_values[i]))) // 2
        else:
            mid_values[i] = 0

    # Check if the last fold contains outliers
    if abs(mid_values[-1] - mid_values[-2]) > 15:
        mid_values[-1] = 0

    return mid_values


def calc_rot_angle(x, poly):
    # Calculate average rotation angle for a given x value and polynomial
    mid = (max(x) + min(x)) // 2
    angle = np.rad2deg(np.arctan2((poly(mid) - poly(mid-1)), 1))
    mid += 200
    angle += np.rad2deg(np.arctan2((poly(mid) - poly(mid-1)), 1))
    mid -= 400
    angle += np.rad2deg(np.arctan2((poly(mid) - poly(mid-1)), 1))

    return angle / 3


def change_contrast(img):

    # Converting image to LAB Color model
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))

    # Converting image from LAB Color model to RGB model
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def get_cropped_and_gray_oct(oct_scan):

    if np.mean(oct_scan) < 2:
        blured = cv2.medianBlur(oct_scan, 3)
        blured = change_contrast(blured)
        oct_scan = crop_black_areas(oct_scan, blured)
        gray_oct = cv2.cvtColor(change_contrast(oct_scan), cv2.COLOR_BGR2GRAY)
        return oct_scan, gray_oct

    blured = get_blured_image(oct_scan)
    # blured = cv2.GaussianBlur(blured, (5,5), 0)
    # plot_image(blured)
    oct_scan = crop_black_areas(oct_scan, blured)
    gray_oct = cv2.cvtColor(oct_scan, cv2.COLOR_BGR2GRAY)
    return oct_scan, gray_oct


def rotate_oct_scan(oct_path):
    try:
        # Remove white areas from the sides
        oct_scan = cv2.imread(oct_path)
        oct_scan = crop_white_areas_sides(oct_scan)
        oct_scan, gray_oct = get_cropped_and_gray_oct(oct_scan)

        same_kp, prev_kp, kp_num, iter = 0, 0, 0, 0
        while kp_num < 50 and same_kp < 10 and iter < 1000:
            edge_image = get_edge_image(gray_oct, iter)
            iter += 1
            # Create key points from edge image
            x, y, kp_num = create_key_points(edge_image)
            if kp_num == prev_kp:
                same_kp += 1
            else:
                prev_kp = kp_num
                same_kp = 0

        if kp_num < 10:
            return None
        mid_values = get_folds_mid_values(x, y, kp_num, edge_image.shape[1])

        # Select key points from retina bottom layer
        idx = np.asarray(range(kp_num))
        mid_values = np.asarray(mid_values)
        tmp_x = x // 100
        th = mid_values[tmp_x]
        idx = idx[th[:] != 0]
        th = th - 5
        idx = idx[y[idx] > th[idx]]
        y = y[idx]
        x = x[idx]

        # Befor Vectorization:
        # indices = list()
        # for i in range(kp_num):
        #     if y[i] > mid_values[int(x[i] // 100)] - 5 and mid_values[int(x[i] // 100)] != 0:
        #         indices.append(i)

        # Update key points after selection
        # x = np.array([x[i] for i in indices])
        # y = np.array([y[i] for i in indices])

        # Fit polynomial of order at most 2 to the selected key points
        deg = 2
        poly = np.poly1d(np.polyfit(x, y, deg))

        # Calculate rotation angle and rotate image
        angle = calc_rot_angle(x, poly)
        rot = ndimage.rotate(oct_scan, angle)

        # Plot comprasion between the original image to the rotated
        # if __debug__:
        #     compare_orig_and_rot(oct_scan, rot, poly, x, y)

        return rot

    except Exception as e:
        print(e)
        return None


def get_crop_black_areas_top_and_bottom(blured_oct_scan):
    bool_arr = np.all(blured_oct_scan[:, :, 0] <= 15, axis=1)
    top_row = np.argmax(bool_arr == False)
    bottom_row = (len(bool_arr) - 1) - np.argmax(bool_arr[::-1] == False)

    # Before Vectorization:
    # bottom_row = blured_oct_scan.shape[0] - 1
    # while np.all(blured_oct_scan[top_row, :] <= 15):
    #     top_row += 1
    #
    # while np.all(blured_oct_scan[bottom_row, :] <= 15):
    #     bottom_row -= 1

    return top_row, bottom_row


def crop_black_areas(oct_scan, blured_oct_scan):
    # Crop black areas from the top and the bottom of the image
    top_row, bottom_row = get_crop_black_areas_top_and_bottom(blured_oct_scan)
    # left_col = 0
    # right_col = oct_scan.shape[1] - 1

    # Crop the black areas on the sides of the image
    bool_arr = np.all(blured_oct_scan[:, :, 0] <= 10, axis=0)
    left_col = np.argmax(bool_arr == False)
    right_col = (len(bool_arr) - 1) - np.argmax(bool_arr[::-1] == False)

    # while np.all(blured_oct_scan[:, left_col] <= 10):
    #     left_col += 1
    #
    # while np.all(blured_oct_scan[:, right_col] <= 10):
    #     right_col -= 1

    return oct_scan[top_row:bottom_row, left_col:right_col]


def get_blured_image(img):
    if np.mean(img) < 2:
        return cv2.medianBlur(img, 3)
    if np.mean(img) < 4:
        return cv2.medianBlur(img, 5)
    if np.mean(img) < 9:
        return cv2.medianBlur(img, 13)
    if np.mean(img) < 11:
        return cv2.medianBlur(img, 19)
    return cv2.medianBlur(img, 25)


def center_oct_scan(rot):
    blured_rot = get_blured_image(rot)
    top, bottom = get_crop_black_areas_top_and_bottom(blured_rot)
    center = rot[top:bottom, :]

    return center


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def save_obj(obj, name):
    with open(PROJ_DIR + 'obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol=4)


def load_obj(name):
    with open(PROJ_DIR + 'obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def clear_to_convert_list():
    to_convert = load_obj('slices_to_pre_process')
    new_list = []
    for _slice in to_convert:
        if isfile(_slice):
            continue
        new_list.append(_slice)

    save_obj(new_list, 'slices_to_pre_process')
    print(len(new_list))
    print(len(to_convert))


def preprocess_image_by_path(slices, out_dir, pid):
    # "/cs/labs/dina/seanco/hadassah/OCT_output/pre_process"
    i = 0
    for _slice in slices:
        if isfile(_slice):
            continue
        tmp = _slice.split('/')
        tmp = tmp[:7] + tmp[8:]
        in_path = '/'.join(tmp)
        rot = rotate_oct_scan(in_path)
        if rot is None:
            continue
        center = center_oct_scan(rot)
        cv2.imwrite(_slice, center)
        i += 1
        if i > 10:
            print("Convert ", _slice)
            i = 0

    print('finished my work ', pid)


def preprocess_images(patients, output_dir, pid):

    # Check if the output dir exist
    if not isdir(output_dir):
        makedirs(output_dir)

    patients_number = len(patients)
    # next_patient = fetch_and_add(counter)
    # print(counter)
    # sys.stdout.flush()
    to_convert = []
    converted = []
    i = 0
    # while next_patient < patients_number: # Go over all patients
    for patient in patients:
        # Get patient to preprocess
        patient_name = basename(patient)
        # print(patient_name + " by process " + str(pid))
        # sys.stdout.flush()

        # if patient_name != "AIA 03029":
        #     next_patient = fetch_and_add(counter)
        #     continue

        # Go over all patient's oct scans
        oct_scans_dirs = [join(patient, d) for d in listdir(patient) if isdir(join(patient, d))]
        for oct_scan in oct_scans_dirs:
            # Check if the output directory exists
            oct_scan_output_path = join(output_dir, patient_name, basename(oct_scan))

            # if oct_scan != "datasets/OCT/AIA 03029/OD 14.06.2017":
            #     continue

            if not isdir(oct_scan_output_path):
                makedirs(oct_scan_output_path)

            # Go over all slices in a given oct scan
            slices = [join(oct_scan, f) for f in listdir(oct_scan) if isfile(join(oct_scan, f)) and f.startswith("slice")]
            for slice in slices: #  Perform preprocessing

                    slice_path = join(oct_scan_output_path, basename(slice))
                    # print(patient_name, basename(oct_scan), basename(slice))
                    # sys.stdout.flush()

                    # if basename(slice) != "slice_9.png":
                    #     continue

                    if isfile(slice_path):
                        converted.append(slice_path)
                        i += 1
                        if i > 1000:
                            print("to_convert:", len(to_convert))
                            print("converted:", len(converted))
                            i = 0
                        continue
                    else:
                        to_convert.append(slice_path)
                        i += 1
                        if i > 1000:
                            print("to_convert:", len(to_convert))
                            print("converted:", len(converted))
                            i = 0
                        continue

                    # Rotate oct slice and get ROI
                    rot = rotate_oct_scan(slice)
                    if rot is None:
                        continue
                    center = center_oct_scan(rot)
                    cv2.imwrite(slice_path, center)

        # next_patient = fetch_and_add(counter)
    print(len(to_convert))
    save_obj(to_convert, 'slices_to_pre_process')
    print('finish my patients ' + str(pid))


def run_multi_process(list_to_process, process_func):
    # preprocess_image_by_path(list_to_process, 'bla', 0)
    available_cpus = len(os.sched_getaffinity(0))
    slices_each_thread = int(len(list_to_process) / (available_cpus - 1))
    try:
        start_idx = 0
        for pid in range(available_cpus):
            end_idx = min(start_idx + slices_each_thread, len(list_to_process))
            multiprocessing.Process(target=process_func, args=(list_to_process[start_idx: end_idx],
                                                                    args.output_dir, pid)).start()
            start_idx += slices_each_thread
            print('process: ' + str(pid))
    except Exception as e:
        print('error')
        print(e)
        exit(1)


def run_preprocess_by_slices():
    to_convert = load_obj('slices_to_pre_process')
    # half = len(to_convert) // 2
    # to_convert = to_convert[:half]
    run_multi_process(to_convert, preprocess_image_by_path)


def run_preprocess_by_patients(args):
    patients = get_patients(args.input_dir)
    random.shuffle(patients)
    # already_processed = get_patients(args.output_dir)
    # patients = patients[len(already_processed):]
    run_multi_process(patients, preprocess_images)


def main(args):
    # clear_to_convert_list()
    run_preprocess_by_slices()


class PreprocessingParser(argparse.ArgumentParser):

    def __init__(self, **kwargs):
        super(PreprocessingParser, self).__init__(**kwargs)
        self.add_argument("-i", "--input_dir",
                          type=str,
                          default=scans_dir,
                          help="Directory which contains all the OCTs in png format")
        self.add_argument("-o", "--output_dir",
                          type=str,
                          default=output_dir,
                          help="Directory which all the output images will be saved to")
        self.add_argument("-p", "--process_num",
                          type=int,
                          default=10,
                          help="The number of processes through which we will perform preprocessing")


    def parse_args(self, args=None, namespace=None):
        """ Parse the input arguments """
        args = super(PreprocessingParser, self).parse_args(args, namespace)
        return args


def parse_args():
    parser = PreprocessingParser()
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
