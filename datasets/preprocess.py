import argparse
import math
from os import listdir, makedirs
from os.path import join, isdir, isfile, basename
import cv2
from cv2 import KeyPoint
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import sys
from skimage import feature, img_as_ubyte
import multiprocessing

np.set_printoptions(threshold=sys.maxsize)
plt.rcParams["figure.figsize"] = (20,20)


scans_dir = "/cs/labs/dina/seanco/hadassah/OCT_output"
output_dir = "/cs/labs/dina/seanco/hadassah/OCT_output/pre_process"


def plot_comparison(original, rotated):
    # Plot two images side by side
    f = plt.figure()
    f.add_subplot(1,2, 1)
    plt.imshow(original)
    f.add_subplot(1,2, 2)
    plt.imshow(rotated)
    plt.show()


def plot_image(img):
    plt.imshow(img)
    plt.show()


def draw_kp(gray, kp):
    # Draw key points on given image
    img=cv2.drawKeypoints(gray,kp,gray,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(img)
    plt.show()


def compare_orig_and_rot(oct_scan, rot, poly, x, y):
    # Plot original image with key points and the rotated image
    kp = list()
    for row, col in zip(x, y):
        kp.append(KeyPoint(int(row), int(col), 1))
    kp = np.array(kp)

    oct_scan = cv2.drawKeypoints(oct_scan, kp, oct_scan)
    red = (255, 0, 0)
    thickness = 3
    for i in range(oct_scan.shape[1]):
        cv2.line(oct_scan, (i, int(poly(i))), (i+1, int(poly(i))), red, thickness)

    plot_comparison(oct_scan, rot)


def fetch_and_add(counter):
	# Update the counter atomically
	with counter.get_lock():
		old_value = counter.value
		counter.value += 1
	return old_value


def get_patients(patients_dir):
    # Get all patient directories in the data directory
	return sorted([join(patients_dir, d) for d in listdir(patients_dir) if isdir(join(patients_dir, d)) and d.startswith("AIA")])


def crop_white_areas_sides(oct_scan, gray_oct):
    # Remove white areas from sides of oct scans
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


    return oct_scan[:, left_col:right_col], gray_oct[:, left_col:right_col]


def get_canny_edge_detector_sigma(img, iter):
    mean = np.mean(img)

    if mean < 1.5:
        return 0.1 - (iter * 0.01)
    elif mean < 3:
        return 0.7 - (iter * 0.1)
    elif mean < 5:
        return 2 - (iter * 0.2)
    elif mean < 10:
        return 2.5 - (iter * 0.2)
    elif mean < 15:
        return 2.7 - (iter * 0.2)
    return 3 - (iter * 0.2)


def get_edge_image(gray_oct, iter):
    # Get the edge image of oct scan
    sigma = get_canny_edge_detector_sigma(gray_oct, iter)
    kernel = np.ones((5, 5), 'uint8')
    edge_img = cv2.dilate(gray_oct, kernel, iterations=3)
    edge_img = feature.canny(edge_img, sigma=sigma)
    edge_img = img_as_ubyte(edge_img)
    return edge_img


def create_key_points(edge_image):

    x, y = list(), list()
    # Find key points in the oct scan edge image
    for col in range(0, edge_image.shape[1], 20):
        edge_points = np.nonzero(edge_image[:, col])[0]
        for row in edge_points:
                x.append(col)
                y.append(row)

    return x, y, len(x)


def get_mid_values(x, y, kp_num, cols_num):
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
            mid_values[i] = (max(mid_values[i]) + min(mid_values[i])) // 2
        else:
            mid_values[i] = 0

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


def rotate_oct_scan(oct_path):
    oct_scan = cv2.imread(oct_path)

    # Convert oct scan to gray image and remove white lines from the sides
    gray_oct = cv2.cvtColor(oct_scan, cv2.COLOR_BGR2GRAY)
    try:
        oct_scan, gray_oct = crop_white_areas_sides(oct_scan, gray_oct)
    except:
        return None, None
    # denoised_1 = cv2.fastNlMeansDenoising(gray_oct, h=5, templateWindowSize=7, searchWindowSize=21)
    kp_num, iter = 0, 0
    while kp_num < 50:
        # edge_image = get_edge_image(denoised_1, iter)
        edge_image = get_edge_image(gray_oct, iter)
        iter += 1
        # Create key points from edge image
        x, y, kp_num = create_key_points(edge_image)

    mid_values = get_mid_values(x, y, kp_num, edge_image.shape[1])

    # Select key points from retina bottom layer
    indices = list()
    for i in range(kp_num):
        if y[i] > mid_values[int(x[i] // 100)] - 5:
            indices.append(i)

    # Update key points after selection
    y = np.array([y[i] for i in indices])
    x = np.array([x[i] for i in indices])

    # Fit polynomial of order at most 2 to the selected key points
    deg = 2
    poly = np.poly1d(np.polyfit(x, y, deg))

    # Calculate rotation angle and rotate image
    angle = calc_rot_angle(x, poly)
    rot = ndimage.rotate(oct_scan, angle)
    edge_image = ndimage.rotate(edge_image, angle)

    # Plot comprasion between the original image to the rotated
    if __debug__:
        compare_orig_and_rot(oct_scan, rot, poly, x, y)

    return rot, edge_image


def crop_black_areas(oct_scan):
    top_row = 0
    bottom_row = oct_scan.shape[0] - 1
    left_col = 0
    right_col = oct_scan.shape[1] - 1

    # Crop the black areas on the sides of the image
    while np.all(oct_scan[top_row, :] <= 15):
        top_row += 1

    while np.all(oct_scan[bottom_row, :] <= 15):
        bottom_row -= 1

    while np.all(oct_scan[:, left_col] <= 10):
        left_col += 1

    while np.all(oct_scan[:, right_col] <= 10):
        right_col -= 1

    return oct_scan[top_row:bottom_row, left_col:right_col]


def center_oct_scan(rot, edge_image):

    # Find the max and min row of pixels in the edge image
    min_row = np.amin(np.argwhere(edge_image > 0), axis=0)[0]
    max_row = np.amax(np.argwhere(edge_image > 0), axis=0)[0]

    neighbour = 10
    # Make the margin bigger in narrow ROI
    if (max_row - min_row) < 100:
        neighbour = 130 - (max_row - min_row)

    # Crop ROI
    neighbour = min(neighbour, min_row - 0, edge_image.shape[0]-max_row) * 2
    center = np.zeros((max_row-min_row+neighbour, rot.shape[1], rot.shape[2]), dtype=rot.dtype)
    neighbour //= 2

    center = rot[min_row-neighbour:max_row+neighbour]
    center = crop_black_areas(center)
    return center


def align_two_frames(im1, im2, r):
    orb = cv2.ORB_create(500)
    # Searching feature points on denoised image
    # denoised_1 = cv2.fastNlMeansDenoising(im1, h=2, templateWindowSize=7, searchWindowSize=21)
    # denoised_2 = cv2.fastNlMeansDenoising(im2, h=2, templateWindowSize=7, searchWindowSize=21)

    keypoints1, descriptors1 = orb.detectAndCompute(im1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2, None)

    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    matches.sort(key=lambda x: x.distance, reverse=False)
    numGoodMatches = int(len(matches) * 0.15)
    matches = matches[:numGoodMatches]

    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    # height, width, channels = im2.shape
    height, width = im2.shape
    # apply transformation on the original image
    im1Reg = cv2.warpPerspective(r, h, (width, height))
    cv2.imshow('im', im1Reg)
    cv2.waitKey(0)
    return im1Reg


def find_corner(arr, direction):
    non_zero_idx = np.nonzero(arr > 0)
    double_step = direction * 2
    if direction > 0:
        cur_idx = 0
    else:
        cur_idx = len(non_zero_idx[0]) - 1
    cur_pix = non_zero_idx[0][cur_idx]
    while 0 <= (cur_pix + double_step) < len(arr) and\
            (arr[cur_pix + direction] <= 0 or arr[cur_pix + double_step] <= 0):
        cur_idx += direction
        cur_pix = non_zero_idx[0][cur_idx]

    # if 1 < cur_pix < len(arr) - 2:
    #     return cur_pix
    return cur_pix


def rot_first_im(im):
    top_left = find_corner(im[:, 10], 1)
    top_right = find_corner(im[:, -10], 1)
    bottom_right = find_corner(im[:, -10], -1)
    bottom_left = find_corner(im[:, 10], -1)

    if top_left != -1 and top_right != -1 and bottom_right != -1 and bottom_left != -1:
        center_x = int(im.shape[1] / 2)
        center_y = int((top_right + top_left + bottom_left + bottom_right) / 4)

        x_vec = im.shape[1]
        y_vec = top_right - top_left
        angle = math.atan2(y_vec, x_vec)
        rot_mat = cv2.getRotationMatrix2D((center_x, center_y), -angle, 1.0)
        result = cv2.warpAffine(im, rot_mat, im.shape[1::-1], flags=cv2.INTER_LINEAR)
        # cv2.imshow('im', im)
        # cv2.waitKey(0)
        # cv2.imshow('res', result)
        # cv2.waitKey(0)
        return result
    return im


def process_first_slice(slice_path):
    try:
        first_slice = cv2.imread(slice_path)
        gray_oct = cv2.cvtColor(first_slice, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('gray', gray_oct)
        # cv2.waitKey(0)
        first_slice, gray_oct = crop_white_areas_sides(first_slice, gray_oct)
        # cv2.imshow('gray_crop', gray_oct)
        # cv2.waitKey(0)

        # gray_oct = cv2.fastNlMeansDenoising(gray_oct, h=1, templateWindowSize=7, searchWindowSize=21)
        # cv2.imshow('denoised', gray_oct)
        # cv2.waitKey(0)
        rotated = rot_first_im(gray_oct)
        return rotated
    except Exception as e:
        print(e)
        return None


def preprocess_images(patients, output_dir, counter):

    # Check if the output dir exist
    if not isdir(output_dir):
        makedirs(output_dir)

    patients_number = len(patients)
    next_patient = fetch_and_add(counter)

    while next_patient < patients_number: # Go over all patients
        # Get patient to preprocess
        patient = patients[next_patient]
        patient_name = basename(patient)

        # if patient_name != "AIA 03859":
        #     print(patient_name)
        #     next_patient = fetch_and_add(counter)
        #     continue

        # Go over all patient's oct scans
        oct_scans_dirs = [join(patient, d) for d in listdir(patient) if isdir(join(patient, d))]
        for oct_scan in oct_scans_dirs:
            oct_scan = '/cs/labs/dina/seanco/hadassah/OCT_output/AIA 03501/OD 18.04.2010'
            # Check if the output directory exists
            oct_scan_output_path = join(output_dir, patient_name, basename(oct_scan))

            # if oct_scan_output_path != "/media/ron/Seagate Backup #3 Drive AI AMD-T/preprocessed/AIA 03335/OD 07.11.2018":
            # if oct_scan_output_path != "/media/ron/Seagate Backup #3 Drive AI AMD-T/preprocessed/AIA 03326/OS 02.09.2020":
            # if oct_scan_output_path != "/media/ron/Seagate Backup #3 Drive AI AMD-T/preprocessed/AIA 03501/OD 18.04.2010":
            # if oct_scan_output_path != "/media/ron/Seagate Backup #3 Drive AI AMD-T/preprocessed/AIA 03859/OD 01.04.2019":
                # continue

            if not isdir(oct_scan_output_path):
                makedirs(oct_scan_output_path)

            # Go over all slices in a given oct scan
            slices = [join(oct_scan, f) for f in listdir(oct_scan) if isfile(join(oct_scan, f)) and f.startswith("slice")]
            # prev_im = None
            # idx = 0
            # while prev_im is None and idx < len(slices):
            #     prev_im = process_first_slice(slices[idx])
            #     idx += 1
            # slices = slices[idx::]

            for slice in slices: #  Perform preprocessing
                slice_path = join(oct_scan_output_path, basename(slice))
                print(next_patient, patient_name, basename(oct_scan), basename(slice))

                # if basename(slice) != "slice_19.png":
                #     continue

                # if isfile(slice_path):
                    # continue

                # Rotate oct slice and get ROI
                rot, edge_image = rotate_oct_scan(slice)

                # oct_scan = cv2.imread(slice)
                # Convert oct scan to gray image and remove white lines from the sides
                # gray_oct = cv2.cvtColor(oct_scan, cv2.COLOR_BGR2GRAY)
                # try:
                #     oct_scan, gray_oct = crop_white_areas_sides(oct_scan, gray_oct)
                # except:
                #     continue
                # gray_oct = cv2.fastNlMeansDenoising(gray_oct, h=10, templateWindowSize=7, searchWindowSize=21)

                if rot is None:
                    continue
                center = center_oct_scan(rot, edge_image)
                # cv2.imwrite(slice_path, aligned_im)
                # cv2.imshow('prev', prev_im)
                # cv2.imshow('cur', aligned_im)
                # prev_im = aligned_im

        next_patient = fetch_and_add(counter)


def main(args):
    counter = multiprocessing.Value("i", 0)

    patients = get_patients(args.input_dir)
    # Use multiprocessing to convert images from E2E to png
    try:
    #     for pid in range(args.process_num):
    #         multiprocessing.Process(target=preprocess_images, args=(patients, args.output_dir, counter)).start()
        preprocess_images(patients, args.output_dir, counter)
    except:
        exit(1)


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
        self.add_argument("--process_num",
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
