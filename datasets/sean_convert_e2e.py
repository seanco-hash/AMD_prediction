from oct_converter.readers import E2E
from os import listdir, mkdir
from os.path import isfile, join, isdir, basename
from multiprocessing import Process, Value
import argparse
import sys


# patients_dir = "/media/ron/Seagate Backup #3 Drive AI AMD-T/E2E/"
patients_dir = "/cs/labs/dina/seanco/hadassah/OCT"
# output_dir = "/media/ron/Seagate Backup #3 Drive AI AMD-T/output"
output_dir = "/cs/labs/dina/seanco/hadassah/OCT_output"


def get_patients():
    return sorted([join(patients_dir, d) for d in listdir(patients_dir) if isdir(join(patients_dir, d)) and d.startswith("AIA")])


def fetch_and_add(counter):
    # Update the counter atomically
    with counter.get_lock():
        old_value = counter.value
        counter.value += 1
    return old_value


def convert_e2e_to_png(patients, output_dir, counter):
    patient_prefix = 10
    E2E_suffix = -4
    patients_number = len(patients)
    next_paitent = fetch_and_add(counter)

    # Go over all the patients directories
    while next_paitent < patients_number:
        # Get next patient
        patient = patients[next_paitent]
        patient_name = basename(patient)

        # print(next_paitent, pid, patient_name)

        # Create directory for the patient
        patient_dir = join(output_dir, patient_name)
        if not isdir(patient_dir):
            mkdir(patient_dir)

        e2e_files = [join(patient, f) for f in listdir(patient) if isfile(join(patient, f)) and (f.endswith(".E2E") or f.endswith(".e2e"))]

        for filepath in e2e_files:

            # Create output directory for the current E2E file
            e2e_dest_dir = join(patient_dir, basename(filepath)[patient_prefix:E2E_suffix])
            if not isdir(e2e_dest_dir):
                mkdir(e2e_dest_dir)

            # Check if the slice is already converted
            slice_path = "{}.png".format(join(e2e_dest_dir, "slice"))
            if isfile(slice_path[:-4] + "_0" + slice_path[-4:]):
                continue
            else:
                print("Converting:", slice_path)

            # Save OCT slices as png
            file = E2E(filepath)
            oct_volumes = file.read_oct_volume()  # returns a list of all OCT volumes with additional metadata if available
            for volume in oct_volumes:
                volume.save(slice_path)

            # Save fundus image
            fundus_images = file.read_fundus_image()
            if len(fundus_images) > 0:
                fundus_images[0].save(join(e2e_dest_dir, "fundus.png"))
            # sys.stdout.flush()

        next_paitent = fetch_and_add(counter)


def main(args):
    patients = get_patients()
    patients = patients[args.s_idx:args.e_idx]
    counter = Value("i", 0)

    # Use multiprocessing to convert images from E2E to png
    # for pid in range(args.process_num):
    convert_e2e_to_png(patients, args.output_dir, counter)
        # Process(target=convert_e2e_to_png, args=(patients, args.output_dir, counter)).start()


class ExtractionParser(argparse.ArgumentParser):

    def __init__(self, **kwargs):
        super(ExtractionParser, self).__init__(**kwargs)
        self.add_argument("-i", "--input_dir",
                          type=str,
                          default=patients_dir,
                          help="Directory which contains all the OCTs in E2E format")
        self.add_argument("-o", "--output_dir",
                          type=str,
                          default=output_dir,
                          help="Directory which all the output images will be saved to")
        self.add_argument("--process_num",
                          type=int,
                          default=10,
                          help="The number of processes through which we will perform preprocessing")
        self.add_argument("--s_idx",
                          type=int,
                          default=0,
                          help="index to start processing dirs in dirs list.")
        self.add_argument("--e_idx",
                          type=int,
                          default=618,
                          help="index to start processing dirs in dirs list.")

    def parse_args(self, args=None, namespace=None):
        """ Parse the input arguments """
        args = super(ExtractionParser, self).parse_args(args, namespace)
        return args


def parse_args():
    parser = ExtractionParser()
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
