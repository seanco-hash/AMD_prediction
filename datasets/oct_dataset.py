from collections import defaultdict
import copy
import re
from numpy import NaN
import numpy as np
from os import listdir, stat
from os.path import join, isdir, basename, isfile
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class PatientEyeLabels():
    """
    Class that represents single OCT scan labels
    """
    date_ind = 2

    def __init__(self, id, eye):
        self.id = id
        self.eye = eye
        self.labels = defaultdict(list)
        self.shared_labels = [id, eye, NaN]

    def add_labels(self, date, patient_labels):
        self.labels[date] = patient_labels

    def create_shared_labels(self):
        # Get the labels of the first and last get all the feature that are the same
        first_oct, second_oct = self.labels.values()

        for i in range(len(first_oct)):
            if first_oct[i] == second_oct[i]:
                self.shared_labels.append(first_oct[i])
            else:
                self.shared_labels.append(NaN)

    def get_shared_labels(self, date) -> list:
        labels = copy.deepcopy(self.shared_labels)
        labels[PatientEyeLabels.date_ind] = date
        return labels

    def check_date(self, date) -> bool:
        return date in self.labels

    def __str__(self) -> str:
        return f"{self.id} {self.eye}"

    def __repr__(self) -> str:
        return self.__str__()


class OCTDataset(Dataset):
    """
    Dataset object on the OCT database
    """
    @staticmethod
    def _get_patients_labels(patient_row):
        return patient_row[['AMD_STAGE', 'DRUSEN', 'SDD', 'PED', 'SCAR', 'SRHEM', 'IRF', 'SRF', 'IRORA', 'IRORA_location', 'CRORA', 'cRORA_location']].values

    @staticmethod
    def get_feat_idx(cfg):
        if cfg.TARGET_FEATURE == 'AMD_STAGE':
            return 0
        elif cfg.TARGET_FEATURE == 'DRUSEN':
            return 1
        elif cfg.TARGET_FEATURE == 'IRORA':
            return 8
        return 0

    @staticmethod
    def filter_scans_by_labels(cfg, labels):
        if cfg.TARGET_FEATURE == 'AMD_STAGE':
            if labels[3] is NaN or int(labels[3]) > 2 or (cfg.MODEL.NUM_CLASSES == 2 and int(labels[3]) == 0):
                return False
        elif cfg.TARGET_FEATURE == 'DRUSEN':
            if labels[4] is NaN:
                return False
        elif cfg.TARGET_FEATURE == 'IRORA':
            if labels[11] is NaN:
                return False
        return True

    @staticmethod
    def filter_dataframe(cfg, scans_labels):
        scans_labels[cfg.TARGET_FEATURE] = pd.to_numeric(scans_labels[cfg.TARGET_FEATURE])
        if cfg.TARGET_FEATURE == 'AMD_STAGE':
            scans_labels = scans_labels[scans_labels.AMD_STAGE < 3]
            if cfg.MODEL.NUM_CLASSES == 2:
                scans_labels = scans_labels[scans_labels.AMD_STAGE > 0]
            return scans_labels
        if cfg.TARGET_FEATURE == 'DRUSEN':
            scans_labels = scans_labels[scans_labels.DRUSEN < 2]
        elif cfg.TARGET_FEATURE == 'IRORA':
            scans_labels = scans_labels[scans_labels.IRORA < 2]
        return scans_labels

    @staticmethod
    def get_labels_by_order(labels):
        key1, key2 = labels.keys()
        key11 = key1.split('.')
        key22 = key2.split('.')
        for i in range(1, len(key1)):
            if key11[-i] < key22[-i]:
                return labels[key1], labels[key2]
            elif key11[-i] > key22[-i]:
                return labels[key2], labels[key1]
        return None

    @staticmethod
    def _get_annotations(annotation_file, img_dir, train, interpolate_data, cfg):
        """
        Complete automated data labels for all the scans
        """
        scans_labels = pd.read_csv(annotation_file)
        scans_labels = OCTDataset.filter_dataframe(cfg, scans_labels)
        if interpolate_data:
            # Get all annotated data.
            patients_labels = dict()
            for ind, row in scans_labels.iterrows():
                id = row["P.I.D"]
                eye = row["EYE"]
                patient_id = f"{id} {eye}"

                if patient_id not in patients_labels:
                    patients_labels[patient_id] = PatientEyeLabels(id, eye)

                # Add OCT scan to the patient's labels.
                date = row["DATE"]
                patients_labels[patient_id].add_labels(date, OCTDataset._get_patients_labels(row))

            patients_to_remove = []
            for patient in patients_labels.values():
                feat_idx = OCTDataset.get_feat_idx(cfg)
                if len(patient.labels) < 2:
                    patients_to_remove.append(f"{patient.id} {patient.eye}")
                else:
                    first_oct, second_oct = OCTDataset.get_labels_by_order(patient.labels)
                    if first_oct[feat_idx] > second_oct[feat_idx]:
                        patients_to_remove.append(f"{patient.id} {patient.eye}")
                    else:
                        patient.create_shared_labels()

            for patient in patients_to_remove:
                patients_labels.pop(patient)

            # Add unlabeled data
            oct_scans = list()
            patients = [join(img_dir, d) for d in listdir(img_dir) if isdir(join(img_dir, d)) and d.startswith("AIA")]
            for patient in patients:
                id = basename(patient)
                oct_scans_dirs = [d for d in listdir(patient) if isdir(join(patient, d))]
                for oct_scan in oct_scans_dirs:  # GO over all OCT images of single patient
                    tmp = oct_scan.split()
                    if len(tmp) == 2:
                        eye, date = tmp
                    else:
                        continue
                    patient_id = f"{id} {eye}"

                    if patient_id in patients_labels and not patients_labels[patient_id].check_date(date):
                        cur_shared_labels = patients_labels[patient_id].get_shared_labels(date)
                        if OCTDataset.filter_scans_by_labels(cfg, cur_shared_labels):
                            oct_scans.append(cur_shared_labels)

            cols = ["P.I.D", "EYE", "DATE", "AMD_STAGE", "DRUSEN", "SDD", "PED", "SCAR", "SRHEM", "IRF",
                    "SRF", "IRORA", "IRORA_location", "CRORA", "cRORA_location"]
            unlabeled = pd.DataFrame(oct_scans, columns=cols)
            scans_labels = scans_labels.append(unlabeled, ignore_index=True)

        scans_labels.sort_values('P.I.D', inplace=True)
        twenty_precent = int(len(scans_labels) * 0.2)
        ten_percent = int(len(scans_labels) * 0.1)
        if train:
            if cfg.TRAIN.DEBUG:
                return scans_labels.iloc[twenty_precent: (twenty_precent + (ten_percent // 2))]
            return scans_labels.iloc[twenty_precent:]
        elif cfg.TEST.TEST_OR_VALIDATION == 'validation':
            return scans_labels.iloc[:ten_percent]
        elif cfg.TEST.TEST_OR_VALIDATION == 'test':
            return scans_labels.iloc[ten_percent: twenty_precent]
        else:
            raise NotImplementedError(
                f"Does not support {cfg.TEST.TEST_OR_VALIDATION} dataset")

    def transform_additional_targets(self):
        """
        For usage of albumentations library
        """
        im_str = 'image'
        targets = {}
        for i in range(1, self.num_frames):
            cur_key = im_str + str(i)
            targets[cur_key] = im_str
        return targets

    def _get_transform(self, transform_type, crop_size, train=True):
        """
        Creates transformation with albumentations library
        """
        add_targets = self.transform_additional_targets()
        mean_arr = self.cfg.DATA_LOADER.MEAN
        std_arr = self.cfg.DATA_LOADER.STD
        if train or transform_type == 'crops':
            if transform_type == "distribution_calc":
                return A.Compose([A.ToFloat(), A.Resize(crop_size, crop_size), ToTensorV2()])
            if transform_type == "flip":
                return A.Compose([A.ToFloat(),
                                  A.Resize(crop_size, crop_size),
                                  A.HorizontalFlip(p=0.5),
                                  A.Rotate(limit=25),
                                  A.Normalize(mean=mean_arr,
                                              std=std_arr,
                                              max_pixel_value=1.0),
                                  ToTensorV2()],
                                 additional_targets=add_targets)
            if transform_type == "crops":
                if train:
                    return A.Compose([A.ToFloat(),
                                      A.HorizontalFlip(p=0.5),
                                      A.SmallestMaxSize(max_size=crop_size),
                                      A.RandomCrop(crop_size, crop_size),
                                      A.Normalize(mean=mean_arr,
                                                  std=std_arr,
                                                  max_pixel_value=1.0),
                                     ToTensorV2()],
                                     additional_targets=add_targets)
                else:
                    return A.Compose([A.ToFloat(),
                                      A.SmallestMaxSize(max_size=crop_size),
                                      A.RandomCrop(crop_size, crop_size),
                                      A.Normalize(mean=mean_arr,
                                                  std=std_arr,
                                                  max_pixel_value=1.0),
                                      ToTensorV2()],
                                     additional_targets=add_targets)
        return A.Compose([A.ToFloat(), A.Resize(crop_size, crop_size), A.Normalize(mean=mean_arr,
                                                                                   std=std_arr,
                                                                                   max_pixel_value=1.0),
                          ToTensorV2()],
                         additional_targets=add_targets)

    def _get_slices_handling_policy(self, cfg):
        """
        Frame picking heuristic
        """
        if cfg.DATA_LOADER.METHOD == "mid slices" or cfg.DATA_LOADER.METHOD == "16 mid slices":
            return self._get_mid_slices
        elif cfg.DATA_LOADER.METHOD == "grouped":
            return self._get_grouped_slices
        elif cfg.DATA_LOADER.METHOD == "down sampling":
            return self._get_slices_by_downsampling
        elif cfg.DATA_LOADER.METHOD == "random":
            if cfg.DATA_LOADER.ORDERED:
                return self._get_ordered_random_slices
            else:
                return self._get_random_slices
        elif cfg.DATA_LOADER.METHOD == 'all':
            return self._get_all_slices
        else:
            raise NotImplementedError(
                f"Does not support {cfg.DATA_LOADER.METHOD} dataset handling policy"
            )

    def __init__(self, cfg, train=True):
        annotation_file = cfg.DATA.ANNOTATION_PATH
        data_dir = cfg.DATA.PATH_TO_DATA_DIR
        interpolate_data = cfg.DATA_LOADER.INTERPOLATE_DATA
        crop_size = cfg.DATA.TRAIN_CROP_SIZE if train else cfg.DATA.TEST_CROP_SIZE

        self.num_frames = cfg.DATA.NUM_FRAMES
        self.scans_labels = OCTDataset._get_annotations(annotation_file, data_dir, train, interpolate_data, cfg)
        self.data_dir = data_dir
        self.features_to_predict = cfg.TARGET_FEATURE.split()
        self.get_slices = self._get_slices_handling_policy(cfg)
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.cfg = cfg
        self.transform = self._get_transform(cfg.DATA_LOADER.TRANSFORMS, crop_size, train)

    def __len__(self):
        return len(self.scans_labels)

    def get_labels_distribution(self):
        counts = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            if self.num_classes == 2 and self.cfg.TARGET_FEATURE == 'AMD_STAGE':
                val = i + 1
            else:
                val = i
            counts[i] = np.sum((self.scans_labels[self.cfg.TARGET_FEATURE].values == val).astype(int))
        return counts

    def _transform_all_images(self, images):
        res = []
        for image in images:
            image = self.transform(image=image)['image']
            res.append(image)
        return res

    def _slices_to_images(self, slices, _all=False):
        images = list()
        for slice in slices:
            image = cv2.imread(slice)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
        if _all:
            return self._transform_all_images(images)
        if self.transform:
            targets = ['image' + str(i) for i in range(0, self.num_frames)]
            targets[0] = 'image'
            if self.num_frames == 16:
                trans = self.transform(image=images[0], image1=images[1], image2=images[2],
                                       image3=images[3], image4=images[4], image5=images[5],
                                       image6=images[6], image7=images[7],
                                       image8=images[8], image9=images[9], image10=images[10],
                                       image11=images[11], image12=images[12], image13=images[13],
                                       image14=images[14], image15=images[15])
            elif self.num_frames == 8:
                trans = self.transform(image=images[0], image1=images[1], image2=images[2],
                                       image3=images[3], image4=images[4], image5=images[5],
                                       image6=images[6], image7=images[7])
            images.clear()
            for i in range(self.num_frames):
                images.append(trans[targets[i]])
        return images

    def _get_all_slices(self, slices):
        while len(slices) < self.num_frames:
            slices += slices
        slices = slices[:self.num_frames]
        images = self._slices_to_images(slices, True)
        return images

    def _get_random_slices(self, slices):
        if len(slices) > self.num_frames:
            slices = random.choices(slices, k=self.num_frames)
        else:
            slices = slices + [slices[-1] for i in range(self.num_frames - len(slices))]  # Duplicate the
        images = self._slices_to_images(slices)
        return images

    def _get_ordered_random_slices(self, slices):
        if len(slices) > self.num_frames:
            idx = random.sample(range(len(slices)), k=self.num_frames)
            idx.sort()
            slices = [slices[i] for i in idx]
        else:
            slices = slices + [slices[-1] for i in range(self.num_frames - len(slices))]  # Duplicate the
        images = self._slices_to_images(slices)
        return images

    def _get_slices_by_downsampling(self, slices):
        if len(slices) > self.num_frames:
            idx = random.randint(0, len(slices)-1)
            step = int(len(slices) / self.num_frames)
            tmp_slices = slices[idx::step]
            if len(tmp_slices) < self.num_frames:
                new_idx = (len(slices) - idx) % step
                tmp_slices += slices[new_idx: idx: step]
            slices = tmp_slices[:self.num_frames]
        else:
            slices = slices + [slices[-1] for i in range(self.num_frames - len(slices))]  # Duplicate the last
            # slice
        return self._slices_to_images(slices)

    def _get_mid_slices(self, slices):
        # Get the middle slices.
        half_range = self.num_frames // 2
        num_slices = len(slices)
        if num_slices >= self.num_frames:
            slices = slices[num_slices//2 - half_range: num_slices // 2 + half_range]
        # Duplicate the last slices.
        else:
            slices = slices + [slices[-1] for i in range(self.num_frames - num_slices)]  # Duplicate the last slice

        images = self._slices_to_images(slices)
        return images

    def _get_grouped_slices(self, slices):
        # Group the slices to num_frames groups.
        images = list()
        for i in range(0, len(slices), self.num_frames):
            images.append(self._get_mid_slices(slices[i:i + self.num_frames]))
        return images

    def __getitem__(self, idx):
        slices = []
        idx -= 1  # Code for handle bug in the data of empty dirs.
        while len(slices) == 0:
            idx += 1
            oct_name = "{} {}".format(self.scans_labels.iloc[idx]['EYE'], self.scans_labels.iloc[idx]['DATE'])
            oct_scan = join(self.data_dir, self.scans_labels.iloc[idx]['P.I.D'], oct_name)
            if isdir(oct_scan):
                slices = [join(oct_scan, f) for f in listdir(oct_scan) if isfile(join(oct_scan, f)) and f.startswith("slice")]
                slices.sort(key=lambda f: int(re.sub('\D', '', f)))
            if idx == len(self.scans_labels) - 1:
                idx = 1

        # Get OCT scans slices.
        slices = self.get_slices(slices)
        slices = torch.stack(slices)

        # Get OCT scan labels.
        labels = self.scans_labels.iloc[idx][self.features_to_predict].values
        if self.cfg.MODEL.NUM_CLASSES == 2 and self.cfg.TARGET_FEATURE == 'AMD_STAGE':
            labels = labels - 1
        # labels = self.one_hot_converter[labels.astype(int)]
        return slices, labels.astype(int)[0]
