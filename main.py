# -*- coding: utf-8 -*-
"""
This module is to test cataract model.
Commands:

python3.7 main.py --input INPUT_FILEPATH --label LABEL_FILE --output outputs --threshold float_value

python3.7 main.py --input images --output outputs

python3.7 main.py --input images --label ground_truth_sample.csv --output outputs --threshold 0.013259

"""

import time
from torchvision import transforms, datasets
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import torch
import pdb
import cv2
import torch.nn as nn
import argparse
from tqdm import tqdm
from sklearn import metrics

import csv
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import roc_curve,auc,roc_auc_score
from sklearn.preprocessing import label_binarize
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt

from imagenet_utils import preprocess_input


prj_path = './'
error_code = 0
# MODEL_NAME = 'referable_0.48_16July_Resnet50.dat'
MODEL_NAME = 'paper_retina_ref_048_Resnet50_cataract_model_p3.dat'
model_index = 1
models = ['VGG16', 'Resnet50', 'Densenet112', 'inres']


def feat_extract(model, img):
    img = np.array(img, dtype=np.float32)
    img = preprocess_input(img)
    feat = model.predict(img)
    feat = np.reshape(feat, (feat.shape[0], -1))
    return feat

def get_model(index):
    print(index)
    # 0 :vgg, 1:resnet 50 2:densenet
    if index == 0:
        from deepLearningModel.vgg16 import VGG16
        model = VGG16(weights="imagenet", include_top=False)
        return model
    if index == 1:
        from deepLearningModel.resnet50 import ResNet50
        model = ResNet50(weights='imagenet', include_top=False)
        return model
    if index == 2:
        from densenet.densenet import DenseNetImageNet121
        # image_dim = (224, 224, 3)
        model = DenseNetImageNet121(weights='imagenet', include_top=False)
        return model
    if index == 3:
        from resnet import ResNet101
        return ResNet101(weights='imagenet', include_top=False)
    if index == 4:
        model = EfficientNetB0(include_top=False, weights="imagenet")
        return model

# auc_value = plot_auc(y, y_pred, savename)
def plot_auc(y_test, y_pred, save_name):
    cls_max = int(np.max(y_test))
    label_name = 'visual_impairment_corrected'
    y_mult = label_binarize(y_test, classes=range(cls_max + 1))

    for c in range(cls_max):
        if cls_max > 1:
            y_truth = y_mult[:, c + 1]
        else:
            y_truth = y_test

        fpr, tpr, thresholds = roc_curve(y_truth, y_pred[:, c + 1])
        # fpr, tpr, thresholds = roc_curve(y_truth, y_pred)
        roc_auc = auc(fpr, tpr)
        print(roc_auc)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('%s ROC - class %d' % (label_name, c + 1))
        plt.savefig("pre_new_model.jpg", format='JPEG')
        plt.legend(loc="lower right")

        plt.show()
    return roc_auc


def regression_plot(y, y_pred_class):
    plt.scatter(y, y_pred_class, s=0.2)
    plt.plot([-2, 2], [-2, 2], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 2.0])
    # plt.ylim([0.0, 2.05])
    plt.xlabel('Ground truth')
    plt.ylabel('Estimated')
    plt.title('Regression plot')
    plt.legend(loc="lower right")
    plt.savefig("../fig/pre_new_model.jpg", format='JPEG')
    plt.show()
    return


class MMD_Dataset(Dataset):
    def __init__(self, data_dir, image_list_file, transform=None, target_transform=None):
        """
        Args:
            data_dir: path to image directory.
                Note: we don't use it in this project, we keep it for future usage
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
            target_transform: optional transform to label image.
        """
        imgs = []
        if image_list_file is not None:
            for line in image_list_file:
                img = line[0]
                mask = line[1]
                imgs.append((img, mask))
        else:
            # insert with dummy label 0
            for each in os.listdir(data_dir):
                if each.startswith(".") or each.endswith(".csv"):
                    continue
                imgs.append((each, 0))

        #imgs = make_dataset(root)
        self.data_path = data_dir
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        x_path, label = self.imgs[index]

        # img_x is array
        file_with_path = os.path.join(self.data_path, x_path)
        imgs = readImageFile(file_with_path)
        head, tail = os.path.split(file_with_path)
        self.filename = tail
        img_y = int(label)

        img_x = imgs

        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)

        return img_x, img_y, x_path

    def __len__(self):
        return len(self.imgs)

    def __current_filename__(self):
        return self.filename


def test_accuracy(y, y_pred, is_regression=False):
    # regression accuracy
    if is_regression:
        # averg_pre = mean_absolute_error(y, y_pred)
        # r2 = r2_score(y, y_pred)
        # return 0, r2, averg_pre
        return

    y_pr = []
    for i in y_pred:
        if i > 0.15:
        # if i > 0.79:
            y_pr.append(0)
        else:
            y_pr.append(1)

    y_pr = np.array(y_pr)
    TP = 0.0
    FN = 0.0
    TN = 0.0
    FP = 0.0

    for i_true, i_pred in zip(y, y_pr):
        if i_true == 1:
            if i_pred == i_true:
                TP += 1
            else:
                FN += 1

        if i_true == 0:
            if i_pred == i_true:
                TN += 1
            else:
                FP += 1
    '''
    Precision = TP/TP+FP    (i.e  PPV)
    Recall = TP / FN+TP
    F1 score = 2 * (precision * recall)/ (precision + recall)
    Sensitivity = TP / FN+TP (i.e  Recall)
    Specificity = TN / FP+TN
    Accuracy = (TP+TN)/(TP+FP+TN+FN)
    PPV = TP / (TP + FP)
    NPV = TN / (TN + FN)
    '''
    if TP + FP == 0:
        Precision = 1.0
    else:
        Precision = TP / (TP + FP)

    if TP + FN == 0:
        Sensitivity = 0.0
    else:
        Sensitivity = TP / (TP + FN)

    if TN + FN == 0:
        NPV = 0.0
    else:
        NPV = TN / (TN + FN)

    if TN + FP == 0:
        Specificity = 1.0
    else:
        Specificity = TN / (TN + FP)

    F1 = 2 * Precision * Sensitivity / (Precision + Sensitivity)
    Accuracy = (TP + TN) / len(y)

    return Precision, Sensitivity, Specificity, F1, Accuracy, NPV

def is_img(ext):
    ext = ext.lower()
    if ext == '.jpg' or ext == '.JPG' :
        return True
    elif ext == '.png' or ext == '.PNG':
        return True
    elif ext == '.jpeg' or ext == '.JPEG':
        return True
    elif ext == '.bmp' or ext == '.BMP':
        return True
    elif ext == '.tif' or ext == '.TIF':
        return True
    elif ext == '.tiff' or ext == '.TIFF':
        return True
    elif ext == '.dcm' or ext == '.dicom':
        return True
    elif ext == '.DCM' or ext == '.DICOM':
        return True
    else:
        return False

def readImageFile(FilewithPath):
    # we could repeat a few times in case the network file transder is not done
    img = None
    fileName = os.path.basename(FilewithPath)
    ext_str = os.path.splitext(fileName)[1]
    repeat_time = 3
    if is_img(ext_str):
        # print("Input Image: ", fileName)
        cycle_cnt = repeat_time
        while cycle_cnt>0 and img is None:
            # try one more time in case libpng error: Read Error
            try:
                img = cv2.imread(FilewithPath, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (224, 224))
                #img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            except:
                print("read image error, ignore the file")

            if img is None:
                cycle_cnt = cycle_cnt - 1
                time.sleep(0.02 * (repeat_time-cycle_cnt))

        if img is not None and img.shape[2] == 1:
            # repeat 3 times to make fake RGB images
            img = np.tile(img, [1, 1, 3])

    return img

def loadImageList(imglist_filepath):
    img_dir = imglist_filepath
    name_list = []
    X = []

    # Loop through the training and test folders, as well as the 'NORMAL' and 'PNEUMONIA' subfolders
    # and append all images into array X.  Append the classification (0 or 1) into array Y.
    #'''
    for fileName in os.listdir(img_dir):
        name_list.append(fileName)
        img = readImageFile(img_dir + fileName)
        if img is not None:
            X.append(img)

        # delete the physical image after read it
        time.sleep(0.01)
        #cmd_str = "rm -f " + img_dir + fileName
        cmd_str = "rm -f " + '"' + img_dir + fileName + '"'
        #print("delete input image file: ", cmd_str)
        os.system(cmd_str)

    return name_list, X


def read_csv_to_list(file):
    label_list = []
    # we only care about the 2 labels -- Image Index / Finding Labels
    if os.path.isfile(file) is True:
        with open(file, "r") as fr:
            reader = csv.reader(fr)
            for line in reader:
                # Image file & label
                label_list.append([line[0], line[1]])
    return label_list

class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module # that I actually define.
    def forward(self, x):
        return self.module(x)


def load_model(model_name):
    if os.path.isfile(model_name):
        print('loading model: %s' % (model_name))
        # clf = pickle.load(open(model_name, "rb"))

        # info = pickle.dumps(clf, protocol=2)
        # clf = pickle.load(open("new_model.dat", "rb"), encoding="latin1")

        # xiaofeng use python3 model instead now
        # clf = pickle.load(open(model_name, "rb"), encoding='latin1')
        clf = pickle.load(open(model_name, "rb"))

        # with open(model_name, 'rb') as f:
        #     clf = pickle.load(f, encoding="bytes")
            # clf = pickle.load(f, encoding = 'iso-8859-1')

        # with open("p3_model.dat", 'wb') as outfile:
        #     pickle.dump(clf, outfile, protocol=-1)

        # with open(model_name, 'rb') as file_object:
        #     clf = pickle.load(file_object)
        #     # print(raw_data)

        # joblib doesn't support python 2/3 convert
        #     # Save to file in the current working directory
        #     joblib_file = "joblib_model.pkl"
        #     joblib.dump(clf, joblib_file)
        # # Load from file
        # joblib_file = "joblib_model.pkl"
        # clf = joblib.load(joblib_file)

        # with open("paper_retina_ref_048_Resnet50_cataract_model_p3.dat", 'wb') as outfile:
        #     pickle.dump(clf, outfile, protocol=-1)
    else:
        raise('No model found...')
    return clf

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='VI CNN')

    parser.add_argument('--input', dest='input_filepath',
                        help='the dataset images folder to be test',
                        default='images', type=str)
    parser.add_argument('--label', dest='label_file',
                        help='the csv file with ground truth for the images, follow the same format as example file',
                        default=None, type=str)
    parser.add_argument('--output', dest='output_filepath',
                        help='the destination folder for output csv file, by default with "outputs" folder',
                        default=None, type=str)
    parser.add_argument('--threshold', dest='threshold_float',
                        help='the threshold for the prediction 1 of the AI model',
                        default=0.013259, type=float)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.output_filepath is None:
        output_dir = 'outputs'
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
    else:
        if os.path.isdir(args.output_filepath) is False:
            try:
                os.mkdir(args.output_filepath)
            except:
                print('Wrong output folder name, or not authority to create a new folder !!!')
                exit()
        output_dir = args.output_filepath

    input_images = args.input_filepath
    if input_images is not None:
        if os.path.isfile(input_images):
            test_mode = 'file_mode'
        elif os.path.isdir(input_images):
            test_mode = 'path_mode'
        else:
            raise("Wrong input file or folder!!!")
    else:
        raise("Please give the correct input!!!")

    if test_mode == 'path_mode':
        label_model = False
        label_file = args.label_file
        if (label_file is not None) and os.path.isfile(label_file):
            label_model = True
            print('Input has ground truth: {}' .format(label_model))
            assert os.path.splitext(label_file)[-1] == ".csv"

    start = time.time()
    net = load_model(MODEL_NAME)
    end_time = time.time()
    print('loading %s model, time %.2f' %(MODEL_NAME, end_time - start))

    num_class = 2
    batch_size = 1
    threshold = args.threshold_float

    start = time.time()
    if test_mode == 'path_mode':
        # need to call dataset class to organize images
        DATA_DIR = input_images
        if label_model is True:
            TEST_IMAGE_LIST = read_csv_to_list(label_file)
        else:
            TEST_IMAGE_LIST = None
        data = {
            'test':
                MMD_Dataset(data_dir=DATA_DIR, image_list_file=TEST_IMAGE_LIST, \
                            transform=None, target_transform=None)
        }
        # Dataloader iterators
        dataloaders = {
            'test': DataLoader(data['test'], batch_size=batch_size, shuffle=False)
        }

        image_list = []
        file_list = []
        Y_list = []
        y_pred = []

        step_num = 0
        sum_dataset = len(dataloaders['test'].dataset)

        model = get_model(model_index)
        # import pdb
        # pdb.set_trace()
        # Note: support batch=1 only
        for imgs, Y, file in tqdm(dataloaders['test']):
            Y_list.append(int(Y[0]))
            file_list.append(file)
            feat_test = feat_extract(model, imgs)
            batch_pred = net.predict_proba(feat_test)
            y_pred.append(batch_pred)

        # import pdb
        # pdb.set_trace()

        Y_list = np.array(Y_list, dtype=int)
        file_list = np.array(file_list)
        y_pred = np.array(y_pred)

        y_pred = np.squeeze(y_pred, axis=1)
        statistic_file = os.path.join(output_dir, 'TestResult.csv')
        C = open(statistic_file, 'w')
        C.write(
            'file name,AI_model,ground truth,threshold,probability of 0,probability of 1,prediction(0=Non-Cataract 1=Cataract)\n')

        for i in range(sum_dataset):
            probability_0 = y_pred[i, 0]
            probability_1 = y_pred[i, 1]
            print("filename:{}, probability:{}".format(file_list[i], probability_1))

            Threshold_Cataract = threshold
            class_result = 0
            if probability_1 > Threshold_Cataract:
                class_result = 1

            C.write('{},{},{},{},{},{},{}\n' \
                .format(file_list[i], models[model_index], Y_list[i],Threshold_Cataract,probability_0,probability_1,class_result))
        C.close()


    elif test_mode == 'file_mode':
        # we only read the input file and give result
        image_list = []
        filename = input_images
        imgs = readImageFile(filename)
        image_list.append(imgs)

        model = get_model(model_index)
        X = np.array(image_list)
        feat_test = feat_extract(model, X)

        y_pred = net.predict_proba(feat_test)
        probability_0 = y_pred[0, 0]
        probability_1 = y_pred[0, 1]
        print("filename:{}, probability:{}" .format(filename, probability_1))

        statistic_file = os.path.join(output_dir, 'TestResult.csv')
        if os.path.isfile(statistic_file):
            C = open(statistic_file, 'a+')
        else:
            C = open(statistic_file, 'w')
            C.write('file name,AI_model,threshold,probability of 0,probability of 1,prediction(0=Non-Cataract 1=Cataract)\n')

        Threshold_Cataract = threshold
        class_result = 0
        if probability_1 > Threshold_Cataract:
            class_result = 1

        C.write('{},{},{},{},{},{}\n' \
                .format(filename, models[model_index], Threshold_Cataract, probability_0, probability_1, class_result))
        C.close()

    print("Cataract Test is Over, Get your results in {} !!!\n" .format(statistic_file))
