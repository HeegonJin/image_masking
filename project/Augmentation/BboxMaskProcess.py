import cv2
import numpy as np
import os
import pandas as pd
from project.Augmentation.CreateBoundBox import processImage
from base.ABCProcess import ABCProcess
import GlobalVariables


class BboxMaskProcess(ABCProcess):
    def load(self):
        image_path = os.path.join(GlobalVariables.DATA_DIR,  self.conf.path.data.x)
        mask_path = os.path.join(GlobalVariables.DATA_DIR, self.conf.path.data.y)

        image_list = [[path, name, os.path.splitext(name)[0]] for path, subdirs, files in os.walk(image_path) for name in
                      files if
                      os.path.splitext(name)[1] == self.conf.ext.image]
        mask_list = [[path, name, os.path.splitext(name)[0]] for path, subdirs, files in os.walk(mask_path) for name in
                     files if
                     os.path.splitext(name)[1] == self.conf.ext.mask]

        image_df = pd.DataFrame(image_list)
        mask_df = pd.DataFrame(mask_list)
        image_df.columns = ['image_path', 'image_file', 'common_file_name']
        mask_df.columns = ['mask_path', 'mask_file', 'common_file_name']
        self.df = pd.merge(image_df, mask_df, on='common_file_name')

    def run(self, data=None):
        os.makedirs(os.path.join(GlobalVariables.DATA_DIR, self.conf.path.save.bbox),exist_ok=True)
        for index, row in self.df.iterrows():
            image_path = os.path.join(row['image_path'], row['image_file'])
            mask_path = os.path.join(row['mask_path'], row['mask_file'])

            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            mask[mask == 1] = 0
            kernel = np.ones((self.conf.kernel_size.height, self.conf.kernel_size.width), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_output = processImage(image, mask, thresh=0)  # cv.canny threshold, just set 0
            cv2.imwrite(os.path.join(GlobalVariables.DATA_DIR, self.conf.path.save.bbox, row['mask_file']), mask_output)

    def save(self):
        pass



if __name__ == "__main__":
    from base.BaseConf import BaseConf

    conf = BaseConf(GlobalVariables.CONF_SEG_DIR + '/MaskCOnf.yaml')

    Preprocess = BboxMaskProcess(conf, None)
    Preprocess.load()
    Preprocess.run()
