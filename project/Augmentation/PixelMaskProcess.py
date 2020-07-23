import cv2
import os
import pandas as pd
from base.ABCProcess import ABCProcess
import GlobalVariables


class PixelMaskProcess(ABCProcess):
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
        os.makedirs(os.path.join(GlobalVariables.DATA_DIR, self.conf.path.save.polygon), exist_ok=True)
        for index, row in self.df.iterrows():
            image_path = os.path.join(row['image_path'], row['image_file'])
            mask_path = os.path.join(row['mask_path'], row['mask_file'])

            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            mask[mask == 1] = 0
            mask_color = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255],
                          [255, 255, 255]]
            for label in self.conf.label:
                image[mask == label] = mask_color[label]

            cv2.imwrite(os.path.join(GlobalVariables.DATA_DIR, self.conf.path.save.polygon, row['mask_file']), image)

    def save(self):
        pass



if __name__ == "__main__":
    from base.BaseConf import BaseConf

    conf = BaseConf(GlobalVariables.CONF_AUG_DIR + '/MaskCOnf.yaml')

    Preprocess = PixelMaskProcess(conf, None)
    Preprocess.load()
    Preprocess.run()
