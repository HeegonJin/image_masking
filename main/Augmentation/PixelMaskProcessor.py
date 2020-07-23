from project.Augmentation.PixelMaskProcess import PixelMaskProcess
from base.BaseConf import BaseConf
import GlobalVariables
import os

if __name__ == "__main__":
    _conf_path = os.path.join(GlobalVariables.CONF_AUG_DIR, 'MaskConf.yaml')
    conf = BaseConf(_conf_path)

    pixel_mask_processor = PixelMaskProcess(conf.MaskProcess, None)
    pixel_mask_processor.load()
    pixel_mask_processor.run()
    pixel_mask_processor.save
