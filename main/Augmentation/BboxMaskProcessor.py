from project.Augmentation.BboxMaskProcess import BboxMaskProcess
from base.BaseConf import BaseConf
import GlobalVariables
import os

if __name__ == "__main__":
    _conf_path = os.path.join(GlobalVariables.CONF_AUG_DIR, 'MaskConf.yaml')
    conf = BaseConf(_conf_path)

    bbox_mask_processor = BboxMaskProcess(conf.MaskProcess, None)
    bbox_mask_processor.load()
    bbox_mask_processor.run()
    bbox_mask_processor.save
