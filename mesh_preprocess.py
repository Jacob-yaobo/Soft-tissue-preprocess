'''
 # @ Author: Yaobo Jia
 # @ Create Time: 2025-05-30 14:31:55
 # @ Modified by: Yaobo Jia
 # @ Modified time: 2025-05-30 14:33:54
 # @ Description:
 '''
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def transform_coordinate(nosetip_excel, pid):
    
    return 


if __name__ == '__main__':
    root_dir = Path('raw_data/2_2_review_seg')
    for post_file_path in tqdm(root_dir.rglob("*_PostOp.nrrd")):
        post_filename = post_file_path.name
        pre_filename = post_filename.replace("_PostOp.nrrd", "_PreOp.nrrd")
        pre_file_path = post_file_path.with_name(pre_filename)

        post_relative_path = post_file_path.relative_to(root_dir)
        pid = post_relative_path.parts[1]
