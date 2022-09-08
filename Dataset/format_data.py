import os
import shutil
import xml.etree.ElementTree as ET
import tarfile, zipfile
import numpy
from PIL import Image
"""
import xml.etree.ElementTree as ET 
#check XML
tree = ET.parse("/home/lyt/demospace/VerticalAttentionOCR/Datasets/raw/IAM/valid.xml") #把目标xml文件解析成一棵ElementTree
root = tree.getroot() #找到tree的根节点，并将节点返回给root变量。
# print(root.tag) ==>DocumentList
# print(root.attrib)==>{}
for child in root[2]:
    print(child.tag)

for neighbor in root.iter('Line'):
	print(neighbor.get("Value"))
"""

def format_IAM():
    IAM_root = "/home/lyt/demospace/VerticalAttentionOCR/Datasets/raw/IAM"
    IAM_target= "/home/lyt/demospace/Vertical_att_line_OCR/Dataset/format/IAM_lines"
    tar_filename = "lines.tgz"
    line_folder_path = os.path.join(IAM_target, "lines")
    tar_path = os.path.join(IAM_root, tar_filename)
    if not os.path.isfile(tar_path):   # if not a file
        print("error - {} not found".format(tar_path))
        exit(-1)

    os.makedirs(IAM_target, exist_ok=True)  #exist_ok：只有在目录不存在时创建目录，目录已存在时不会抛出异常。
    tar = tarfile.open(tar_path)
    tar.extract(line_folder_path)  #解压
    tar.close()

    set_names = ["train", "valid", "test"]

    gt = {
        "train": dict(),
        "valid": dict(),
        "test": dict(),
    }
    charset = set()

    for set_name in set_names:
        id = 0
        current_folder = os.path.join(IAM_target, set_name)
        os.makedirs(current_folder, exist_ok=True)
        xml_path = os.path.join(IAM_root, "{}.xml".format(set_name))
        print(xml_path)
        xml_path
