import os
import shutil
import xml.etree.ElementTree as ET
import tarfile, zipfile
import numpy
from PIL import Image
import pickle
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
    # print(line_folder_path)
    tar_path = os.path.join(IAM_root, tar_filename)
    if not os.path.isfile(tar_path):   # if not a file
        print("error - {} not found".format(tar_path))
        exit(-1)

    os.makedirs(IAM_target, exist_ok=True)  #exist_ok：只有在目录不存在时创建目录，目录已存在时不会抛出异常。
    tar = tarfile.open(tar_path)
    tar.extractall(line_folder_path) #解压
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
        current_folder = os.path.join(IAM_target, set_name)   # IAM_lines/train
        os.makedirs(current_folder, exist_ok=True)      # make
        xml_path = os.path.join(IAM_root, "{}.xml".format(set_name))  # xml路径
        xml_root = ET.parse(xml_path).getroot()   # 找根节点
        for page in xml_root:   # 遍历
            name = page.attrib.get("FileName").split("/")[-1].split(".")[0]   # 把序号切出来 c04-156
            # print(name, "切出来的序号")
            img_fold_path = os.path.join(line_folder_path, name.split("-")[0], name)  # /home/lyt/demospace/Vertical_att_line_OCR/Dataset/format/IAM_lines/lines/c04/c04-156
            # print(img_fold_path, "切出来的路径")
            img_paths = [os.path.join(img_fold_path, p) for p in sorted(os.listdir(img_fold_path))]
            # print(img_paths,"imgpath======")
            for i, line in enumerate(page[2]):
                label = line.attrib.get("Value")   # 把xml里的line取出来 就是label 图片对应的文字
                img_name = "{}_{}.png".format(set_name, id)
                # print(img_name, "什么样格式的imgname") # train_4567.png
                # TODO: 这里的gt没太懂  懂了 就是一个字典
                gt[set_name][img_name] = {
                    "text": label,
                }
                charset = charset.union(set(label))
                # print(charset, "看看charset是什么")  # {'0', 'r', '.', 'V', 'Z', 'n', ',', 'U', 'H', '-', '8', 'O', 'S', ' ', 'y', 'w', 'R', '"', '?', '(', 'K', '/', 'J', '7', 'I', 'a', 'k', 'j', 'E', 'D', 'u', 'F', 'C', 'T', '4', ')', 'N', '#', 'Y', 'o', 'z', '&', 'm', 'x', 'W', 'M', 'Q', '6', '!', 'A', 'f', 'p', 'e', 's', 'v', '1', '2', 'h', '3', 'X', ';', '+', 'b', '5', 'g', 'q', 'i', 'G', ':', "'", 't', 'P', '*', 'd', '9', 'L', 'c', 'B', 'l'}
                new_path = os.path.join(current_folder, img_name)
                # print(new_path, "看看new_path是什么")
                os.replace(img_paths[i], new_path)
                id += 1

    shutil.rmtree(line_folder_path)   # 递归删除一个目录以及目录内的所有内容
    # print(gt)  # {'train': {'train_0.png': {'text': 'A MOVE to stop Mr. Gaitskell from'}, 'train_1.png': {'text': 'nominating any more Labour life Peers'}, 'train_2.png': {'text': 'is to be made at a meeting of Labour'}, 'train_3.png': {'text': 'M Ps tomorrow. Mr. Michael Foot has'},
    print(list(charset)) # TODO:这里的charset有什么用？ 已解决
    with open(os.path.join(IAM_target, "labels.pkl"), "wb") as f:
         pickle.dump({
            "ground_truth": gt,
            "charset": sorted(list(charset)),
        }, f)
    # dump一下 默认是0(ASCII协议，表示以文本的形式进行序列化)，值为1和2(1和 2表示以二进制的形式进行序列化。1是老式的二进制协议，2是新二进制协议)



if __name__ == '__main__':
    format_IAM()