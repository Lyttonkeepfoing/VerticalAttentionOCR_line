import editdistance # 指两个字串之间，由一个转成另一个所需的最少编辑操作次数，如果它们的距离越大，说明它们越是不同。 编辑操作只有三种插入、删除、替换三种操作。
import re


def LM_str_to_ind(labels,str):  # 遍历str 返回str在labels中的index
    return [labels.index(c) for c in str]


def LM_ind_to_str(labels, ind, oov_symbol=None):# 取出labels中对应index的str
    if oov_symbol is not None:
        res = []
        for i in ind:
            if i < len(labels):
                res.append(labels[i])
            else:
                res.append(oov_symbol)
    else:
        res = [labels[i] for i in ind]
    return "".join(res)

def edit_cer_from_list(truth, pred):
    edit = 0
    for t, p in zip(truth, pred):
        edit += editdistance.eval(t, p)
    return edit
# 算truth 和pred之间的editdistance(str)
# print(edit_cer_from_list("adsa", "asdc"))  # 3

def format_string_for_wer(str):
    str = re.sub('([\[\]{}/\\()\"\'&+*=<>?.;:,!\-—_€#%°])', r' \1 ', str)
    str = re.sub('([ \n])+', " ", str).strip()
    return str
# print(re.sub('([\[\]{}/\\()\"\'&+*=<>?.;:,!\-—_€#%°])', r' \1 ', "sdasda??ASD?>%%.!&"))
# print(format_string_for_wer("sdasda??ASD?>%%.!&"))
# print("sdasda??ASD||%%")
# print(r' \1')
# print(format_string_for_wer("asda,dsa"))
# 特殊符号两边加空格 分割

def edit_wer_from_list(truth, pred):
    edit = 0
    for pred, gt in zip(pred, truth):
        gt = format_string_for_wer(gt)
        pred = format_string_for_wer(pred)
        gt = gt.split(" ")
        pred = pred.split(" ")  # 按特殊符号切开
        edit += editdistance.eval(gt, pred) # 算距离
    return edit

# tr1 = ("sadad.dasdawvv//")
# sad = ("dcccasavvv..../asd")
# print(edit_wer_from_list(tr1, sad))

def nb_words_from_list(list_gt):
    len_ = 0
    for gt in list_gt:
        gt = format_string_for_wer(gt)
        gt = gt.split(" ")
        len_ += len(gt)
    return len_
# 切分成多少段 sads一段 特殊字符一段
# a = ["sads???aasda", "sada///asd"]
# print(nb_words_from_list(a))
# for i in a:
#     print(i)
#     i = format_string_for_wer(i)
#     print(i)
#     i = i.split(" ")
#     print(i)

def nb_chars_from_list(list_gt):
    return sum([len(t) for t in list_gt])
# 总字段多长 包含字母和特殊字符 不切分
# print(nb_chars_from_list(["sada//?", "das..ccc/"]))

def cer_from_list_str(str_gt, str_pred):
    len_ = 0
    edit = 0
    for pred , gt in zip(str_pred, str_gt):
        edit += editdistance.eval(gt, pred)
        # print(edit, "edit")
        len_ += len(gt)
        # print(gt, "gt")
    cer = edit / len_
    return cer

# print(cer_from_list_str(["dsa", "dsad"], ["dasdc", 'asdacccca']))
# 第一轮 edit 3 第二论 edit 7 第一轮len ground_truth len 3 第二轮 4  cer = 10/7

def wer_from_list_str(str_gt, str_pred):
    len_ = 0
    edit = 0
    for pred, gt in zip(str_pred, str_gt):
        gt = format_string_for_wer(gt)
        pred = format_string_for_wer(pred)
        gt = gt.split(" ")
        # print(gt)
        pred = pred.split(" ")
        # print(pred)
        edit += editdistance.eval(gt, pred)
        # print(edit, "edit")
        len_ += len(gt)
    cer = edit / len_
    return cer
# 做切分的cer计算 TODO:为什么这里写wer return cer
# print(wer_from_list_str("sad///a", "aac...s")) edit = 6 len_ = 7 把print注释解开就可以一目了然

def cer_from_files(file_gt, file_pred):
    with open(file_pred, "r") as f_p:
        str_pred = f_p.readlines()
    with open(file_gt, "r") as f_gt:
        str_gt = f_gt.readlines()
    return cer_from_list_str(str_gt, str_pred)
# 读取 gt和pred的文件 然后 readlines

def wer_from_files(file_gt, file_pred):
    with open(file_pred, "r") as f_p:
        str_pred = f_p.readlines()
    with open(file_gt, "r") as f_gt:
        str_gt = f_gt.readlines()
    return wer_from_list_str(str_gt, str_pred) # 同上



