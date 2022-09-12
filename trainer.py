from basic.training_manager import GenericTrainingManager
from basic.utils import edit_wer_from_list, nb_words_from_list, nb_chars_from_list, LM_ind_to_str
import editdistance
import re
import torch
from torch.nn import CTCLoss

"""
搞清楚数据的shape
batch_data: 
{'names': 八张img的名字
['IAM_lines/train/train_5323.png', 'IAM_lines/train/train_4374.png', 'IAM_lines/train/train_3647.png', 'IAM_lines/train/train_3779.png', 'IAM_lines/train/train_5424.png', 'IAM_lines/train/train_5465.png', 'IAM_lines/train/train_3341.png', 'IAM_lines/train/train_770.png'], 
'ids': 八张img对应的ids
[5323, 4374, 3647, 3779, 5424, 5465, 3341, 770],   
'nb_lines':有多少行 line-level 都是一行 
[1, 1, 1, 1, 1, 1, 1, 1],  
 'labels': # 每个label对应的tensor   padding value为1000
 tensor([[  29,   67,   65,   65,   67,   66,   71,    0,   34,   67,   73,   71,
           57,    0,   73,   68,   67,   66,    0,   53,    0,   56,   61,   71,
           71,   67,   64,   73,   72,   61,   67,   66,   10,    0,   75,   60,
           61,   55,   60,    0,   65,   73,   71,   72,    0,   54,   57,    0,
           60,   53,   56,   10,    0,   75,   61,   64,   64,    0,   54,   57,
         1000, 1000, 1000],
        [  71,   53,   64,   57,    0,   54,   57,   61,   66,   59,    0,   54,
           77,    0,   72,   60,   57,    0,   59,   53,   64,   64,   67,   66,
           12,    0,   42,   70,   53,   77,   61,   66,   59,    0,   65,   53,
           66,   72,   61,   71,    0,    6,    0,   39,   53,   66,   72,   61,
           71, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
         1000, 1000, 1000],
        [  58,   53,   70,   55,   57,    0,   61,   71,    0,   61,   72,    0,
           75,   60,   57,   66,    0,   61,   72,    0,   55,   67,   65,   57,
           71,    0,   72,   67,    0,   72,   57,   64,   57,   74,   61,   71,
           61,   67,   66,   12, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
         1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
         1000, 1000, 1000],
        [  58,   61,   74,   57,    0,   77,   57,   53,   70,   71,    0,   71,
           61,   66,   55,   57,    0,   60,   61,   71,    0,   58,   61,   70,
           71,   72,    0,   68,   64,   53,   77,   10,    0,   46,   60,   57,
            0,   43,   73,   53,   70,   57,    0,   32,   57,   64,   64,   67,
           75,   10,    0,   75,   53,   71, 1000, 1000, 1000, 1000, 1000, 1000,
         1000, 1000, 1000],
        [  65,   53,   77,    0,   59,   67,    0,   75,   57,   64,   64,   12,
            0,   45,   61,   70,    0,   44,   12,    0,   42,   57,   57,   64,
            0,   75,   53,   71,    0,   60,   57,   70,   57,   10,    0,   35,
            0,   73,   66,   56,   57,   70,   71,   72,   53,   66,   56,   10,
            0,   54,   73,   72,    0,   53,   66,    0,   57,   76,   68,   70,
           57,   71,   71],
        [  49,   60,   61,   64,   57,    0,   60,   57,    0,   75,   53,   71,
            0,   61,   66,    0,   40,   53,   68,   64,   57,   71,    0,   72,
           60,   57,   70,   57,    0,   60,   53,   56,    0,   67,   68,   57,
           66,   57,   56,    0,   53, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
         1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
         1000, 1000, 1000],
        [  57,   58,   58,   57,   55,   72,   61,   74,   57,   10,    0,   59,
           61,   74,   61,   66,   59,    0,   58,   73,   64,   64,    0,   74,
           53,   64,   73,   57,    0,   72,   67,    0,   72,   60,   57,    0,
           58,   67,   70,   65,   53,   64,    0,   57,   64,   57,   65,   57,
           66,   72,   71, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
         1000, 1000, 1000],
        [  65,   67,   71,   72,    0,   57,   76,   72,   70,   53,   67,   70,
           56,   61,   66,   53,   70,   77,    0,   72,   60,   61,   66,   59,
           71,    0,   53,   70,   57,    0,   60,   53,   68,   68,   57,   66,
           61,   66,   59,   12,    0,    2,    0,   32,   61,   58,   72,   77,
         1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
         1000, 1000, 1000]]), 
         'raw_labels':  # 原始标签的的内容 
         ['Commons House upon a dissolution, which must be had, will be', 'sale being by the gallon. Praying mantis ( Mantis', 'farce is it when it comes to television.', 'five years since his first play, The Quare Fellow, was', 'may go well. Sir R. Peel was here, I understand, but an express', 'While he was in Naples there had opened a', 'effective, giving full value to the formal elements', 'most extraordinary things are happening. " Fifty'],
         'unchanged_labels': # 没改变的标签内容 
         ['Commons House upon a dissolution, which must be had, will be', 'sale being by the gallon. Praying mantis ( Mantis', 'farce is it when it comes to television.', 'five years since his first play, The Quare Fellow, was', 'may go well. Sir R. Peel was here, I understand, but an express', 'While he was in Naples there had opened a', 'effective, giving full value to the formal elements', 'most extraordinary things are happening. " Fifty'], 
         'labels_len': # 每个label的长度 
         [60, 49, 40, 54, 63, 41, 51, 48],
          'imgs': # 每张img对应的tensor 
          tensor([[[[100., 100., 100.,  ...,   0.,   0.,   0.],
          [100., 100., 100.,  ...,   0.,   0.,   0.],
          [100., 100., 100.,  ...,   0.,   0.,   0.],
          ...,
          'imgs_shape':每张img的shape 
          [[47, 968, 3], [71, 877, 3], [40, 772, 3], [85, 898, 3], [44, 1016, 3], [83, 866, 3], [63, 916, 3], [81, 909, 3]],
          'imgs_reduced_shape': 处理过后的img的shape 
          [[1, 121, 3], [2, 109, 3], [1, 96, 3], [2, 112, 3], [1, 127, 3], [2, 108, 3], [1, 114, 3], [2, 113, 3]], 
           'line_raw': 每一行label 取成一个list 所有行再放进一个list里面
           [['Commons House upon a dissolution, which must be had, will be'], ['sale being by the gallon. Praying mantis ( Mantis'], ['farce is it when it comes to television.'], ['five years since his first play, The Quare Fellow, was'], ['may go well. Sir R. Peel was here, I understand, but an express'], ['While he was in Naples there had opened a'], ['effective, giving full value to the formal elements'], ['most extraordinary things are happening. " Fifty']], 
           'line_labels': 每个label对应的tensor 
           [tensor([[  29,   67,   65,   65,   67,   66,   71,    0,   34,   67,   73,   71,
           57,    0,   73,   68,   67,   66,    0,   53,    0,   56,   61,   71,
           71,   67,   64,   73,   72,   61,   67,   66,   10,    0,   75,   60,
           61,   55,   60,    0,   65,   73,   71,   72,    0,   54,   57,    0,
           60,   53,   56,   10,    0,   75,   61,   64,   64,    0,   54,   57,
         1000, 1000, 1000],
        [  71,   53,   64,   57,    0,   54,   57,   61,   66,   59,    0,   54,
           77,    0,   72,   60,   57,    0,   59,   53,   64,   64,   67,   66,
           12,    0,   42,   70,   53,   77,   61,   66,   59,    0,   65,   53,
           66,   72,   61,   71,    0,    6,    0,   39,   53,   66,   72,   61,
           71, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
         1000, 1000, 1000],
        [  58,   53,   70,   55,   57,    0,   61,   71,    0,   61,   72,    0,
           75,   60,   57,   66,    0,   61,   72,    0,   55,   67,   65,   57,
           71,    0,   72,   67,    0,   72,   57,   64,   57,   74,   61,   71,
           61,   67,   66,   12, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
         1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
         1000, 1000, 1000],
        [  58,   61,   74,   57,    0,   77,   57,   53,   70,   71,    0,   71,
           61,   66,   55,   57,    0,   60,   61,   71,    0,   58,   61,   70,
           71,   72,    0,   68,   64,   53,   77,   10,    0,   46,   60,   57,
            0,   43,   73,   53,   70,   57,    0,   32,   57,   64,   64,   67,
           75,   10,    0,   75,   53,   71, 1000, 1000, 1000, 1000, 1000, 1000,
         1000, 1000, 1000],
        [  65,   53,   77,    0,   59,   67,    0,   75,   57,   64,   64,   12,
            0,   45,   61,   70,    0,   44,   12,    0,   42,   57,   57,   64,
            0,   75,   53,   71,    0,   60,   57,   70,   57,   10,    0,   35,
            0,   73,   66,   56,   57,   70,   71,   72,   53,   66,   56,   10,
            0,   54,   73,   72,    0,   53,   66,    0,   57,   76,   68,   70,
           57,   71,   71],
        [  49,   60,   61,   64,   57,    0,   60,   57,    0,   75,   53,   71,
            0,   61,   66,    0,   40,   53,   68,   64,   57,   71,    0,   72,
           60,   57,   70,   57,    0,   60,   53,   56,    0,   67,   68,   57,
           66,   57,   56,    0,   53, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
         1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
         1000, 1000, 1000],
        [  57,   58,   58,   57,   55,   72,   61,   74,   57,   10,    0,   59,
           61,   74,   61,   66,   59,    0,   58,   73,   64,   64,    0,   74,
           53,   64,   73,   57,    0,   72,   67,    0,   72,   60,   57,    0,
           58,   67,   70,   65,   53,   64,    0,   57,   64,   57,   65,   57,
           66,   72,   71, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
         1000, 1000, 1000],
        [  65,   67,   71,   72,    0,   57,   76,   72,   70,   53,   67,   70,
           56,   61,   66,   53,   70,   77,    0,   72,   60,   61,   66,   59,
           71,    0,   53,   70,   57,    0,   60,   53,   68,   68,   57,   66,
           61,   66,   59,   12,    0,    2,    0,   32,   61,   58,   72,   77,
         1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
         1000, 1000, 1000]])], 
         'line_labels_len':未padding前每行label的长度 最长为63 padding到63
         [[60, 49, 40, 54, 63, 41, 51, 48]]} 
"""


class TrainerLineCTC(GenericTrainingManager):

    def __init__(self, params):
        super(TrainerLineCTC, self).__init__(params)

    def ctc_remove_successives_identical_ind(self, ind):  # 相邻相同的index变为1个 [1,1,1,2,3,3] ->[1,2,3]
        res = []
        for i in ind:
            if res and res[-1] == i:
                continue
            res.append(i)
        return res

    def train_batch(self, batch_data, metric_names):
        # print(batch_data, "batch_data是什么？？？？？？？？？？？？？？？？？？？？？")
        x = batch_data["imgs"].to(self.device) # x为imgs 且送入device
        y = batch_data["labels"].to(self.device) # y为label
        x_reduced_len = [s[1] for s in batch_data["imgs_reduced_shape"]]  # 取出 121 109 这些

        """
        'imgs_reduced_shape': 处理过后的img的shape 
          [[1, 121, 3], [2, 109, 3], [1, 96, 3], [2, 112, 3], [1, 127, 3], [2, 108, 3], [1, 114, 3], [2, 113, 3]], 
          """
        y_len = batch_data["labels_len"]  # label的长度
        loss_ctc = CTCLoss(blank=self.dataset.tokens["blank"], reduction="sum")  # TODO:这里的loss有待考究 sum和mean可能会对结果产生影响 用mean的话可能loss太小了？
        """
        ctcloss 参数说明：
        (1) blank: 空白标签所在的label值，默认为0，需要根据实际的标签定义进行设定；
        我们在预测文本时，一般都是有一个空白字符的，整个blank表示的就是空白字符在总字符集中的位置。
        (2) reduction: 处理output losses的方式，string类型，可选’none’ 、 ‘mean’ 及 ‘sum’，'none’表示对output losses不做任何处理，
        ‘mean’ 则对output losses (即输出的整个batch_size的损失做操作) 取平均值处理，‘sum’则是对output losses求和处理，默认为’mean’ 。

        """

        self.optimizer.zero_grad()
        x = self.models["encoder"](x)
        global_pred = self.models["decoder"](x)

        loss = loss_ctc(global_pred.permute(2, 0, 1), y, x_reduced_len, y_len)
        """
        loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
        (1)log_probs: shape=(T, N, C) 的模型输出张量，T: 表示输出的序列的长度； N: 表示batch_size值； C: 表示包含有空白标签的所有要预测的字符集总长度。
        (2)targets： shape=(N, S) 或 （sum(target_lengths)）的张量。其中对于第一种类型，N表示batch_size, S表示标签长度。如：shape =（32， 50），其中的32为batch_size, 50表示每个标签有50个字符。
        (3)input_lengths： shape为(N)的张量或元组，但每一个元素的长度必须等于T即输出序列长度，一般来说模型输出序列固定后则该张量或元组的元素值均相同；
        (4)target_lengths： shape为(N)的张量或元组，其每一个元素指示每个训练输入序列的标签长度，但标签长度是可以变化的；
        """
        self.backward_loss(loss)
        self.optimizer.step()
        pred = torch.argmax(global_pred, dim=1).cpu().numpy()
        # 计算metric
        metrics = self.compute_metrics(pred, y.cpu().numpy(), x_reduced_len, y_len, loss=loss.item(), metric_names=metric_names)
        return metrics

    def evaluate_batch(self, batch_data, metric_names):
        x = batch_data["imgs"].to(self.device)
        y = batch_data["labels"].to(self.device)
        x_reduced_len = [s[1] for s in batch_data["imgs_reduced_shape"]]
        y_len = batch_data["labels_len"]

        loss_ctc = CTCLoss(blank=self.dataset.tokens["blank"], reduction="sum")

        """
        
        """
        x = self.models["encoder"](x)
        global_pred = self.models["decoder"](x)

        loss = loss_ctc(global_pred.permute(2, 0, 1), y, x_reduced_len, y_len)   # 为什么做一个permute
        pred = torch.argmax(global_pred, dim=1).cpu().numpy()
        metrics = self.compute_metrics(pred, y.cpu().numpy(), x_reduced_len, y_len, loss=loss.item(), metric_names=metric_names)
        if "pred" in metric_names:
            metrics["pred"].extend([batch_data["unchanged_labels"], batch_data["names"]])
        return metrics

    def compute_metrics(self, x, y, x_len, y_len, loss=None, metric_names=list()):  # x是pred y是ground-truth
        batch_size = y.shape[0]  # 几个label
        ind_x = [x[i][:x_len[i]] for i in range(batch_size)]  # ind_x是预测的index
        # print(ind_x, "ind_x是什么==================================================")
        ind_y = [y[i][:y_len[i]] for i in range(batch_size)]  # ind_y是label的index
        # print(ind_y, "ind_y是什么、、、、、、、、、、、、、、、、、、、、、、、、、")
        ind_x = [self.ctc_remove_successives_identical_ind(t) for t in ind_x]  # 删除了相邻相同的index
        str_x = [LM_ind_to_str(self.dataset.charset, t, oov_symbol="") for t in ind_x]  # 取出labels中对应index的str
        # print(str_x, "str_x是什么‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘")  # str_x是预测出来的一句话
        str_y = [LM_ind_to_str(self.dataset.charset, t) for t in ind_y]  # str_y是label的text
        # print(str_y, "str_y是什么 ’。。。。。。。。。。。。。。。。。。。。。。。。。。")
        str_x = [re.sub("( )+", ' ', t).strip(" ") for t in str_x]
        # print(str_x, "处理之后的str_x是什么 、、、、、、、、、、、、、、、、、、、、、、‘’‘’‘’‘’‘’‘’‘’")
        metrics = dict()
        for metric_name in metric_names:
            if metric_name == "cer":
                metrics[metric_name] = [editdistance.eval(u, v) for u,v in zip(str_y, str_x)]
                metrics["nb_chars"] = nb_chars_from_list(str_y)
                # # 总字段多长 包含字母和特殊字符 不切分
                # # print(nb_chars_from_list(["sada//?", "das..ccc/"]))
            elif metric_name == "wer":
                metrics[metric_name] = edit_wer_from_list(str_y, str_x)
                metrics["nb_words"] = nb_words_from_list(str_y)  # 切分
            elif metric_name == "pred":
                metrics["pred"] = [str_x, ]
        if "loss_ctc" in metric_names:
            metrics["loss_ctc"] = loss / metrics["nb_chars"]
        metrics["nb_samples"] = len(x)
        return metrics  # 就是算cer wer loss这些 主要是editdistance



