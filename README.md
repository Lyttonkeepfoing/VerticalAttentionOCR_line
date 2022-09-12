# Vertical_att_line_OCR

本代码来自 https://github.com/FactoDeepLearning/VerticalAttentionOCR


现阶段focus on lines-level task # paragraph-level 是这篇工作的核心 后续会更新


基本每行代码 每个函数 每个类都写了详细注释 帮助理解整个代码结构及训练细节


需要注意的是：
1.环境配置问题 该论文的代码使用了混合精度训练方法 需要安装apex


apex 安装方法：
1)conda create -n YOUR NAME python=3.8.10  # python版本需要注意


2)安装对应cuda的pytorch
3)安装git lfs (最好使用lfs)
4)安装apex  
git lfs clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir ./
5)安装paper提供的requirements.txt
pip install -r requirements.txt
环境建好之后 python main.py 即可开始lines-level task的训练

2.本代码针对paragraph-level task 设置了5000epochs 需要注意训练时间

3.paragraph-level的训练最好基于lines-level的pretrained weights 否则非常难收敛 


具体设置在main.py "model_params" "transfer_learning"


相关pretrained weights 可以从此处下载 https://git.litislab.fr/dcoquenet/VerticalAttentionNetwork

4.TODO..

Contact me:
liyuting_nlp@outlook.com
