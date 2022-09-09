import itertools
import numpy as np
from skimage import transform as stf
from numpy import random, floor
from PIL import Image, ImageOps
from cv2 import erode, dilate



class SignFlipping:  # 顔色反轉
    """
    Color inversion
    """

    def __init__(self):
        pass

    def __call__(self, x):
        return ImageOps.invert(x)

"""
img = Image.open("/home/lyt/demospace/Vertical_att_line_OCR/basic/20200404211850_avnbr.jpg")
# img = img.convert('1') # 黑白
img = img.convert('L')
signflipping = SignFlipping()
signflipping(img)
img = img.convert('1')
img.show()
"""

class DPIAdjusting:  # 像素縮放
    """
    Resolution modification
    """

    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        w, h = x.size
        return x.resize((int(np.ceil(w * self.factor)), int(np.ceil(h * self.factor))), Image.BILINEAR)
"""
Image.BILINEAR
1.两次线性插值算法是一种通过平均周围像素颜色来添加像素的方法，该方法可生成中等品质的图像。
2.两次线性插值算法‘输出的图像的每个像素都是原图中四个像素（2x2)运算的结果’，由于它是从原图四个像素中运算的，因此这种算法很大程度上消除了锯齿现象，而且效果也比较好。
"""
"""
np.ceil 计算大于等于改值的最小整数
img = Image.open("/home/lyt/demospace/Vertical_att_line_OCR/basic/20200404211850_avnbr.jpg")
# img = DPIAdjusting(1.25)(img)  # 1280*2275 pixels
img = DPIAdjusting(0.75)(img)   # 768*1365 pixels
img.show()
"""

class Dilation:
    """
    OCR: stroke width increasing
    """

    def __init__(self, kernel, iterations):
        self.kernel = np.ones(kernel, np.uint8)  # uint8类型存储图像
        self.iterations = iterations

    def __call__(self, x):
        return Image.fromarray(dilate(np.array(x), self.kernel, iterations=self.iterations))

"""
Image.fromarray 
img = np.asarray(image) img to array
Image.fromarray(np.uint8(img)) array2img
didilate()函数可以对输入图像用特定结构元素进行膨胀操作，该结构元素确定膨胀操作过程中的邻域的形状，各点像素值将被替换为对应邻域上的最大值

img = Image.open("/home/lyt/demospace/Vertical_att_line_OCR/basic/下载.jpg")
# img = Dilation(3, 1)(img)  # 笔画边界变浅 越大越浅
img = Dilation(1, 20)(img)  # 肉眼无变化
img.show()
"""

class Erosion: # 同上
    """
    OCR: stroke width decreasing
    """

    def __init__(self, kernel, iterations):
        self.kernel = np.ones(kernel, np.uint8)
        self.iterations = iterations

    def __call__(self, x):
        return Image.fromarray(erode(np.array(x), self.kernel, iterations=self.iterations))


#TODO: 没看懂
class ElasticDistortion: # 弹性变形？
    """
    Elastic Distortion adapted from https://github.com/IntuitionMachines/OrigamiNet
    Used in "OrigamiNet: Weakly-Supervised, Segmentation-Free, One-Step, Full Page TextRecognition by learning to unfold",
        Yousef, Mohamed and Bishop, Tom E., The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020
    """
    def __init__(self, grid, magnitude, min_sep):

        self.grid_width, self.grid_height = grid
        self.xmagnitude, self.ymagnitude = magnitude
        self.min_h_sep, self.min_v_sep = min_sep

    def __call__(self, x):
        w, h = x.size

        horizontal_tiles = self.grid_width
        vertical_tiles = self.grid_height

        width_of_square = int(floor(w / float(horizontal_tiles)))
        height_of_square = int(floor(h / float(vertical_tiles)))

        width_of_last_square = w - (width_of_square * (horizontal_tiles - 1))
        height_of_last_square = h - (height_of_square * (vertical_tiles - 1))

        dimensions = []
        shift = [[(0, 0) for x in range(horizontal_tiles)] for y in range(vertical_tiles)]

        for vertical_tile in range(vertical_tiles):
            for horizontal_tile in range(horizontal_tiles):
                if vertical_tile == (vertical_tiles - 1) and horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif vertical_tile == (vertical_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])
                else:
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])

                sm_h = min(self.xmagnitude, width_of_square - (self.min_h_sep + shift[vertical_tile][horizontal_tile - 1][
                    0])) if horizontal_tile > 0 else self.xmagnitude
                sm_v = min(self.ymagnitude, height_of_square - (self.min_v_sep + shift[vertical_tile - 1][horizontal_tile][
                    1])) if vertical_tile > 0 else self.ymagnitude

                dx = random.randint(-sm_h, self.xmagnitude)  # 返回一个随机整型数，范围从低（包括）到高（不包括），即[low, high)
                dy = random.randint(-sm_v, self.ymagnitude)
                shift[vertical_tile][horizontal_tile] = (dx, dy)

        shift = list(itertools.chain.from_iterable(shift))  # 将shift列表转化为chain迭代器

        last_column = []
        for i in range(vertical_tiles):
            last_column.append((horizontal_tiles - 1) + horizontal_tiles * i)

        last_row = range((horizontal_tiles * vertical_tiles) - horizontal_tiles, horizontal_tiles * vertical_tiles)

        polygons = []
        for x1, y1, x2, y2 in dimensions:
            polygons.append([x1, y1, x1, y2, x2, y2, x2, y1])

        polygon_indices = []
        for i in range((vertical_tiles * horizontal_tiles) - 1):
            if i not in last_row and i not in last_column:
                polygon_indices.append([i, i + 1, i + horizontal_tiles, i + 1 + horizontal_tiles])

        for id, (a, b, c, d) in enumerate(polygon_indices):
            dx = shift[id][0]
            dy = shift[id][1]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[a]
            polygons[a] = [x1, y1,
                           x2, y2,
                           x3 + dx, y3 + dy,
                           x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[b]
            polygons[b] = [x1, y1,
                           x2 + dx, y2 + dy,
                           x3, y3,
                           x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[c]
            polygons[c] = [x1, y1,
                           x2, y2,
                           x3, y3,
                           x4 + dx, y4 + dy]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[d]
            polygons[d] = [x1 + dx, y1 + dy,
                           x2, y2,
                           x3, y3,
                           x4, y4]

        generated_mesh = []
        for i in range(len(dimensions)):
            generated_mesh.append([dimensions[i], polygons[i]])

        self.generated_mesh = generated_mesh

        return x.transform(x.size, Image.MESH, self.generated_mesh, resample=Image.BICUBIC)  # 双三次插值


"""
仿射变换：https://www.matongxue.com/madocs/244/
Elastic Distortion：https://zhuanlan.zhihu.com/p/342274228
img = Image.open("/home/lyt/demospace/Vertical_att_line_OCR/basic/20200404211850_avnbr.jpg")
img = ElasticDistortion(grid=(20, 20), magnitude=(20, 20), min_sep=(1, 1))(
                    img)
img.show()  
"""
# TODO:参数没调明白 grid增大 可以弯曲严重



class RandomTransform:
    """
    Random Transform adapted from https://github.com/IntuitionMachines/OrigamiNet
    Used in "OrigamiNet: Weakly-Supervised, Segmentation-Free, One-Step, Full Page TextRecognition by learning to unfold",
        Yousef, Mohamed and Bishop, Tom E., The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020
    """
    def __init__(self, val):

        self.val = val

    def __call__(self, x):
        w, h = x.size

        dw, dh = (self.val, 0) if random.randint(0, 2) == 0 else (0, self.val)

        def rd(d):
            return random.uniform(-d, d)

        def fd(d):
            return random.uniform(-dw, d)

        # generate a random projective transform
        # adapted from https://navoshta.com/traffic-signs-classification/
        tl_top = rd(dh)
        tl_left = fd(dw)
        bl_bottom = rd(dh)
        bl_left = fd(dw)
        tr_top = rd(dh)
        tr_right = fd(min(w * 3 / 4 - tl_left, dw))
        br_bottom = rd(dh)
        br_right = fd(min(w * 3 / 4 - bl_left, dw))

        tform = stf.ProjectiveTransform()
        tform.estimate(np.array((        #从对应点估计变换矩阵
            (tl_left, tl_top),
            (bl_left, h - bl_bottom),
            (w - br_right, h - br_bottom),
            (w - tr_right, tr_top)
        )), np.array((
            [0, 0],
            [0, h - 1],
            [w - 1, h - 1],
            [w - 1, 0]
        )))

        # determine shape of output image, to preserve size
        # trick take from the implementation of skimage.transform.rotate
        corners = np.array([
            [0, 0],
            [0, h - 1],
            [w - 1, h - 1],
            [w - 1, 0]
        ])

        corners = tform.inverse(corners)  # 从目标坐标逆推源坐标
        minc = corners[:, 0].min()
        minr = corners[:, 1].min()
        maxc = corners[:, 0].max()
        maxr = corners[:, 1].max()
        out_rows = maxr - minr + 1
        out_cols = maxc - minc + 1
        output_shape = np.around((out_rows, out_cols))

        # fit output image in new shape
        translation = (minc, minr)
        tform4 = stf.SimilarityTransform(translation=translation)  # 相似变换
        tform = tform4 + tform
        # normalize
        tform.params /= tform.params[2, 2]

        x = stf.warp(np.array(x), tform, output_shape=output_shape, cval=255, preserve_range=True)
        x = stf.resize(x, (h, w), preserve_range=True).astype(np.uint8)

        return Image.fromarray(x)

"""
img = Image.open("/home/lyt/demospace/Vertical_att_line_OCR/basic/20200404211850_avnbr.jpg")
img = RandomTransform(5)(img) # 没看出效果 具体参数作用没弄懂 看后面aug params
img.show() 

random.uniform(5, 6, size=(2,3))  
array([[5.82416021, 5.68916836, 5.89708586],
       [5.63843125, 5.22963754, 5.4319899 ]])

"""