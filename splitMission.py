import cv2
import os
import base64
from PIL import  Image
import PIL
import io
import json
import numpy as np
from  multiprocessing import  Pool
from generateLabel import *
import shutil

class imgToSplit():
    
    def __init__(self,imgFile):
        self.objName = imgFile.strip('.jpg')
        self.img = cv2.imread(imgFile)
        self.h,self.w = self.img.shape[:-1]
        self.half_h,self.half_w = self.h//2,self.w//2
        self.sub_json_list = []
        self.disp = np.array([ (0,0) , (self.half_w,0) , (0,self.half_h) , (self.half_w,self.half_h)])
        self.category = None  #在3rd顺便获得category

        print('executing',self.objName)
        self.splitImage()
        self.gen_4sub_json()
        self.justDrawLabel()
        self.saveSubJson()

    
    
    def splitImage(self):
        '''1st step ： split origin image into 4 pieces！'''


        cv2.imwrite(r'.\target\%s_1.jpg' % self.objName, self.img[:self.half_h, :self.half_w])
        cv2.imwrite(r'.\target\%s_2.jpg' % self.objName, self.img[:self.half_h, self.half_w:])
        cv2.imwrite(r'.\target\%s_3.jpg' % self.objName, self.img[self.half_h:, :self.half_w])
        cv2.imwrite(r'.\target\%s_4.jpg' % self.objName, self.img[self.half_h:, self.half_w:])

    def gen_4sub_json(self):
        '''2nd step : generate 4 json file to sub image!'''

        def apply_exif_orientation(image):
            try:
                exif = image._getexif()
            except AttributeError:
                exif = None

            if exif is None:
                return image

            exif = {
                PIL.ExifTags.TAGS[k]: v
                for k, v in exif.items()
                if k in PIL.ExifTags.TAGS
            }

            orientation = exif.get('Orientation', None)

            if orientation == 1:
                # do nothing
                return image
            elif orientation == 2:
                # left-to-right mirror
                return PIL.ImageOps.mirror(image)
            elif orientation == 3:
                # rotate 180
                return image.transpose(PIL.Image.ROTATE_180)
            elif orientation == 4:
                # top-to-bottom mirror
                return PIL.ImageOps.flip(image)
            elif orientation == 5:
                # top-to-left mirror
                return PIL.ImageOps.mirror(image.transpose(PIL.Image.ROTATE_270))
            elif orientation == 6:
                # rotate 270
                return image.transpose(PIL.Image.ROTATE_270)
            elif orientation == 7:
                # top-to-right mirror
                return PIL.ImageOps.mirror(image.transpose(PIL.Image.ROTATE_90))
            elif orientation == 8:
                # rotate 90
                return image.transpose(PIL.Image.ROTATE_90)
            else:
                return image

        def generateImgData(path):
            try:
                image_pil = PIL.Image.open(path)
            except IOError:
                pass

            # apply orientation to image according to exif
            image_pil = apply_exif_orientation(image_pil)

            with io.BytesIO() as f:
                image_pil.save(f, format='PNG')
                f.seek(0)
                raw = f.read()
                return base64.b64encode(raw).decode('utf8')


        for i in range(1, 5):
            content = dict()
            imgName = '%s_%i.jpg'%(self.objName,i)
            img = cv2.imread(r'.\target\%s' % imgName)

            content['version'] = '3.10.0'
            content['flags'] = dict()
            content['shapes'] = []
            content['lineColor'] = [0, 255, 0, 128]
            content['fillColor'] = [255, 0, 0, 128]
            content['imagePath'] = imgName
            content['imageData'] = generateImgData(r'.\target\%s' % imgName)

            h, w = img.shape[:-1]
            content['imageHeight'] = h
            content['imageWidth'] = w

            # content = json.dumps(content,indent=4)
            # print(content)

            # with open(r'.\target\%s' % imgName.replace('jpg', 'json'), 'w') as f:
            #     json.dump(content, f)
            self.sub_json_list.append(content)

    def justDrawLabel(self):
        '''3rd 将label图片分门别类 画出label'''


        with open('%s.json'%self.objName, 'r') as f:
            data = json.load(f)

        #获取类别
        classes = [shape['label'] for shape in  data['shapes']]
        self.category = tuple(set(classes))


        imgData = data['imageData']

        img = img_b64_to_arr(imgData)

        categories = pointsDivByLabel(data['shapes'])
        for cateName, shapes in categories.items():
            label_name_to_value = {'_background_': 0}

            for shape in sorted(shapes, key=lambda x: x['label']):
                label_name = shape['label']
                if label_name in label_name_to_value:
                    label_value = label_name_to_value[label_name]
                else:
                    label_value = len(label_name_to_value)
                    label_name_to_value[label_name] = label_value

            lbl = shapes_to_label(img.shape, shapes, label_name_to_value)
            self._mapPiecesToJson(cateName,lbl)
            # lblsave(r'.\labelme_temp\%s_label_%s.png' % (self.objName,cateName), lbl)


    def _mapPiecesToJson(self,cate,img):
        """ 4th 读取label_png图片 分割 并将每块 映射到对应的sub json"""

        def generateNewShape(labelName,pointSet):
            """生成新的字典 方便json添加到shapes列表"""
            res = dict()
            res['label'] = labelName
            res['line_color'] = None
            res['fill_color'] = None
            res['points'] = pointSet
            res['shape_type'] = 'polygon'

            return res




        # img = cv2.imread(r'.\labelme_temp\%s_label_%s.png' % (self.objName,cate),0)
        img = np.uint8(img)


        #横竖两道杠
        img[:, self.half_w - 2: self.half_w + 3] = 0
        img[self.half_h - 2:self.half_h + 3, :] = 0

        #找到轮廓并拟合
        _, contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        aprox = [cv2.approxPolyDP(c, 1, True) for c in contours]
        aprox = [a.reshape(-1,2) for a in aprox if len(a)>=3] #过滤掉长度不符合规范的点 并reshape
        aprox = [a for a in aprox if (np.std(a,axis=0) > 2).all() ] #过滤掉 噪声点 （纠结在一团的小块）



        for pointSet in aprox:

            #任意取出一点 作为判断 在哪个象限标注
            judgeX = pointSet[0,0]
            judgeY = pointSet[0,1]

            if judgeX < self.half_w and judgeY < self.half_h:
                order = 0
            elif judgeX > self.half_w and judgeY < self.half_h:
                order = 1
            elif judgeX < self.half_w and judgeY > self.half_h:
                order = 2
            elif judgeX > self.half_w and judgeY > self.half_h:
                order = 3

            #减去偏移量
            pointSet -= self.disp[order]


            #写入json
            self.sub_json_list[order]\
            ['shapes'].append(generateNewShape(cate,pointSet.tolist()))


            # os.remove(r'.\labelme_temp\%s_label_%s.png'% (self.objName,cate))






    def saveSubJson(self):
        '''最后将所有json保存'''
        for i in range(1,5):
            with open(r'.\target\%s_%i.json' %(self.objName,i), 'w') as f:
                json.dump(self.sub_json_list[i-1], f)







def mission(file):
    file = file.replace('JPG', 'jpg')
    imgToSplit(file)







if __name__ == '__main__':
    # mission = imgToSplit('DJI_0301.jpg')
    # mission.splitImage()
    # mission.gen_4sub_json()
    # mission.confirmNewPoints()
    # mission.saveSubJson()

    #存放目标图片
    if not os.path.exists(r'.\target'):
        os.mkdir(r'.\target')



    imgFiles = [ f for f in os.listdir('.') if 'jpg' in f or 'JPG' in f]
    # for file in imgFiles:
    #     file = file.replace('JPG','jpg')
    #     imgToSplit(file)
    pool = Pool(processes=4)
    pool.map(mission,imgFiles)
    print('done!')

