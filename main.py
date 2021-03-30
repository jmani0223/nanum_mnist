import os
import cv2
import numpy as np
def hangulFilePathImageRead ( filePath ) :
    stream = open( filePath.encode("utf-8"), "rb")
    bytes = bytearray(stream.read())
    numpyArray = np.asarray(bytes, dtype=np.uint8)
    return cv2.imdecode(numpyArray , cv2.IMREAD_UNCHANGED)

def dataload(path_dir):
    dataset = {}
    dir_list = os.listdir(path_dir)
    print(dir_list)
    for dir in dir_list:
        data = []
        dir_list_2 = os.listdir(path_dir +'/'+ dir)
        for filename in dir_list_2:
            fname = hangulFilePathImageRead(path_dir + '/' + dir + '/' + filename)
            img = cv2.imread(path_dir +'/' + dir + '/' +fname)

            #img = cv2.resize(img, dsize=(256,256), interpolation=cv2.INTER_AREA) #사이즈 변경
            data.append(img)
        dataset[dir] = data
    return dataset

dataset = dataload('./img')
print(dataset)