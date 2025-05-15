import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import math
import torch.utils.data as data
import numpy as np
import os
import torch.nn as nn
from tqdm import tqdm
import argparse
from osgeo import gdal
# import gdal
from skimage import io
from HEdataset import PredictDataset
from HEnetwork12 import ConvAndTransformer
# from HEnetwork15 import SQSFormer


from torch.utils.data import Dataset, DataLoader
import time

def gettifinfo(tifpath):
    tifpath=tifpath
    ds=gdal.Open(tifpath)
    im_data=ds.ReadAsArray()
    im_geotrans = ds.GetGeoTransform()
    im_proj = ds.GetProjection()
    im_width = ds.RasterXSize  #栅格矩阵的列数
    im_height = ds.RasterYSize
    return im_geotrans,im_proj,im_width,im_height,im_data

def numpy2tif(parameter1_im_data,path,im_geotrans,im_proj,im_width,im_height):
    driver = gdal.GetDriverByName("GTiff") #数据类型必须有，因为要计算需要多大内存空间GTiff是tif格式的数据类型
    dataset=driver.Create(path,im_width,im_height,1, gdal.GDT_Float32,options=["INTERLEAVE=PIXEL"])###需要知道文件名，宽度，高度，波段数
    dataset.SetGeoTransform(im_geotrans)#####写入仿射变换参数
    # print(im_geotrans)
    dataset.SetProjection(im_proj)  #####写入投影
    
    band=dataset.GetRasterBand(1)
    band.WriteArray(parameter1_im_data) #写入数组数据
    nodata=parameter1_im_data[0][0]
    band.SetNoDataValue(-999)

    dataset.FlushCache()####确保数据写入磁盘
    for i in range(1,2):###统计波段数据
            band=dataset.GetRasterBand(i)
            # band.GetRasterBand(i).ComputeStatistics(False)
            # band
    dataset.BuildOverviews("average",[2,4,8,16,32,64])
    del dataset



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagefile',  default="../FeatureImage/...tif",type=str) 

    parser.add_argument('--nodata', type=float, default=-888.0)
    parser.add_argument('--batch_size', type=int, default=256)

    parser.add_argument('--device', type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()

    im_geotrans,im_proj,im_width,im_height,image_data=gettifinfo(args.imagefile)
    predict_dataset = PredictDataset(image_data)
    predict_loader = DataLoader(predict_dataset,batch_size=args.batch_size,shuffle=False)

    print(len(predict_loader))
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # print(image_data.shape)
    # model=SQSFormer()
    model=ConvAndTransformer().to(device)


    model=torch.load("./model/...pkl")
    model = model.to(device)
    model.eval()

    # begin = time.time()  # Start the timing
    result = np.zeros((image_data.shape[1], image_data.shape[2]))
    with torch.no_grad():
        # result = np.zeros((image_data.shape[1], image_data.shape[2]))
        # distence=np.zeros((im_dataA.shape[0],im_dataA.shape[1]))
        for index,(data,coordinate) in enumerate(predict_loader):
            print(index)
            data=data.to(device)
            # print(data.shape)
            # print(dataA)
            # print(dataB)
            output=model(data)
            predicted = output.squeeze().cpu().numpy()
            # _,predicted = torch.max(output, dim=1)
            # print(predicted)
            # print(len(predicted))
            # # euclidean_distance = F.pairwise_distance(output1, output2)
            # # print(euclidean_distance.shape)
            # # print(euclidean_distance)
            for i in range(len(predicted)):
                x = coordinate[0][i].item()
                y = coordinate[1][i].item()
                result[x, y] = predicted[i]
            # for idd, i in enumerate(range(len(predicted))):
            #     x=coordinate[0][i].numpy()
            #     y=coordinate[1][i].numpy()
            #
            #     # print(predicted[idd].cpu().numpy())
            #     result[x,y]=predicted[idd].cpu().numpy()

        path = "./...tif"
        numpy2tif(result, path, im_geotrans, im_proj, im_width, im_height)
        #
        # end = time()
        # print(end-begin)

