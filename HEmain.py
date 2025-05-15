import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import warnings
warnings.filterwarnings("ignore")
import torch
import random
import torch.utils.data as data
import numpy as np
from HEdataset import get_dataset,sample_gt,TrainDataset
import argparse
import torch.nn as nn
import torch
# import gdal
import shapefile
from osgeo import gdal
import torch.optim as optim
from HEtrain import train
from HEnetwork import ConvAndTransformer

def gettifinfo(tifpath):
    tifpath=tifpath
    ds=gdal.Open(tifpath)
    im_data=ds.ReadAsArray()
    im_geotrans = ds.GetGeoTransform()
    im_proj = ds.GetProjection()
    im_width = ds.RasterXSize  
    im_height = ds.RasterYSize
    return im_geotrans,im_proj,im_width,im_height,im_data

def GetPixelsandSample(shp_path,im_all_data,height_data,im_geotrans):

	Positive_sample=[]
	Negative_sample=[]
	xOrigin = im_geotrans[0]
	yOrigin = im_geotrans[3]
	pixelWidth = im_geotrans[1]
	pixelHeight = im_geotrans[5]
	# datasource=ogr.Open(shp_path)
	sf = shapefile.Reader(shp_path)
	shapes=sf.shapes()
	print(len(shapes))
	sample=[]
	for shp in range(len(shapes)):
		shap = shapes[shp]
		print(len(shap.points))
		for i in range(len(shap.points)):
			x_coor=float(shap.points[i][0])
			y_coor=float(shap.points[i][1])
			xOffset = int((x_coor - xOrigin) / pixelWidth)
			yOffset = int((y_coor - yOrigin) / pixelHeight)
			sample.append(im_all_data[:,xOffset,yOffset])
			sample.append(height_data[xOffset,yOffset])
	return sample

def Split_data(Positive_sample,Negative_sample,train_number,test_number):

	temp_pos_sample=random.sample(Positive_sample,int(train_number/2))
	temp_neg_sample=random.sample(Negative_sample,int(train_number/2))

	train_sample=temp_pos_sample + temp_neg_sample

	test_sample=[i for i in Positive_sample if not i in temp_pos_sample] + [j for j in Negative_sample if not j in temp_neg_sample]

	train_sample=np.array(train_sample)
	test_sample=np.array(test_sample)
	# print(train_sample.shape)
	x_train=train_sample[:,0:8]
	y_train=train_sample[:,8:9].flatten()
	x_pred=test_sample[:,0:8]
	y_pred=test_sample[:,8:9].flatten()
	



if __name__ == '__main__':
	parser = argparse.ArgumentParser()


	parser.add_argument('--TrV_pointfile',  default="../TrainingGCP/...shp",type=str)
	parser.add_argument('--V_pointfile',  default="../ValidationGCP/...shp",type=str)
	parser.add_argument('--imagefile',  default="../FeatureImage/...tif",type=str) 
	parser.add_argument('--heightfile',  default="../RealHeight/...tif",type=str)

	parser.add_argument('--lr', type=float, default=0.0003) 
	parser.add_argument('--nodata', type=float, default=-888.0)
	parser.add_argument('--patch_size', type=int, default=1)
	parser.add_argument('--step', type=int, default=1)
	parser.add_argument('--center_pixel', type=bool, default=True)
	parser.add_argument('--sample_percentage', type=float, default=0.7)
	parser.add_argument('--Train_batch_size', type=int, default=64)
	parser.add_argument('--Val_batch_size', type=int, default=1)
	parser.add_argument('--Test_batch_size', type=int, default=1)
	parser.add_argument('--num_epochs', type=int, default=300)
	parser.add_argument('--seed', type=int, default=123)
	parser.add_argument('--device', type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
	args = parser.parse_args()

	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)


	im_geotrans,im_proj,im_width,im_height,image_data=gettifinfo(args.imagefile)
	print(image_data.shape)


	im_geotrans,im_proj,im_width,im_height,height_data=gettifinfo(args.heightfile)
	print(height_data.shape)

	Train_dataset=TrainDataset(args.TrV_pointfile,image_data,height_data,im_geotrans,args)
	Val_dataset=TrainDataset(args.V_pointfile,image_data,height_data,im_geotrans,args)
	# Test_dataset=TrainDataset(args.Te_pointfile,image_data,height_data,im_geotrans,args)

	Train_loader = data.DataLoader(Train_dataset,batch_size=args.Train_batch_size,shuffle=True)
	Val_loader = data.DataLoader(Val_dataset,batch_size=args.Val_batch_size,shuffle=True)
	# Test_loader = data.DataLoader(Test_dataset,batch_size=args.Test_batch_size,shuffle=True)

	for data, target in Train_loader:
		data, target = data.to(args.device), target.to(args.device)



	net=ConvAndTransformer().to(args.device)  # 确保模型在指定设备上

	optimizer = optim.Adam(net.parameters(), lr=args.lr,weight_decay=0.0001)
	torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.8, last_epoch=-1)

	criterion=nn.MSELoss()


	train(args,Train_loader,Val_loader,net,optimizer,criterion)
