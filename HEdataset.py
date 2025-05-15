import numpy as np
import torch
import torch.utils
import torch.utils.data
import os
import random
import numpy as np
from skimage import io
import shapefile
from osgeo import gdal
import torch.utils.data as data
def get_dataset(args):
    gt=np.asarray(open_file(args.y_path), dtype='float32')
    gt[gt==np.min(gt)]=args.nodata
    
    x=np.zeros((len(args.x_list),gt.shape[0],gt.shape[1]))
    for i in range(len(args.x_list)):
        x1=np.asarray(open_file(args.x_list[i]), dtype='float32')
        x1[x1==np.min(x1)]=args.nodata
        x[i,:,:]=x1
        del x1
    return x,gt



def open_file(dataset):
    _, ext = os.path.splitext(dataset)
    ext = ext.lower()
    if ext == '.tif' or ext == '.tiff':
        return io.imread(dataset)
    else:
        raise ValueError("The data you entered is incorrect")
    
def sample_gt(x,gt, args):

    im_width=gt.shape[1] ##136
    im_height=gt.shape[0] ##168 

    step=args.step
    sample_percentage=args.sample_percentage
    Middle_position_coordinates=[]
    train_label=np.zeros(gt.shape)
    val_label=np.zeros(gt.shape)

    for i in range(1, im_width-1,step):
        for j in range(0, im_height-1,step):
            kernel_patch=gt[j-1:j+2,i-1:i+2]
            if (gt[j,i]!= args.nodata) :
                Middle_position_coordinates.append([j, i])

    trainlist=random.sample(Middle_position_coordinates,int(len(Middle_position_coordinates)*sample_percentage))
    vallist=[i for i in Middle_position_coordinates if i not in trainlist]

    for i in trainlist:
        train_label[i[0],i[1]]=1
    for j in vallist:
        val_label[j[0],j[1]]=1


    return train_label,val_label

class TrainDataset(torch.utils.data.Dataset):


    def __init__(self, shp_path,image_data,height_data,im_geotrans,args):
        super(TrainDataset, self).__init__()
        '''
        x represents the position, 
        so xdata is used here to represent the independent variable data
        '''
        self.args=args
        self.image_data = image_data
        self.shp_path = shp_path
        self.height_data=height_data
        self.im_geotrans = im_geotrans

        self.coor=[]

        sf = shapefile.Reader(shp_path)
        shapes=sf.shapes()
        print(len(shapes))
        xOrigin = im_geotrans[0]
        yOrigin = im_geotrans[3]
        pixelWidth = im_geotrans[1]
        pixelHeight = im_geotrans[5]
        sample=[]
        for shp in range(len(shapes)):
            shap = shapes[shp]
            for i in range(len(shap.points)):
                x_coor=float(shap.points[i][0])
                y_coor=float(shap.points[i][1])
                xOffset = int((x_coor - xOrigin) / pixelWidth)
                yOffset = int((y_coor - yOrigin) / pixelHeight)
                self.coor.append([xOffset,yOffset])

    def __len__(self):

        return len(self.coor) 

    def __getitem__(self, i):
        [xOffset,yOffset] = self.coor[i]

        data = self.image_data[:,yOffset-2:yOffset+3,xOffset-2:xOffset+3]
        label = self.height_data[yOffset,xOffset]


        data = np.asarray(np.copy(data), dtype='float32')
        label = np.asarray(np.copy(label), dtype='float32')

        data=torch.from_numpy(data)
        label=torch.from_numpy(label)


        return data,label
def gettifinfo(tifpath):
    tifpath=tifpath
    ds=gdal.Open(tifpath)
    im_data=ds.ReadAsArray()
    im_geotrans = ds.GetGeoTransform()
    im_proj = ds.GetProjection()
    im_width = ds.RasterXSize  #栅格矩阵的列数
    im_height = ds.RasterYSize
    return im_geotrans,im_proj,im_width,im_height,im_data


class PredictDataset(torch.utils.data.Dataset):


    def __init__(self,image):
        super(PredictDataset, self).__init__()

        self.image = image

        self.patch_size = 5
        self.step=1
        self.indices=[]

        for i in range(int(self.patch_size/2),self.image.shape[1]-int(self.patch_size/2),self.step):
            for j in range(int(self.patch_size/2),self.image.shape[2]-int(self.patch_size/2),self.step):
                self.indices.append([i,j])


    def __len__(self):

        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i][0],self.indices[i][1]
        x1, y1 = x - int(self.patch_size/2), y - int(self.patch_size/2)
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        # print("self.image.shape",self.image.shape)

        data = self.image[:,x1:x2, y1:y2]   
        # print("data",data.shape)

        data = np.asarray(np.copy(data), dtype='float32')
        # data=data.transpose(2,0,1)


        data = torch.from_numpy(data)
        # print(dataA.shape)
        

        return data,(x,y)

# if __name__ == '__main__':
#     file="../FeatureImage/vir_NES3.tif"
#     im_geotrans,im_proj,im_width,im_height,image_data=gettifinfo(args.imagefile)
#     print(image_data.shape)

#     Train_dataset=TrainDataset(file,image_data,height_data,im_geotrans,args)
#     Train_loader = data.DataLoader(Train_dataset,batch_size=3,shuffle=True)
#     for i in Train_loader:
        # pass
        # # print(i[0].shape,i[1].shape)
        # # print(i)
