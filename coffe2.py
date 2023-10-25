#todo

# 1. single training run on georgies images with resnet TL. 



import pandas as pd
import numpy as np
import os 
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torch.utils.data.dataloader import Dataset
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import math 
import numpy as np
import numpy.random as random
import pandas as pd
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torchvision
from torch.utils.data.dataloader import Dataset
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset
from torch.cuda.amp import autocast, GradScaler
from torchinfo import summary
from torchvision.io import read_image
from torchvision.models import resnet101, ResNet101_Weights
# from scrape_imgs import scrapper as scpr
from sklearn.model_selection import KFold
from PIL import Image
import torch.optim as optim
import gc
import torchmetrics

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

import tensorflow as tf
import datetime
from tensorboardX import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#ignore cameras 10 and 11



class macro_return_df():
    def __init__(self):
        self.read_spreadsheets()
        self.amend_cam_16_column()
        self.build_df()
            
    def read_spreadsheets(self):
        cam_list = []
        cam_list_csv = []
        base = 'Cam_'

        for i in range(1, 17):
            cam_list.append(base+str(i))

        spreadsheet_dict = {}

        for cam in cam_list:
            dir = os.path.join('/', 'Users', 'rhyscooper', 'Desktop', 'wildlifeCNN', 'animalCNN', f'{cam}.csv')
            spreadsheet_dict[cam] = pd.read_csv(dir)
        del spreadsheet_dict['Cam_10']
        del spreadsheet_dict['Cam_11']
        self.spreadsheet_dict =  spreadsheet_dict

    def amend_cam_16_column(self):
        c16 = self.spreadsheet_dict['Cam_16']
        c16 = c16.iloc[:, 1:]

        c16.insert(0, 'Camera no', 'Cam 16')
        self.spreadsheet_dict['Cam_16'] = c16


    def sanity_check(self):
        sanity_check = [800, 3406, 2540, 426, 4946, 3909, 4076, 3990, 2517, 2262, 2205, 444, 5636, 6624]
        sanity_check_sum = sum(sanity_check)
        # print("sanity check", sanity_check_sum,  len(sanity_check))
        # print(sanity_check())


    def build_df(self):
        complete_df = pd.concat(self.spreadsheet_dict.values(), axis=0)
        complete_df.reset_index(drop=True, inplace=True)
        self.complete_df = complete_df

df = macro_return_df()
df = df.complete_df

class Record():
    def __init__(self, dataframe):
        self.df = dataframe
        self.dir =  os.path.join('/', 'Users', 'rhyscooper', 'Desktop', 'wildlifeCNN', 'images',)
        self.num_rows, self.num_columns = self.df.shape
        
    def image_loader(self, index): 
        index_row = np.array(self.df.iloc[index]) 
        # print(index_row)
        filename = index_row[2]
        path = os.path.join (self.dir,filename)
        img = Image.open(path)
        species = index_row[5]
        # if show:
        #     plt.imshow(img)
        #     plt.axis('off')  # Turn off axis labels and ticks
        #     plt.show()
        # if path:
        #     return path, species
        # if not path:
        return img, species 
    
    def train_set_list(self, N_train_samples):
        self.N_train_samples = N_train_samples
        self.train_dir_list = []
        for i in range(0, N_train_samples):
            self.train_dir_list.append((self.image_loader(i)))
            
    def test_set_list(self):
        self.test_dir_list = []
        for i in range(self.N_train_samples, self.num_rows) :
            self.test_dir_list.append((self.image_loader(i)))        
            
    def test_set_list_override(self, N_train_samples_override):
        self.N_train_samples_overide = N_train_samples_override
        self.test_dir_list_override = []
        for i in range(0, N_train_samples_override):
            self.test_dir_list_override.append((self.image_loader(i)))
            
class CreateDataset():
    def __init__(self, df, train, N_train_samples):
        self.key_map = { np.nan:0, 'R':1, 'N':2, 'F':3, 'S':4, 'Owl':5, 'M':6, 'P':7, 'O':8, 'H':9, 'Bi':10, 'Bu':11, 'Jay':12 ,'W':13, 'Dog':14 ,'C':15,'B':16, 'cow': 17}
        self.label_map = {'Nan': 'Nan', 'R': 'rabbit', 'N': 'nothing', 'F':'fox', 'S':'squirrel', 'Owl': 'owl', 'M': 'muntjac', 'P': 'pheasent', 'O':'unidentifiable', 'H': 'hare', 'Bi': 'bird', 'Bu': 'buzzard', 'Jay': 'jay', 'W': 'wind', 'Dog': 'dog', 'C':'car', 'cow':'Cow' }
        self.label_set = {'rabbit', 'fox', 'squirrel', 'owl', 'muntjac', 'pheasent', 'hare', 'bird', 'buzzard', 'jay', 'dog', 'Dog', 'car', 'cow'}
        self.cleaned_df = self.clean_species_values(df)
        self.category_values = self.get_unique_categorical_values(self.cleaned_df)
        self.N_category_values = len(self.category_values)
        self.classification_mapped_df = self.classification_map(self.cleaned_df)
        self.dataset_df = self.classification_mapped_df
        
        
        self.dir =  os.path.join('/', 'Users', 'rhyscooper', 'Desktop', 'wildlifeCNN', 'images',)
        self.num_rows, self.num_columns = self.dataset_df.shape
        
        self.train = train
        self.N_train_samples = N_train_samples
        
        self.dataset = self.dataset_list_compiler(self.train)
    
    def clean_species_values(self, df):
        df['Species '] = df['Species '].str.replace(' ', '')
        # df.loc[df['Camera no'] == 'Cam 6', 'Species '] = 'cow'
        return df

    def get_unique_categorical_values(self, df, column_name = "Species "):
        if column_name in df.columns:
            unique_values = df[column_name].unique()
            return unique_values
        else:
            return []

    def classification_map(self, df):
        df['Species '] = df['Species '].replace(self.key_map)
        return df

    def image_and_label_loader(self, index): 
        index_row = np.array(self.dataset_df.iloc[index]) 
        # print(index_row)
        filename = index_row[2]
        path = os.path.join (self.dir,filename)
        img = Image.open(path)
        species = index_row[5]
        return img, species 
        
    def dataset_list_compiler(self, train):
        image_label_list = []
        if train:
            for i in range(0, self.N_train_samples):
                image_label_list.append((self.image_and_label_loader(i)))        
        if not train:
            image_label_list = []
            for i in range(self.N_train_samples, self.num_rows) :
                image_label_list.append((self.image_and_label_loader(i)))                

        return image_label_list 

class MyDataset2(torch.utils.data.Dataset):
    '''
    Required class for formatting a dataset and allowing the pytorch DataLoader to retreive images
    in the correct form. Takes as input a dataset of PIL images and a transform to allow the images 
    to be of the right format to be used in subsequent models. Pairs the images with their masks and
    turns them into tensors with shape [3, height, width].
    '''
    def __init__(self, dataset, preprocess):
        self.dataset = dataset
        # self.transform = A.Compose([A.Resize(520, 520, p=1),ToTensorV2()])
        self.transform = preprocess

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx][0]
        label = self.dataset[idx][1]
        # image_np = np.array(img)
        img = self.transform(img)
        

        return img, label
        
class RN_model():
    def __init__(self, n_labels, label_set):
        self.n_labels = n_labels
        self.label_set = label_set
        self.weights = ResNet101_Weights.DEFAULT
        self.model = resnet101(weights=self.weights)
        self.freeze_layers('layer4')
        self.add_layer()
        self.original_labels = self.weights.meta["categories"]
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        
    def freeze_layers(self, freeze_until_layer):
        for name, param in self.model.named_parameters():
            if freeze_until_layer in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
    def add_layer(self):
        self.model.fc = nn.Linear(self.model.fc.in_features, self.n_labels)

    def print_structure(self):
        print(summary(model=self.model, 
                    input_size=(10, 3, 520, 520),
                    col_names=["input_size", "output_size", "num_params", "trainable"],
                    col_width=20,
                    row_settings=["var_names"]
                )) 

    def get_single_pred(self):
        self.model.eval()
        img = read_image("/Users/rhyscooper/Desktop/wildlifeCNN/images/Cam 1/IMG_0412.jpg")
        preprocess = self.weights.transforms()
        batch = preprocess(img).unsqueeze(0)
        prediction =self.model(batch).squeeze(0).softmax(0)
        class_id = prediction.argmax().item()
        score = prediction[class_id].item()
        category_name = self.weights.meta["categories"][class_id]
        print(f"{category_name}: {100 * score:.1f}%")
         
    def train_loop(self, trainloader, num_epochs ):
        Path = os.path.join('/', 'Users', 'rhyscooper', 'Desktop', 'wildlifeCNN', 'animalCNN', 'tensorlogs')
        writer = Tensorboard_setup(path=Path)
        self.model.train()
        
        for epoch_n in range(1, num_epochs+1):
            e = epoch(epoch_n, trainloader, self.optimizer, self.model, self.criterion)
            e.run()
            writer.add_scalars('Loss_Accuracy', {'Loss': e.running_loss, 'Accuracy': e.accuracy}, global_step=epoch_n)
            
            writer.add_scalar('Loss', e.running_loss, global_step=epoch_n)
            writer.add_scalar('Accuracy', e.accuracy, global_step=epoch_n)
        writer.add_text('Y-Axis', 'Loss: Left, Accuracy: Right', global_step=num_epochs)

    def test(self, testloader):
        self.model.eval()
        test_e = epoch(1, testloader, self.optimizer, self.model, self.criterion, train = False)
        self.test_e = test_e
        test_ac, test_loss = test_e.run()
        print(f'test accuracy:{test_ac}, test loss {test_loss}')
        
    def translate_predictions(self, epoch, n_translations, label_map):
        true_labels = epoch.true_labels 
        predictions = epoch.predicted_labels 
        inverted_map = {value: key for key, value in label_map.items()}
        
        for i in range(0, n_translations):
            t_label = true_labels[i]
            pred_label = predictions[i]
            t_label_invert, pred_label_invert = inverted_map[t_label], inverted_map[pred_label]
            print(f'True{t_label_invert}, predictions {pred_label_invert}')
        
class epoch():
    def __init__(self, number, trainloader, model_optim, model_model, model_criterion, train = True, verbose=False):
        self.number = number
        self.trainloader = trainloader
        self.optimizer =  model_optim
        self.model = model_model
        # self.scheduler = model_scheduler
        self.criterion = model_criterion
        self.train = train 
        self.verbose = verbose 
        
    def run(self):
        self.running_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        predicted_labels = []
        true_labels = []
        
        for i, data in enumerate(self.trainloader, 0):
            inputs, labels = data

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
        
            loss = self.criterion(outputs, labels)
            if self.train:
                loss.backward()
                self.optimizer.step()
            # self.scheduler.step()

            # each row of outputs is a sample. each coloumn is the class logits
            # returns value and indicy so ignore the first 
            _, predicted = torch.max(outputs, 1)
            # the brackets part creates a tuple with indexed booblean values. the sum adds up the number of trues
            # item converts the tensor into an interger.
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

            self.running_loss += loss.item() 
            self.accuracy_2 = total_correct / total_samples
            self.accuracy = accuracy_score(true_labels, predicted_labels)
            # self.epoch_precision = precision_score(true_labels, predicted_labels)
            # self.epoch_recall = recall_score(true_labels, predicted_labels)
            # self.epoch_f1 = f1_score(true_labels, predicted_labels)
            if not self.train:
                print("ac2", self.accuracy_2)
                self.predicted_labels = predicted_labels
                self.true_labels = true_labels
                return self.accuracy, self.running_loss    
   
class general_metric_plotter():
    def __init__(self, epoch_storage: dict, loss= False, accuracy = False, precision= False, recall = False, f1=False):
        self.epoch_storage = epoch_storage
        self.running_loss = loss
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall 
        self.f1 = f1
        
    def create_x(self, atrib_list):
        x_series = [i for i in range(0, len(atrib_list))] 
        return x_series
    
    def sing_plot(self, x, y, title):
        fig1, ax  = plt.subplots()
        ax.plot(x, y, label = title)
        plt.title = title
        plt.legend()
        plt.show()
        
            
    def create_plots(self):
        for atrb in dir(self):
            if getattr(self, atrb) is True:
                y_series =[]
                for epoch in self.epoch_storage:
                    y_series.extend([getattr(epoch, atrb)])
                x_series = self.create_x(y_series)
                self.sing_plot(x_series, y_series, str(atrb))
                y_series.clear()
                                 
class Model():

    def __init__(self, model_name, optim, loss_type, output_shape=32, verbose=False):
        # Initialise class parameters
        self.model_name = model_name
        self.optimiser_type = optim
        self.loss_type = loss_type

        self.model_dic = {
        'DeepLabV3': [torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT,
                        torchvision.models.segmentation.deeplabv3_resnet101],
    
        'FCN' :      [torchvision.models.segmentation.FCN_ResNet101_Weights.DEFAULT,
                        torchvision.models.segmentation.fcn_resnet101],
    
        'LRASPP' :   [torchvision.models.segmentation.LRASPP_MobileNet_V3_Large_Weights.DEFAULT, 
                        torchvision.models.segmentation.lraspp_mobilenet_v3_large],
        
        'ResNet101': [resnet101, ResNet101_Weights],

        'IoU' :       torchmetrics.classification.MultilabelJaccardIndex(num_labels=output_shape).to(device),

        'PixAcc':     torchmetrics.classification.MultilabelAccuracy(num_labels=output_shape).to(device),
        }

        # Call the createModel function to return the specified model, and initialise accuracy metrics
        self.createModel(output_shape, verbose)
        
    def createModel(self, output_shape, verbose):
        """Pretrained semantic segmentation model with custom head
        Args:
            output_shape (int, optional): The number of output channels
            in your dataset masks. Defaults to 32.

            verbose (bool, optional): Print out the model architecture. 
            Default is False.

        Returns:
            model: Returns the desired model with either the ResNet101 (for DeepLabV3 and FCN), 
            or MobileNet (for LRASPP) backbone.
        """
        
        self.weights = self.model_dic[self.model_name][0]
        self.model = self.model_dic[self.model_name][1](weights=self.weights)
        self.auto_transform = self.weights.transforms()
        
        # Freeze pretrained "backbone" layers
        if self.model_name == 'LRASPP':
            for name, param in self.model.named_parameters():
                if "backbone" in name:
                    if "14" in name or "15" in name or "16" in name:
                        pass     
                    else:
                        param.requires_grad = False

        if self.model_name == 'DeepLabV3' or self.model_name == 'FCN':
            for name, param in self.model.named_parameters():
                if "backbone" in name:
                    if "layer3" in name or "layer4" in name:
                        pass     
                    else:
                        param.requires_grad = False        


        # Replace the last classifier layer with a Conv2d layer with the correct output shape
        # If the model has an auxiliary classifier, replace the last classifier layer with 32 output channels
        
        if self.model_name == 'DeepLabV3':
            self.model.classifier[-1] = nn.Conv2d(256, output_shape, kernel_size=1, stride=1)
            try:
                self.model.aux_classifier[-1] = nn.Conv2d(256, output_shape, kernel_size=1, stride=1)
                self.model.aux_classifier.add_module('softmax', nn.Softmax(dim=1))
            except:
                pass
        
        if self.model_name == 'FCN':
            self.model.classifier[-1] = nn.Conv2d(512, output_shape, kernel_size=1, stride=1)
            try:
                self.model.aux_classifier[-1] = nn.Conv2d(256, output_shape, kernel_size=1, stride=1)
                self.model.aux_classifier.add_module('softmax', nn.Softmax(dim=1))
            except:
                pass  

        if self.model_name == 'LRASPP':
            self.model.classifier.high_classifier = nn.Conv2d(128, output_shape, kernel_size=1, stride=1)
            self.model.classifier.low_classifier = nn.Conv2d(40, output_shape, kernel_size=1, stride=1)
            try:
                self.model.aux_classifier[-1] = nn.Conv2d(256, output_shape, kernel_size=1, stride=1)
                self.model.aux_classifier.add_module('softmax', nn.Softmax(dim=1))
            except:
                pass

        #Create optimiser and learning rate scheduler
        params = [p for p in self.model.parameters() if p.requires_grad]

        if self.model_name == 'DeepLabV3' or self.model_name == 'FCN':
            if self.optimiser_type == 'SGD':
                self.optimiser = torch.optim.SGD(params, lr=0.0001, momentum=0.9, weight_decay=0.0005)
            if self.optimiser_type == 'Adam': 
                self.optimiser = torch.optim.Adam(params, lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
            if self.optimiser_type == 'RMSprop':
                self.optimiser = torch.optim.RMSprop(params, lr=0.0001, alpha=0.99, eps=1e-08, weight_decay=0.001, momentum=0.9)
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimiser, step_size=4, gamma=0.01) 

        if self.model_name == 'LRASPP':
            if self.optimiser_type == 'SGD':
                self.optimiser = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
            if self.optimiser_type == 'Adam': 
                self.optimiser = torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
            if self.optimiser_type == 'RMSprop':
                self.optimiser = torch.optim.RMSprop(params, lr=0.001, alpha=0.99, eps=1e-08, weight_decay=0.001, momentum=0.9)
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimiser, step_size=3, gamma=0.01)

        # Initialise either the weighted Cross Entropy Loss or unweighted
        if self.loss_type == 'Standard_CEL':
            self.loss = nn.CrossEntropyLoss()
        if self.loss_type == 'Weighted_CEL':
            self.loss = nn.CrossEntropyLoss(weight=class_weights)
        
        #Optionally print out the new model architecture
        if verbose:
            print(summary(model=self.model, 
                input_size=(10, 3, 520, 520),
                col_names=["input_size", "output_size", "num_params", "trainable"],
                col_width=20,
                row_settings=["var_names"]
            )) 

        return self.model

def Tensorboard_setup(path):
    writer = SummaryWriter(path)
    return writer

# scpr()
# # type, animal, then this list 'rabbit', 'fox', 'squirrel', 'owl', 'muntjac', 'pheasent', 'hare', 'bird', 'buzzard', 'jay', 'dog', 'cow'
# scpr()
# # do car, transportation and then 'car'


train_dataset = CreateDataset(df, True, 30)
my_labels = train_dataset.label_set

m1 = RN_model(n_labels=18, label_set= my_labels)
og_labels = set(m1.original_labels)

intersection_set = og_labels.intersection(my_labels)
print(intersection_set)
# key_map =train_dataset.key_map

# train_set = train_dataset.dataset 

# # test_dataset = CreateDataset(df, True, 30)
# # test_set = test_dataset.dataset

# preprocess = m1.weights.transforms()
# dataset1= MyDataset2(train_set, preprocess=preprocess)
# # dataset2 = MyDataset2(test_set, preprocess=preprocess)

# trainLoader = DataLoader(dataset1, batch_size=50, shuffle=False, num_workers=0)
# # testLoader = DataLoader(dataset2, batch_size = 30, shuffle=False,num_workers=0)
# output_shape = train_dataset.N_category_values 

# m1.train_loop(trainloader=trainLoader, num_epochs=5)
# m1.test(testloader=trainLoader)
# m1.translate_predictions(m1.test_e, 10, label_map=key_map)