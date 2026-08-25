import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from preprocessing import caption_to_sequence,image_transform

class Flickr8kDataset(Dataset):
    def __init__(self,df,image_dir,word2idx,transform,max_len=25):
        self.df=df.reset_index(drop=True)
        self.image_dir=image_dir
        self.word2idx=word2idx
        self.transform=transform
        self.max_len=max_len
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        row=self.df.ilox[idx]
        img_path=os.path.join(self.image_dir,row['image'])
        image=Image.open(img_path).convert('RGB')
        image=self.transform(image)
        
        caption_seq=caption_to_sequence(row['caption'],self.word2idx,self.max_len)
        caption_tensor=torch.tensor(caption_seq,dtype=torch.long)
        return image,caption_tensor
def get_dataloaders(df,image_dir,word2idx,batch_size=32):
    unique_images=df['images'].unique()
    train_img,temp_img=train_test_split(unique_images,test_size=0.2,random_state=42)
    val_imgs,test_imgs=train_test_split(temp_img,test_size=0.5,random_state=42)
    
    train_df=df[df['image'].isin(train_img)]
    val_df=df[df['image'].isin(val_imgs)]
    test_df=df[df['image'].isin(test_imgs)]
    
    train_ds=Flickr8kDataset(train_df,image_dir,word2idx,image_transform)
    val_ds=Flickr8kDataset(val_df,image_dir,word2idx,image_transform)
    test_ds=Flickr8kDataset(test_df,image_dir,word2idx,image_transform)
    
    train_laoder=DataLoader(train_ds,batch_size=batch_size,shuffle=True)
    val_loader=DataLoader(val_ds,batch_size=batch_size,shuffle=False)
    test_loader=DataLoader(test_ds,batch_size=batch_size,shuffle=False)
    
    return train_laoder,val_loader,test_loader