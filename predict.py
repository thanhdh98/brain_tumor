import nibabel as nib
import torch
import numpy as np
from architectures.unet3d import UNet3d


def predict(images, model, device):
    imgs = images.to(device)
    logits = model(imgs)
    return logits


def normalize_image(img):
    # print(img)
    img = (img-np.min(img))/(np.max(img)-np.min(img))
    return img


if __name__ == "__main__":
    # load model
    pretrained_model_path = 'C:\\Users\\thanhdh6\\Documents\project\\brain_tumor_segmentation\\models\\saved_model_at_epoch_4_tv.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    unet3d = UNet3d(in_channels=4, n_classes=3, n_channels=24).to(device)
    unet3d.load_state_dict(torch.load(pretrained_model_path, map_location=torch.device(device)))
    unet3d.eval()
    # load 3 file .nii and stack
    flair_file_path = 'C:\\Users\\thanhdh6\\Documents\\project\\brain_tumor_segmentation\\samples\\BraTS20_Training_001\\BraTS20_Training_001_flair.nii'
    t1_file_path = 'C:\\Users\\thanhdh6\\Documents\\project\\brain_tumor_segmentation\\samples\\BraTS20_Training_001\\BraTS20_Training_001_t1.nii'
    t2_file_path = 'C:\\Users\\thanhdh6\\Documents\\project\\brain_tumor_segmentation\\samples\\BraTS20_Training_001\\BraTS20_Training_001_t2.nii'
    t1ce_file_path = 'C:\\Users\\thanhdh6\\Documents\\project\\brain_tumor_segmentation\\samples\\BraTS20_Training_001\\BraTS20_Training_001_t1ce.nii'
    # load imgs and normalize
    flair_data = nib.load(flair_file_path)
    flair_img = np.asarray(flair_data.dataobj)
    # print(flair_img.shape)
    flair_img = normalize_image(flair_img)

    t1_data = nib.load(t1_file_path)
    t1_img = np.asarray(t1_data.dataobj)
    t1_img = normalize_image(t1_img)

    t2_data = nib.load(t2_file_path)
    t2_img = np.asarray(t2_data.dataobj)
    t2_img = normalize_image(t2_img)

    t1ce_data = nib.load(t1ce_file_path)
    t1ce_img = np.asarray(t1ce_data.dataobj)
    t1ce_img = normalize_image(t1ce_img)

    # stack and transpose
    images = []
    images.append(flair_img)
    images.append(t1_img)
    images.append(t2_img)
    images.append(t1ce_img)
    imgs = np.stack(images)
    imgs = np.moveaxis(imgs, (0, 1, 2, 3), (0, 3, 2, 1))

    # predict
    imgs_tensor=torch.from_numpy(imgs).float().to(device)
    logits = unet3d(imgs_tensor.unsqueeze(0))
    print(logits.size())
