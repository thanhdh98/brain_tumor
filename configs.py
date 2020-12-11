class GlobalConfig:
    root_dir = '../input/brats20-dataset-training-validation'
    train_root_dir = '../input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    test_root_dir = '../input/brats20-dataset-training-validation/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'
    #####
    path_to_csv = '../input/unet3dlogs/train_data.csv'
    pretrained_model_path = '../input/unet/last_epoch_model.pth'
    train_logs_path = './train_log.csv'
    saved_model_folder = './'
    ae_pretrained_model_path = '../input/ae/autoencoder_best_model.pth'
    tab_data = '../input/brats20logs/brats2020logs/data/df_with_voxel_stats_and_latent_features.csv'
    seed = 55


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


config = GlobalConfig()
seed_everything(config.seed)
