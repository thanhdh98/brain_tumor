def get_augmentations(phase):
    list_transforms = []
    
    list_trfms = Compose(list_transforms)
    return list_trfms


def get_dataloader(
    dataset: torch.utils.data.Dataset,
    path_to_csv: str,
    phase: str,
    fold: int = 0,
    batch_size: int = 1,
    num_workers: int = 4,
):
    '''Returns: dataloader for the model training'''
    df = pd.read_csv(path_to_csv)
    # print(df.describe)
    # split data on train, test, split
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=69, shuffle=True)
    train_df, val_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True)
    print("train_df:", train_df.shape, "val_df:", val_df.shape)
    df = train_df if phase == "train" else val_df
    dataset = dataset(df, phase)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,   
    )
    return dataloader