training_name_mapping_df = pd.read_csv('../input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/name_mapping.csv')
# validating_name_mapping_df = pd.read_csv('../input/brats20-dataset-training-validation/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/name_mapping_validation_data.csv')


# print(df.describe(include='all'))
paths = []
ids=[]
# Paths for training datasets
for _, row  in training_name_mapping_df.iterrows():
    id_=row['BraTS_2020_subject_ID']
    ids.append(id_)
    path=os.path.join(config.train_root_dir,id_)
    paths.append(path)
# # Paths for validating datasets
# for _, row  in validating_name_mapping_df.iterrows():
#     id_=row['BraTS_2020_subject_ID']
#     ids.append(id_)
#     path=os.path.join(config.test_root_dir,id_)
#     paths.append(path)
# Create dataframe 
data={'Brats20ID':ids,
      'path':paths}
df=pd.DataFrame(data)
# discard sampple 355
df = df.loc[df['Brats20ID'] != 'BraTS20_Training_355'].reset_index(drop=True, )
# print(df.head())

# save to csv file
df.to_csv('train_data.csv', index=False)