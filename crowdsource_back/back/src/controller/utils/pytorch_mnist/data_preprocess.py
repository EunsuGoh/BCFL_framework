import h5py
import torchvision
import torchvision.transforms as transforms

save_root_path = "crowdsource_back/back/src/utils/pytorch_mnist/data/MNIST/MNIST"
model_name = "mnist-client-"
option="data"

trainset = torchvision.datasets.MNIST(
    root = './data/MNIST',
    train = True,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor() # 데이터를 0에서 255까지 있는 값을 0에서 1사이 값으로 변환
    ])
)
print(len(trainset))

interval = len(trainset)//5

for i in range(5):
  train_file = h5py.File(save_root_path+model_name+str(i+1)+"_"+option+"_train.hdf5",'w')
  dset = train_file.create_group(model_name+str(i+1))
  dset['data'] = trainset.data[interval*i:interval*(i+1)]
  dset['targets'] = trainset.targets[interval*i:interval*(i+1)]
  # dset['preprocessed_data'] = return_dict[model_name+str(i+1)]['preprocessed_data']
  print(dset)
  train_file.close()

