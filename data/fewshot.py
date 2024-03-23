from torchvision.datasets.folder import ImageFolder, default_loader
import os

class general_dataset_few_shot(ImageFolder):
    def __init__(self, root, train=True, transform=None, target_transform=None, mode=None,is_individual_prompt=False,shot=2,seed=0,**kwargs):
        self.dataset_root = root
        self.loader = default_loader
        self.target_transform = None
        self.transform = transform
        
        train_list_path = os.path.join(self.dataset_root, 'annotations/train_meta.list.num_shot_'+str(shot)+'.seed_'+str(seed))

        test_list_path = os.path.join(self.dataset_root, 'annotations/test_meta.list')


        self.samples = []
        if train:
            with open(train_list_path, 'r') as f:
                for line in f:
                    img_name = line.rsplit(' ',1)[0]
                    label = int(line.rsplit(' ',1)[1])
                    if 'stanford_cars' in root:
                        self.samples.append((os.path.join(root,img_name), label))
                    else:
                        self.samples.append((os.path.join(root+'/images',img_name), label))
                    
        else:
            with open(test_list_path, 'r') as f:
                for line in f:
                    img_name = line.rsplit(' ',1)[0]
                    label = int(line.rsplit(' ',1)[1])
                    if 'stanford_cars' in root:
                        self.samples.append((os.path.join(root,img_name), label))
                    else:
                        self.samples.append((os.path.join(root+'/images',img_name), label))
