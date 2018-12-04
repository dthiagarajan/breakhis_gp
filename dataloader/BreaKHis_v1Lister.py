import torch.utils.data as data
from PIL import Image
import os
import os.path


base_dir = '/scratch/datasets/BreaKHis_v1/histology_slides/breast/'
classes = {
    'benign': ['SOB/adenosis/', 'SOB/fibroadenoma/', 'SOB/phyllodes_tumor/', 'SOB/tubular_adenoma/'],
    'malignant': ['SOB/ductal_carcinoma/', 'SOB/lobular_carcinoma/', 'SOB/mucinous_carcinoma/', 'SOB/papillary_carcinoma/']
}
# Need to listdir from here
magnifications = {'40X', '100X', '200X', '400X'}


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):
    images = [[], []]
    params = [[], []]
    labels = [[], []]

    for i, cls in enumerate(classes):
        for hist in classes[cls]:
            curr_dir = filepath + cls + '/' + hist
            for sub_dir in os.listdir(curr_dir):
                slide_num = sub_dir.split('_')[-1]
                for magnification in magnifications:
                    magnification_dir = curr_dir + sub_dir + '/' + magnification + '/'
                    for image in os.listdir(magnification_dir):
                        attr = {}
                        attr['slide_num'] = slide_num
                        attr['magnification'] = magnification
                        attr['image_num'] = image.split(".")[0][-3:]
                        image_path = magnification_dir + image
                        images[i].append(image_path)
                        params[i].append(attr)
                        labels[i].append(cls)
    return images, params, labels
