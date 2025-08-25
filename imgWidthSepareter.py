import os
import glob
from PIL import Image
from collections import defaultdict
# Directory containing images
myvars = {}
dictionary = defaultdict()
dictonaryLabels = defaultdict()
def iterate_over_dict():
    for key in dictionary:
        print(key, dictionary[key])

    
with open("ref.cnf") as myfile:
    for line in myfile:
        name, var = line[:-1].strip().partition("=")[::2]
        myvars[name.strip()] = var.strip()
dir_path = myvars['imgFolder']+'/*/*.png'
dict_imgSize_toList = {}

label_file="data/English_data/IAM64_train.txt"

#mapping labels
with open(label_file, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            folder_image_name, text = line.strip().split(" ",1)
            folder, image_name =  folder_image_name.strip().split(",")
            dictonaryLabels.setdefault(folder+"/"+image_name, []).append(text)

            #self.samples.append((folder+"/"+image_name+".png", text))
        except:
            print(line)



from pathlib import Path

# List all files in the directory
listing = glob.glob(dir_path)
for file_name in listing:

    # Construct full file path
    # Open and find the image size
    with Image.open(file_name) as img:
        path = Path(file_name)
        os.path.split(path.parent.absolute())[1]
        label = os.path.split(path.parent.absolute())[1]+"/"+ os.path.basename(path)
        label = label.replace(".png","")
        print(label)

        dictionary.setdefault(img.size, []).append(file_name+"|"+"".join((dictonaryLabels[label])))

        print(type(img.size))
        print(f"{file_name}: {img.size}")



#print(dictionary[(1661, 89)][1])
#iterate_over_dict()
print(len(dictionary))