import os
import glob
from PIL import Image
import torch
import pandas as pd
import csv
import sys
import torch
import torchvision.transforms as transforms
from models import find_model_using_name


def load_resized_img(path):
    return Image.open(path).convert('RGB').resize((256, 256))

def process_cityscapes(gtFine_dir, output_dir):
    phase = 'val'
    savedir = os.path.join(output_dir, phase)
    # os.makedirs(savedir, exist_ok=True)
    # os.makedirs(savedir + 'A', exist_ok=True)
    os.makedirs(savedir + 'B', exist_ok=True)
    print("Directory structure prepared at %s" % output_dir)
    
    segmap_expr = os.path.join(gtFine_dir + "/*/*_color.png")
    segmap_paths = glob.glob(segmap_expr)
    segmap_paths = sorted(segmap_paths)

    temp_file_names = []
    original_names = []

    for i , segmap_path in enumerate(segmap_paths):
        segmap = load_resized_img(segmap_path)

        # Store images in a directory
        temp_name =  "%d_B.jpg" % i
        savepath = os.path.join(savedir + 'B', temp_name)
        segmap.save(savepath, format='JPEG', subsampling=0, quality=100)

        temp_file_names.append(temp_name)
        original_names.append(segmap_path)
        
        if i % (len(segmap_paths) // 10) == 0:
            print("%d / %d: last image saved at %s, " % (i, len(segmap_paths), savepath))

    dic = {
        "Temp name" : temp_file_names, 
        "orignal names" : original_names
    }

    df = pd.DataFrame(dic)
    df.to_csv('name mapping.csv', index=False)

def load_and_preprocess_image(file_path):
    # Load the image using PIL
    image = Image.open(file_path).convert('RGB')

    # Define the transformation pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Apply the transformations to the image
    input_tensor = transform(image)

    # Add an extra dimension to represent batch size
    input_tensor = input_tensor.unsqueeze(0)

    return input_tensor

def load_name_mapping(file_path):

    # Initialize an empty dictionary
    data_dict = {}

    # Read the CSV file and populate the dictionary
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            key = row[0]  # First column (Temp name)
            value = row[1]  # Second column (original names)
            data_dict[key] = value
    
    return data_dict

def main():
    
    print("Image loading")
    # do some image loading and tensor
    IMAGE_PATH = "/Users/taichi/Documents/Github/pytorch-CycleGAN-and-pix2pix/val/"
    OUTPUT_PATH = "/Users/taichi/Documents/Github/pytorch-CycleGAN-and-pix2pix/val2/"
    PREDICTION_PATH = "/Users/taichi/Documents/Github/pytorch-CycleGAN-and-pix2pix/prediction/"

    MODEL_PATH = "/Users/taichi/Documents/Github/pytorch-CycleGAN-and-pix2pix/latest_model/"
    MAPPING_PATH = "/Users/taichi/Documents/Github/pytorch-CycleGAN-and-pix2pix/scripts/eval_cityscapes/name mapping.csv"
    # images = process_cityscapes(IMAGE_PATH, OUTPUT_PATH)

    model = find_model_using_name("cycle_gan")
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, "latest_net_G_B.pth")))
    model.eval()

    val2_path = os.path.join(OUTPUT_PATH + "/*/*.jpg")
    val2_path = glob.glob(val2_path)
    val2_path = sorted(val2_path)

    name_mapping = load_name_mapping(MAPPING_PATH)

    for p in val2_path:
        image = load_and_preprocess_image(p)
        output = model(image)
        output_image = Image.fromarray(output.numpy())

        image_name = os.path.basename(p)

        orignal_name = name_mapping[image_name]
        path_area = os.path.sep.join(orignal_name.split(os.path.sep)[-1:])

        output_image.save(os.path.join(PREDICTION_PATH, path_area))
        print(f"Saved {path_area}")

    


    # print("Image loading complete")

    # print("Model loading")
    # model = torch.load(os.path.join(MODEL_PATH, "latest_net_G_B.pth"))
    # print(type(model))
    # model.eval()
    # print("Model loading complete")

    # for img in images:
    #     output = model(img.image)
    #     save_dir = PREDICTION_PATH + img.orignal_name
    #     output.save(save_dir, format='JPEG', subsampling=0, quality=100)

main()
