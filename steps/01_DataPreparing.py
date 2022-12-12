from ctypes import resize
from glob import glob
import json
import os
from datetime import datetime
import math
import random
import shutil

from utils import connectWithAzure

import cv2
from dotenv import load_dotenv
from azureml.core import Dataset
from azureml.data.datapath import DataPath


# When you work locally, you can use a .env file to store all your environment variables.
# This line read those in.
load_dotenv()

FOOD = os.environ.get('FOOD').split(',')
SEED = int(os.environ.get('RANDOM_SEED'))
TRAIN_TEST_SPLIT_FACTOR = float(os.environ.get('TRAIN_TEST_SPLIT_FACTOR'))


def processAndUploadFoodImages(datasets, data_path, processed_path, ws, food_name):

    # We can't use mount on these machines, so we'll have to download them

    food_path = os.path.join(data_path, 'food', food_name)

    # Get the dataset name for this animal, then download to the directory
    # Overwriting means we don't have to delete if they already exist, in case something goes wrong.
    datasets[food_name].download(food_path, overwrite=True)
    print('Downloading all the images')

    # Get all the image paths with the `glob()` method.
    print(f'Resizing all images for {food_name} ...')
    # CHANGE THIS LINE IF YOU NEED TO GET YOUR ANIMAL_NAMES IN THERE IF NEEDED!
    image_paths = glob(f"{food_path}/*.jpg")

    # Process all the images with OpenCV. Reading them, then resizing them to 64x64 and saving them once more.
    print(f"Processing {len(image_paths)} images")
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (64, 64))  # Resize to a square of 64, 64
        cv2.imwrite(os.path.join(processed_path, food_name,
                    image_path.split('/')[-1]), image)
    print(f'... done resizing. Stopping context now...')

    # Upload the directory as a new dataset
    print(f'Uploading directory now ...')
    resized_dataset = Dataset.File.upload_directory(
        # Enter the sourece directory on our machine where the resized pictures are
        src_dir=os.path.join(processed_path, food_name),
        # Create a DataPath reference where to store our images to. We'll use the default datastore for our workspace.
        target=DataPath(datastore=ws.get_default_datastore(),
                        path_on_datastore=f'processed_food/{food_name}'),
        overwrite=True)

    print('... uploaded images, now creating a dataset ...')

    # Make sure to register the dataset whenever everything is uploaded.
    new_dataset = resized_dataset.register(ws,
                                           name=f'resized_{food_name}',
                                           description=f'{food_name} images resized tot 64, 64',
                                           # Optional tags, can always be interesting to keep track of these!
                                           tags={
                                               'food': food_name, 'AI-Model': 'CNN', 'GIT-SHA': os.environ.get('GIT_SHA')},
                                           create_new_version=True)
    print(
        f" ... Dataset id {new_dataset.id} | Dataset version {new_dataset.version}")
    print(f'... Done. Now freeing the space by deleting all the images, both original and processed.')
    emptyDirectory(food_path)
    print(f'... done with the original images ...')
    emptyDirectory(os.path.join(processed_path, food_name))
    print(f'... done with the processed images. On to the next Animal, if there are still!')


def emptyDirectory(directory_path):
    shutil.rmtree(directory_path)


def prepareDataset(ws):
    data_folder = os.path.join(os.getcwd(), 'data')
    os.makedirs(data_folder, exist_ok=True)

    for food_name in FOOD:
        os.makedirs(os.path.join(data_folder, 'food',
                    food_name), exist_ok=True)

    # Define a path to store the food images onto. We'll choose for `data/processed/food` this time. Again, create subdirectories for all the food
    processed_path = os.path.join(os.getcwd(), 'data', 'processed', 'food')
    os.makedirs(processed_path, exist_ok=True)
    for food_name in FOOD:
        os.makedirs(os.path.join(processed_path, food_name), exist_ok=True)

    # Make sure to give our workspace with it
    datasets = Dataset.get_all(workspace=ws)
    for food_name in FOOD:
        processAndUploadFoodImages(
            datasets, data_folder, processed_path, ws, food_name)


def trainTestSplitData(ws):

    training_datapaths = []
    testing_datapaths = []
    default_datastore = ws.get_default_datastore()
    for food_name in FOOD:
        # Get the dataset by name
        food_dataset = Dataset.get_by_name(ws, f"resized_{food_name}")
        print(f'Starting to process {food_name} images.')

        # Get only the .JPG images
        food_images = [
            img for img in food_dataset.to_path() if img.split('.')[-1] == 'jpg']

        print(f'... there are about {len(food_images)} images to process.')

        # Concatenate the names for the food_name and the img_path. Don't put a / between, because the img_path already contains that
        # Make sure the paths are actual DataPaths
        food_images = [
            (default_datastore, f'processed_foods/{food_name}{img_path}') for img_path in food_images]

        # Use the same random seed as I use and defined in the earlier cells
        random.seed(SEED)
        random.shuffle(food_images)  # Shuffle the data so it's randomized

        # Testing images
        # Get a small percentage of testing images
        amount_of_test_images = math.ceil(
            len(food_images) * TRAIN_TEST_SPLIT_FACTOR)

        food_test_images = food_images[:amount_of_test_images]
        food_training_images = food_images[amount_of_test_images:]

        # Add them all to the other ones
        testing_datapaths.extend(food_test_images)
        training_datapaths.extend(food_training_images)

        print(
            f'We already have {len(testing_datapaths)} testing images and {len(training_datapaths)} training images, on to process more animals if necessary!')

    training_dataset = Dataset.File.from_files(path=training_datapaths)
    testing_dataset = Dataset.File.from_files(path=testing_datapaths)

    training_dataset = training_dataset.register(ws,
                                                 # Get from the environment
                                                 name=os.environ.get(
                                                     'TRAIN_SET_NAME'),
                                                 description=f'The Food Images to train, resized tot 64, 64',
                                                 tags={'food': os.environ.get('FOOD'), 'AI-Model': 'CNN', 'Split size': str(
                                                     1 - TRAIN_TEST_SPLIT_FACTOR), 'type': 'training', 'GIT-SHA': os.environ.get('GIT_SHA')},
                                                 create_new_version=True)

    print(
        f"Training dataset registered: {training_dataset.id} -- {training_dataset.version}")

    testing_dataset = testing_dataset.register(ws,
                                               # Get from the environment
                                               name=os.environ.get(
                                                   'TEST_SET_NAME'),
                                               description=f'The Food Images to test, resized tot 64, 64',
                                               tags={'food': os.environ.get('FOOD'), 'AI-Model': 'CNN', 'Split size': str(
                                                   TRAIN_TEST_SPLIT_FACTOR), 'type': 'testing', 'GIT-SHA': os.environ.get('GIT_SHA')},
                                               create_new_version=True)

    print(
        f"Testing dataset registered: {testing_dataset.id} -- {testing_dataset.version}")


def main():
    ws = connectWithAzure()

    print('Processing the images')
    prepareDataset(ws)

    print('Splitting the images')
    trainTestSplitData(ws)


if __name__ == '__main__':
    main()
