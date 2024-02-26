import csv
import os
import shutil

# Define the base path where the folders containing audio files are located
base_path = 'audio/'

# Define the destination path where you want to move the files
train_destination_path = 'dataset/train/'
test_destination_path = 'dataset/test/'

# Ensure the destination directory exists
if not os.path.exists(train_destination_path):
    os.makedirs(train_destination_path)
if not os.path.exists(test_destination_path):
    os.makedirs(test_destination_path)

# Open and read the CSV file
with open('partitions/split01_train.csv') as csvfile:
    filereader = csv.reader(csvfile)
    for row in filereader:
        # Extract the file name from the row (assuming there's no header and each row contains just the file name)
        file_name = row[0]

        # Construct the folder name from the first 3 characters of the file name
        folder_name = file_name[:3]

        # Construct the source path
        source_path = os.path.join(base_path, folder_name, file_name + '.ogg')

        # Construct the destination path
        dest_file_path = os.path.join(train_destination_path, file_name + '.ogg')

        # Move the file
        shutil.move(source_path, dest_file_path)

        print(f"Moved {source_path} to {dest_file_path}")
        
# Open and read the CSV file
with open('partitions/split01_test.csv') as csvfile:
    filereader = csv.reader(csvfile)
    for row in filereader:
        # Extract the file name from the row (assuming there's no header and each row contains just the file name)
        file_name = row[0]

        # Construct the folder name from the first 3 characters of the file name
        folder_name = file_name[:3]

        # Construct the source path
        source_path = os.path.join(base_path, folder_name, file_name + '.ogg')

        # Construct the destination path
        dest_file_path = os.path.join(test_destination_path, file_name + '.ogg')

        # Move the file
        shutil.move(source_path, dest_file_path)

        print(f"Moved {source_path} to {dest_file_path}")
