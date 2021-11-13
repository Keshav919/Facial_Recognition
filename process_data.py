import os
import json
import random
import shutil
import numpy as np
random.seed(42)

if __name__ == '__main__':

    # Setting the paths for datasets
    data_path = "./lfw/lfw-deepfunneled/lfw-deepfunneled/"
    dataset_path = "./data/"

    # Getting all the faces in LFW
    lfw_dir = os.listdir(data_path)

    #Initialising dataset details
    faces = []
    num_faces = []
    images = []
    labels = []
    count = 0
    name2id = {}

    # Go through each face and find the number of images 
    for name in lfw_dir:
        face_dir = os.listdir(data_path + name + "/")
        num_images = len(face_dir)
        num_faces.append(num_images)
        faces.append(name)

    # To simplify dataset we only want the best 10 faces
    num_faces = np.array(num_faces)
    faces = np.array(faces)
    sort_idx = np.argsort(num_faces)[::-1]
    faces = faces[sort_idx][:10]
    nums = num_faces[sort_idx][:10]

    
    # Create dataset by associating image id to face id
    for id_, face in enumerate(faces):
            face_dir = os.listdir(data_path + face + "/")
            # Associate name to label
            name2id[id_] = face
            print(face)
            # Save name to label
            for image in face_dir:
                shutil.copy(data_path + face + "/"+image,dataset_path+'{0:06d}'.format(count)+".jpg")
                labels.append(id_)
                images.append('{0:06d}'.format(count))
                count += 1
            id_ += 1


    # Size of dataset
    print("There are ", count, "faces in the dataset!")
    train_test_split = 0.8

    # Generate random dataset
    sample_list = list(zip(images,labels))
    random.shuffle(sample_list)

    im_lab = list(zip(*sample_list))
    images = im_lab[0]
    labels = im_lab[1]

    # Split into train and test
    train_im = images[:int(train_test_split*len(images))]
    test_im = images[int(train_test_split*len(images)):]

    train_lab = labels[:int(train_test_split*len(labels))]
    test_lab = labels[int(train_test_split*len(labels)):]
    
    # Write dataset
    unique, counts = np.unique(train_lab, return_counts=True)
    np.save(dataset_path+"num_faces.npy", counts)
    
    with open(dataset_path+"train_im.txt", 'w') as f:
        f.write('\n'.join(train_im))
        f.close()
    with open(dataset_path+"test_im.txt", 'w') as f:
        f.write('\n'.join(test_im))
        f.close()
    with open(dataset_path+"train_lab.txt", 'w') as f:
        f.write('\n'.join([str(lab) for lab in train_lab]))
        f.close()
    with open(dataset_path+"test_lab.txt", 'w') as f:
        f.write('\n'.join([str(lab) for lab in test_lab]))
        f.close()
    
    json_obj = json.dumps(name2id)
    with open(dataset_path+"name2id.json", 'w') as f:
        f.write(json_obj)
        f.close()
