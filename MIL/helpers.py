import numpy as np
import SimpleITK as sitk
import os
import cv2
import time
def show(img,speed=1):
    img = img[0,0]
    img = img.permute(2,0,1)
    #img = img.permute(4, 2, 3, 1, 0) #monai
    img=img.numpy(force=True)
    # Define video writer
    img = process_image_clip(img)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 10.0, (img.shape[2], img.shape[1]))

    # Loop through image slices and add them to the video
    for i in range(0, img.shape[0],speed):
        frame = img[i, :, :]
        frame = (frame * 255).astype(np.uint8)  # Convert to 8-bit format
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel format

        # Draw frame number
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottom_right_corner = (img.shape[2] - 100, img.shape[1] - 20)
        font_scale = 0.5
        font_color = (255, 255, 255)  # White color
        thickness = 1
        cv2.putText(frame, f'Frame: {i}', bottom_right_corner, font, font_scale, font_color, thickness)

        out.write(frame)

    # Release the video writer
    out.release()
    slow_mode = False
    slow_t = 300
    cap = cv2.VideoCapture('output.avi')

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to the beginning if it reaches the end
            ret, frame = cap.read()
            if slow_mode:
                slow_t+=100

        cv2.imshow('Video', frame)
        if slow_mode:
            t=slow_t
        else:
            t = 10
            slow_t=140
        key = cv2.waitKey(t) & 0xFF
        if key == 27:  # Press the 'Esc' key to exit the video window
            break
        elif key == 115: #S
            slow_mode = not slow_mode
            time.sleep(0.2)
    cap.release()
    cv2.destroyAllWindows()



def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
def process_image_clip(image):
    
    image=np.clip(image, -200, 300)
    image=image.astype('float32')
    image=(image-np.min(image))/(np.max(image)-np.min(image))
    return image

def process_image(image):
#     image=np.clip(image, -200, 300)
    image=image.astype('float32')
    image=(image-np.min(image))/(np.max(image)-np.min(image))
    return image

def class_to_index(cls):
    classes = ['B', 'M']
    if isinstance(cls,list):
        indices = []
        for _class in cls:
            if _class in classes:
                indices.append(classes.index(_class))
            else:
                print("Invalid class: " + str(_class))
                raise ValueError("Invalid class: " + str(_class))
        return indices




    if cls in classes:
        return classes.index(cls)
    else:
        print("Invalid class: "+str(cls))
        raise ValueError("Invalid class: "+str(cls))
    return -1

def check_files_in_folder(file_list, folder_path):
    # Get the list of files in the folder
    folder_contents = os.listdir(folder_path)

    # Check if each file in the file list is present in the folder
    missing_files = []
    for file_name in file_list:
        if file_name not in folder_contents:
            missing_files.append(file_name)

    return missing_files








def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
