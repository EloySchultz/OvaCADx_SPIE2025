import os
import pandas as pd
import subprocess
import time
import SimpleITK as sitk
import cv2
import numpy as np
from Helpers import process_image_clip, process_image
DIR = "/home/eloy/Documents/Data/OVARY_DATA/Data_gyn_3.4"
SAMPLES_NAME = "dataset.csv"#"Samples.csv"
csv_file = os.path.join(DIR,SAMPLES_NAME)
df = pd.read_csv(csv_file)



#Tool for manually checking samples and annotations. Will convert each scan and annotations to video and display this full screen, after which the user can give a label regarding presence of artifacts.
#Controls:
#space = Toggle pause
#S = Toggle slow motion mode
#B = go to previous sample without labling
#R = reload current sample
#G = Give label "Good/OK, no artifacts" (quality = 5)
#A = Give label "Artifacts". You will be prompted to give a description and a quality/usability grade between 1 and 5.
#I = Open in ITKSNAP without altering label
#N = Go to next image without altering label
#ESC = Exit (progress so far is always saved in CSV file).


import os
comments=[]

indices_mode=True


df = df[df['Inclusion']==1]
indices = df.index #df[df['Usability'].isin([4])].index
# i=273-2

#i=294-2
i=0
while (i < len(df)): # Need while loop for moving backwards!
    if indices_mode:
        ind = indices[i]
    else:
        ind = i
    IP = os.path.join(DIR,df['Image path'][ind])

    AP = os.path.join(DIR,df['Annotation path'][ind])
    print(ind," AP:  ",AP)

    # Create an output video writer

    image = sitk.ReadImage(IP)
    image = process_image_clip(sitk.GetArrayFromImage(image))
    ann = sitk.ReadImage(AP)
    ann = process_image(sitk.GetArrayFromImage(ann))


    def superimpose_annotation(image, annotation):
        # Convert image and annotation to 3-channel format
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        annotation_bgr = cv2.cvtColor(annotation, cv2.COLOR_GRAY2BGR)
        annotation_bgr[:, :, :2] = 0 #zero all channels except red


        # Blend the annotation with the image
        #blended = cv2.addWeighted(image_bgr, 1.0, red_mask, 0.5, 0)
        #blended = cv2.addWeighted(image_bgr, 1.0, annotation_bgr, 0.5, 0)
        blended = cv2.max(image_bgr, annotation_bgr)
        # blended = np.clip(blended, 0, 255).astype(np.uint8)
        return blended


    # Initialize the video writer
    # fourcc = cv2.VideoWriter_fourcc(*'X264')
    fourcc = cv2.VideoWriter_fourcc(*'RGBA')
    out = cv2.VideoWriter('output.avi', fourcc, 10.0, (2 * image.shape[2], image.shape[1]))

    # Loop through image slices and add them to the video
    for k in range(image.shape[0]):
        if np.any(ann[k,:,:]):
            frame_left = image[k, :, :]
            frame_right = superimpose_annotation(image[k, :, :], ann[k, :, :])

            frame_left = (frame_left * 255).astype(np.uint8)
            frame_right = (frame_right * 255).astype(np.uint8)

            frame_left = cv2.cvtColor(frame_left, cv2.COLOR_GRAY2BGR)
            #frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2BGR)

            # Add the slice number in green text to the frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f'Img: {ind}, Slice: {k}'
            org = (10, frame_left.shape[0] - 10)  # Bottom-left corner
            font_scale = 0.5
            font_color = (0, 255, 0)  # Green color
            font_thickness = 1
            cv2.putText(frame_left, text, org, font, font_scale, font_color, font_thickness)


            frame = np.hstack((frame_left, frame_right))
            out.write(frame)

    # Release the video writer
    out.release()

    # Play the created video
    cap = cv2.VideoCapture('output.avi')
    cv2.namedWindow('Video', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    slow_mode = False
    slow_t = 300
    record_comment=False
    openITK=False
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to the beginning if it reaches the end
            ret, frame = cap.read()
            if slow_mode:
                slow_t+=100
        cv2.imshow('Video', frame)
        cv2.setWindowTitle('Video', df['Tumor ID'][ind])


        if slow_mode:
            t=slow_t
        else:
            t = 70
            slow_t=140
        key = cv2.waitKey(t) & 0xFF
        if key == 27:  # Press the 'Esc' key to exit the video window
            i=99999; #exit!
            break
        if key ==32:
            pause=True
            time.sleep(0.4)
            while(pause):
                key = cv2.waitKey(200000) & 0xFF#Wait for very long time, or user presses space
                if key == 32:
                    pause=False
                    time.sleep(0.4)
        elif key == 103: #ord('G'):  # Press 'G'
            df.at[ind,'Comments'] = "OK"
            df.at[ind,'Usability'] = 5
            break
        elif key == 97: #ord('A'):  # Press 'A'
            #df.at[ind, 'Comments'] = "ARTIFACTS!"
            record_comment=True;
            break
        elif key == 105: #ord('I'):  # Press 'I'
            openITK=True
            break
        # elif key == 110: #ord('N'):  # Press 'N'
        #     df.at[i, 'Comments'] = "NOT SURE, CHECK AGAIN IN ITKSNAP!"
        #     break
        elif key == 104: #ord('H'):  # Press 'H'
            df.at[ind, 'Comments'] = "QUALITY IMPEDED BY IMPLANT"
            df.at[ind, 'Usability'] = 4
            break
        elif key == 98: #ord('B'):  # Press 'B'
            i=i-2 #Go back
            break
        elif key == 110: #ord('N'):  # Press 'N'
            break #next
        elif key == 114: #ord('R'):  # Press 'R'
            i=i-1 #Reload current
            break
        elif key == 115:
            slow_mode=not slow_mode
            time.sleep(0.2)
    cap.release()
    cv2.destroyAllWindows()
    if record_comment==True:
        print("Current comment:" + str(df.at[ind, 'Comments']) + " | Current quality:" + str(df.at[ind, 'Usability']))
        df.at[ind, 'Comments'] = input("Please write any notes here:")
        df.at[ind, 'Usability'] = input("Please give a usability grade between 1 (worst) and 5 (best)")
    elif openITK==True:
        process = subprocess.Popen(
            ['/usr/local/itksnap/bin/itksnap', '-g', IP, '-s', AP])
        print("Please put down any comments on sample: "+df['ID'][ind]+" and press enter.")
        x = input("Please press enter to continue")
        process.kill()
    #print("Saving results to: " + os.path.join(DIR, "Manual_check_samples.csv"))
    df.to_csv(os.path.join(DIR, "Manual_check_samples.csv"), index=False)
    #

    print("DONE")
    i=i+1

print("Saving results to: "+os.path.join(DIR,"Manual_check_samples.csv"))
df.to_csv(os.path.join(DIR,"Manual_check_samples.csv"), index=False)
#
