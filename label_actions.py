from utils import get_videos
import cv2
import numpy as np
import os
import pickle as pkl

vid_ids = ['1644964597']
for vid_id in vid_ids:
    videos = get_videos(vid_id)
    print(videos)
    vidcap = cv2.VideoCapture(videos[0])
    success,image = vidcap.read()
    imgs = []
    labeled_actions = []
    while success:
        imgs.append(image)
        success, image = vidcap.read()
    action_started = False
    window_name = 'current image'
    objs = ['teabag', 'cup', 'pitcher', 'tap', np.nan]
    i = 0
    while i < len(imgs):
        cv2.imshow(window_name, imgs[i])
        key = cv2.waitKey(100)
        if key == ord('s'):
            if action_started:
                temp = input('Did not finish labeling previous action, Going back to the start of previous action. Press any key to continue.')
                i = act['start']
                continue
            else:
                action_started = True
                act = {}
                act['start'] = i
        elif key == ord('e'):
            if not action_started:
                temp = input('No action has been started, ready to continue? Press any key to continue')
                continue
            try:
                act['end'] = i
                moving_obj = input('Input the index of the MOVING object. 0: teabag, 1: cup, 2:pitcher, 3:tap')
                act['moving_obj'] = objs[int(moving_obj)]
                label1 = input('Is it a PICK-AND-PLACE? 1 for yes, and 0 for no.')
                act['label1'] = label1
                label2 = input('Input the index of the TARGET object. 0: teabag, 1: cup, 2:pitcher, 3:tap, 4: None')
                act['label2'] = objs[int(label2)]
            except:
                input('Input error, press any key to start from the begining of the action: ')
                i = act['start']
                continue
            labeled_actions.append(act)
            action_started = False
        elif key == ord('m'):
            input('Press any key to go back to the end of previous action: ')
            i = act['end']
        elif key == ord('n'):
            input('Press any key to go back to the start of previous action: ')
            i = act['start']
        elif key == ord('r'):
            temp = input('Do you want to relabel the video? y or n')
            if temp == 'y':
                i = 0
        elif key == ord('q'):
            break
        i += 1
    cv2.destroyAllWindows()
    output_dir = os.path.dirname(os.path.dirname(videos[0]))
    outputname = output_dir + '/' + 'labeld_actions.pkl'

    with open(outputname, 'wb') as f:
        print(f'Saving to {outputname}')
        pkl.dump(labeled_actions, f)
