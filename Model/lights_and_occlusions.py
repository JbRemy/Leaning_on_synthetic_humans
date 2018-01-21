import numpy as np
import pandas as pd
from PIL import Image
import random as rdm

def Avg_Brightness(rgb_target, ix):
    rgb_target = pd.DataFrame(rgb_target[ix])
    avg_brightness = rgb_target.mean().sum() / 3
    return np.floor(avg_brightness)

def Center_Brightness(rgb_img, ix, target_brightness):
    current_brightness = Avg_Brightness(rgb_img, ix)
    for t in range(len(ix[1])):
        i = ix[0][t]
        j = ix[1][t]
        tmp = rgb_img[i,j,] - current_brightness + target_brightness
        tmp[tmp > 255] = 255
        tmp[tmp < 0] = 0
        rgb_img[i,j,] = tmp
    return None

def Pre_Process_Images(img, segm, proba_crop=0.8, prop_min=0.2, prop_max=0.5):
#for i in tqdm(range(len(names_segm))):
    #img = Image.open(path_images + names_images[i])
    #segm = segm_dict[names_segm[i]]
    rgb_img = np.array(img)

    #### Center brightness on background
    ix_bg = np.where(segm == 0)
    ix_hum = np.where(segm != 0)
    bg_brightness = Avg_Brightness(rgb_img, ix_bg)
    Center_Brightness(rgb_img, ix_hum, bg_brightness)


    #### Crop the human with proba_crop
    if True :
    #if (np.random.binomial(1,proba_crop) == 1) & (np.array(ix_hum).shape[1]>0):

        # Get points of reference #####################
        # x and y axis are inversed in the arrays and in directions
        frame_lim_y = len(segm)-1
        frame_lim_x = len(segm[0])-1
        ix_hum = np.where(segm != 0)
        x_hum = ix_hum[1]
        y_hum = ix_hum[0]
        top_hum = min(y_hum)
        bottom_hum = max(y_hum)
        left_hum = min(x_hum)
        right_hum = max(x_hum)
        len_x_hum = len(np.unique(x_hum))
        len_y_hum = len(np.unique(y_hum))

        # Get cropping surface #####################
        # Proportion of cropping
        prop_range = prop_max - prop_min
        prop_x = rdm.random() * prop_range + (prop_max - prop_range)
        prop_y = rdm.random() * prop_range + (prop_max - prop_range)
        # Location of cropping
        ix_loc = rdm.sample(range(len(ix_hum[0])),1)
        x_ref_crop = ix_hum[1][ix_loc][0]
        y_ref_crop = ix_hum[0][ix_loc][0]
        dir_x_crop = rdm.sample([-1,1],1)[0]
        dir_y_crop = rdm.sample([-1,1],1)[0]
        if dir_x_crop == 1:
            len_x_crop = min(frame_lim_x - x_ref_crop, int(np.floor(len_x_hum * prop_x)))
        else:
            len_x_crop = min(x_ref_crop, int(np.floor(len_x_hum * prop_x)))
        if dir_y_crop == 1:
            len_y_crop = min(frame_lim_y - y_ref_crop, int(np.floor(len_y_hum * prop_y)))
        else:
            len_y_crop = min(y_ref_crop, int(np.floor(len_y_hum * prop_y)))

        x_lim = np.sort([x_ref_crop, max(0, min(frame_lim_x, x_ref_crop + dir_x_crop * len_x_crop))])
        y_lim = np.sort([y_ref_crop, max(0, min(frame_lim_y, y_ref_crop + dir_y_crop * len_y_crop))])
        surf_crop = np.meshgrid(np.arange(y_lim[0], y_lim[1]), np.arange(x_lim[0], x_lim[1]))

        # Get bg surface to paste #####################
        # Surface of background available (without any part of human in the selection area)
        # Remove negative values
        surf_1 = np.meshgrid(range(top_hum - len_y_crop), range(frame_lim_x - len_x_crop))
        surf_2 = np.meshgrid(np.arange(max(0,top_hum - len_y_crop), frame_lim_y - len_y_crop),
                             np.arange(right_hum, max(right_hum,frame_lim_x - len_x_crop)))
        surf_3 = np.meshgrid(np.arange(bottom_hum, max(bottom_hum,frame_lim_y - len_y_crop)), range(right_hum))
        surf_4 = np.meshgrid(np.arange(max(0,top_hum - len_y_crop), min(frame_lim_y - len_y_crop, bottom_hum)), range(left_hum - len_x_crop))
        surf = [surf_1, surf_2, surf_3, surf_4]

        # Select a region in the available background #####################
        ix_surf = rdm.randint(0,3)
        count = 0
        while min(surf[ix_surf][0].shape) == 0 & count < 10:
            ix_surf = rdm.randint(0,3)
            count += 1

        if count == 10:
            print('Rejected')
            return rgb_img

        x_x_ref_bg = np.random.choice(range(surf[ix_surf][1].shape[0]))
        y_x_ref_bg = np.random.choice(range(surf[ix_surf][0].shape[1]))
        x_ref_bg = surf[ix_surf][1][x_x_ref_bg, 0]
        y_ref_bg = surf[ix_surf][0][0, y_x_ref_bg]
        surf_bg = np.meshgrid(np.arange(y_ref_bg, y_ref_bg + len_y_crop), np.arange(x_ref_bg, x_ref_bg + len_x_crop))

        # Paste the background on the cropping area #####################
        rgb_img[surf_crop] = rgb_img[surf_bg]

        # Update the segmentation matrix #####################
        #for k in range(surf_crop[0].shape[0]):
         #   for j in range(surf_crop[0].shape[1]):
          #      ix_x = surf_crop[1][k,j]
           #     ix_y = surf_crop[0][k,j]
            #    segm[ix_x, ix_y] = 0
        #segm[surf_crop] = 0
        #segm_dict[names_segm[i]] = segm
        # Save the resulting image #####################
        #img = Image.fromarray(rgb_img, 'RGB')
        #img.save(path_images+'abc_'+str(i)+'.jpg')
#### Save the segmentation dictionary updated
#scipy.io.savemat(path_segm + 'test_segm', segm_dict)
    else:
        print('No Humans')

    return rgb_img

#img = Image.open(path_images+"abc_10.jpg")
#segm_mat = scipy.io.loadmat(path + 'test_segm.mat')
#segm = segm_mat['segm_11']
#rgb_img = np.array(img)
#surf_segm = np.where(segm != 0)
#rgb_img[surf_segm] =  np.array([255, 0, 0])
#res = Image.fromarray(rgb_img, 'RGB')
#res
#img
