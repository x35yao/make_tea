import os
import math
import imageio
import numpy as np
import pickle as pkl

from statistics import mean
from sklearn.metrics import mean_squared_error


GRID_LENGTH = 4.95
BASELINE = 12 # cm
PIXEL_LENGTH = 0.0002 # cm
FOCAL_LENGTH = PIXEL_LENGTH*1000 # cm (may need to double check camera model)

def distanceCal(x,y,z):
    return math.sqrt(x**2 + y**2 + z**2)

def formHeaders(prefix, parts):
    new_headers = prefix
    for part in parts:
        for j, word in enumerate(part):
            new_headers[j] = '-'.join([new_headers[j], word])
    return new_headers

def getVectorDisplacement(coord3D, selected_indices=[]):
    """
    convert 3D coordinates of objects to their corresponding displacement
    vectors from frame to frame by selected indices or all frames if empty.
    """
    prev_coord, result_vectors = False, []
    for ind, object_coord in enumerate(coord3D):
        if len(selected_indices) > 1 and not ind in selected_indices:
            continue
        if not prev_coord:
            prev_coord = object_coord
        else:
            obj_vectors = []
            for m in range(len(object_coord)):
                X_delta = object_coord[m][0] - prev_coord[m][0]
                Y_delta = object_coord[m][1] - prev_coord[m][1]
                Z_delta = object_coord[m][2] - prev_coord[m][2]
                obj_vectors.append([X_delta, Y_delta, Z_delta])
            result_vectors.append(obj_vectors)
            prev_coord = object_coord
    return result_vectors

def pixelInterpolation(x, y, f):
    """
    Interpolate the value of the f at the location of x, y
    using bilinear interpolation.
    """
    dims = f.shape
    epsilon = 0.000001
    x, y = min(x + epsilon, dims[1]), min(y + epsilon, dims[0])
    x1, x2 = math.floor(x), math.ceil(x)
    y1, y2 = math.floor(y), math.ceil(y)
    fxy2 = (x2 - x)*f[y2, x1] + (x - x1)*f[y2, x2]
    fxy1 = (x2 - x)*f[y1, x1] + (x - x1)*f[y1, x2]
    return (y - y1)*fxy2 + (y2 - y)*fxy1

def pixelTo3DCameraCoord(img, disp_map, coords, is_left_image=True):
    """
    Returns a list of 3D coordinates and pixel coordinates
    by triangulation calculation done on the disparity map
    at the pixel locations provided by coords as a list.
    """
    result_coords = []
    img_dims, disp_dims = img.shape, disp_map.shape
    fixed_ratios = [img_dims[dim]/disp_dims[dim] for dim in range(2)]
    for _, pix in enumerate(coords):
        # Different ordering of dims between coordinate and images fixed here.
        if len(pix) == 0: continue
        if np.isnan(pix).any():
            result_coords.append({'x1':np.nan, 'y':np.nan, 'x2':np.nan, 'X':np.nan, 'Y':np.nan, 'Z':np.nan})
        else:
            x1, y = (int(i) for dim, i in enumerate(pix))
            d_x, d_y = pix[0]/fixed_ratios[1], pix[1]/fixed_ratios[0]
            d = pixelInterpolation(d_x, d_y, disp_map)*fixed_ratios[1]
            if is_left_image:
                x2 = int(x1 - d)
            else:
                x2 = int(x1 + d)
            # Z is the depth from camera center in cm and X, Y for the other 2 axis.
            Z = BASELINE*FOCAL_LENGTH/(d*PIXEL_LENGTH)
            X, Y = (x1 - img_dims[1]/2)*PIXEL_LENGTH*Z/FOCAL_LENGTH, (y - img_dims[0]/2)*PIXEL_LENGTH*Z/FOCAL_LENGTH
            result_coords.append({'x1':x1, 'y':y, 'x2':x2, 'X':X, 'Y':Y, 'Z':Z})
    return result_coords

def cameraCoord3DToPixel(img, coord3D):
    """
    Returns a list of original pixel coordinates provided by coord3D as a list of dict.
    """
    pixel_coords = []
    img_dims, disp_dims = img.shape, disp_map.shape
    for _, point3d in enumerate(coord3D):
        Z = point3d['Z']
        x = point3d['X']*FOCAL_LENGTH/(PIXEL_LENGTH*Z) + img_dims[1]/2
        y = point3d['Y']*FOCAL_LENGTH/(PIXEL_LENGTH*Z) + img_dims[0]/2
        pixel_coords.append([x, y])
    return pixel_coords

def LEAStereoCoordinate(img_path, disp_path, data, is_left_image=True):
    """
    Returns a dict of all images in img_path containing
    the pixel and 3D coordinates of the interested pixel
    location provided by datafile.
    """
    files_id = [file.split('.')[0][5:] for file in os.listdir(img_path)]
    disp_files = os.listdir(disp_path)
    coordinate3D = {}
    for framename in data.keys():
        if framename[:5]=='frame':
            frame_id = framename[5:]
            # fixing some frame number inconsistencies.
            correspond_id = None
            for file_id in files_id:
                if int(file_id)==int(framename[5:]):
                    correspond_id = file_id
            if correspond_id==None: continue
            img_file = 'frame{}.jpg'.format(correspond_id)
            disp_file = "frame{}_disp.npy".format(correspond_id)
            img_disp = np.load(disp_path + disp_file)
            img = imageio.imread(img_path + img_file)
            # convert left view disparity to right
            if is_left_image:
                img_disp = Left2RightDisparity(img_disp)
            frame_coord = data[framename]
            best_pixels = []
            coord, confid = frame_coord['coordinates'][0], frame_coord['confidence']
            for k, _ in enumerate(coord):
                confid_k = list(np.reshape(confid[k], (-1)))
                best_index = confid_k.index(max(confid_k))
                best_pixels.append(coord[k][best_index])
            coordinate3D[framename] = pixelTo3DCameraCoord(img, img_disp, best_pixels, is_left_image)
    return coordinate3D

def getMeasurements(model_pos):
    """
    Returns vector displacement and magnitude of displacement
    for each position change provided by model_pos.
    """
    model_dist = []
    model_vect = getVectorDisplacement(model_pos)
    for vector_set in model_vect:
        dist_set = []
        for vec in vector_set:
            dist_set.append(distanceCal(*vec))
        model_dist.append(mean(dist_set))
    return model_vect, model_dist

def findLosses(true_values, raw_values, scales):
    losses = []
    for scaling in scales:
        new_raw_values = [scaling*i for i in raw_values]
        losses.append(mean_squared_error(true_values, new_raw_values))
    return losses

def consistencyLoss(vectors):
    """
    Compare scaled unit steps to the set of steps actually moved to test consistency.
    Assumes vectors input is consisted of vectors moving in a straight line.
    """
    scaled_vectors, actual_vectors = [], []
    sum_vectors = []
    for frame in vectors:
        for point in frame:
            sum_vectors.append(np.linalg.norm(point))
    norm_val = mean(sum_vectors)
    for i, cur_vectors in enumerate(vectors):
        before_set, after_set = vectors[:i], vectors[i+1:]
        for j, _ in enumerate(cur_vectors):
            if len(before_set) > 0:
                extrap_vector = [-sub_vector*len(before_set)/norm_val for sub_vector in cur_vectors[j]]
                scaled_vectors = scaled_vectors + extrap_vector
                actual_vector = [-k/norm_val for k in before_set[-1][j]]
                for obj_vectors in before_set[::-1][1:]:
                    actual_vector = [actual_vector[k] - sub_vector/norm_val for k, sub_vector in enumerate(obj_vectors[j])]
                actual_vectors = actual_vectors + actual_vector
            if len(after_set) > 0:
                extrap_vector = [sub_vector*len(after_set)/norm_val for sub_vector in cur_vectors[j]]
                scaled_vectors = scaled_vectors + extrap_vector
                actual_vector = [k/norm_val for k in after_set[0][j]]
                for obj_vectors in after_set[1:]:
                    actual_vector = [actual_vector[k] + sub_vector/norm_val for k, sub_vector in enumerate(obj_vectors[j])]
                actual_vectors = actual_vectors + actual_vector
    return mean_squared_error(actual_vectors, scaled_vectors)


def Left2RightDisparity(disp_map, width=1):
    """
    Convert disparity from left view to right view.
    """
    dims = disp_map.shape
    new_disparity = np.zeros(dims, dtype='float32')
    for x in range(dims[0]):
        for y in range(dims[1]):
            dxy = disp_map[x, y]
            h_shift = round(y-dxy)

            if h_shift > 0 and new_disparity[x, h_shift] < dxy:
                new_disparity[x, h_shift] = dxy
    # For any missing or occlusions, apply smoothing.
    smooth_disparity = new_disparity.copy()
    for x in range(dims[0]):
        for y in range(width, dims[1]-width):
            if new_disparity[x, y]==0:
                window = new_disparity[x, y-width:y+width+1]
                lst = window[window!=0.0]
                if len(lst) > 0:
                    smooth_disparity[x, y] = np.mean(window[window!=0.0])
    return smooth_disparity
