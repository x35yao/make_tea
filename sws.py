import numpy as np
from sklearn.linear_model import LinearRegression,RANSACRegressor
from scipy.spatial.transform import Rotation as R

# def get_velocity(x):
#     v = np.diff(x)
#     return np.r_[np.nan, v]
#
# def get_acceleration(v):
#     a = np.diff(v)
#     return np.r_[np.nan, a]


def get_gradient(x):
    x_p = np.diff(x)
    return np.r_[np.nan, x_p]

def get_kinematics(x):
    v = get_gradient(x)
    a = get_gradient(v)
    a_p = get_gradient(a)

    return v, a, a_p

def harersine_distance(v1, v2, r = 1):
    delta1 = np.sqrt(np.sum((v1 - v2)**2, axis = 1))
    ang1 = np.arcsin(delta1/(2 * r))
    dist1 = 2 * ang1 * r

    # delta2 = np.sqrt(np.sum((v1 - v2)**2, axis = 1))
    # ang2 = np.arcsin(delta2/(2 * r))
    # dist2 = 2 * ang2 * r
    #
    # dist = np.min( np.c_[dist1, dist2], axis = 1)
    return dist1

def pos_interpolation(x, t, window_size, kernel, direction = 'forward', linear_model = 'linear'):
    '''
    Parameters:
    x: the original trajectory data
    window_size: the size of the moving window
    kernel: The way to carry out the interpolation.
            Options: 'x_linear': The velocity is assumed a constant and the trajectory will be linear.
                     'v_linear': The velocity is assumed linear.
                     'a_linear': The acceleration is assumed linear.

    '''
    assert kernel in['x_linear', 'v_linear', 'a_linear', 'a_p_linear'], 'Invalid kernel'
    if direction == 'backward':
        x = np.flip(x)
    if linear_model == 'ransac':
        lmodel = RANSACRegressor
    elif linear_model == 'linear':
        lmodel = LinearRegression
    x_p = np.zeros_like(x)
    half_ind = int(window_size / 2)
    if kernel == 'x_linear':
        assert window_size >= 5, 'The smallest window size for x_linear is 5'
        for i, data in enumerate(x):
            if i < half_ind:
                x_p[i] = np.nan
            else:
                Y = x[i - half_ind: i].reshape((-1, 1))
                X = t[i - half_ind: i +1 ]
                X_train = X[:-1].reshape((-1, 1))# All the points except the last one
                X_test = X[-1].reshape((-1, 1)) # Just the last point
                lr = lmodel()
                lr.fit(X_train, Y)
                x_p[i] = lr.predict(X_test)

    elif kernel == 'v_linear':
        assert window_size >= 7, 'The smallest window size for v_linear is 7'
        v = get_gradient(x)
        for i, data in enumerate(x):
            if i < half_ind:
                x_p[i] = np.nan
            else:
                # method 1
                Y = v[i - half_ind + 1: i].reshape((-1, 1))
                X = t[i - half_ind + 1: i + 1 ]
                X_train = X[:-1].reshape((-1, 1))# All the points except the last one
                X_test = X[-1].reshape((-1, 1)) # Just the last point
                lr = lmodel()
                lr.fit(X_train, Y)
                v_test = lr.predict(X_test)
                x_p[i] = x[i - 1] + v_test
#                 # method 2
#                 a = lr.coef_[0][0]
#                 x_p[i] = x[i - 1] + v[i - 1] * (t[i] - t[i - 1]) + 0.5 * a *(t[i] - t[i - 1])**2

    elif kernel == 'a_linear':
        assert window_size >= 9, 'The smallest window size for a_linear is 9'
        v, a, _ = get_kinematics(x)
        for i, data in enumerate(x):
            if i < half_ind:
                x_p[i] = np.nan
            else:
                Y = a[i - half_ind + 2: i].reshape((-1, 1))
                X = t[i - half_ind + 2: i + 1]
                X_train = X[:-1].reshape((-1, 1))# All the points except the last one
                X_test = X[-1].reshape((-1, 1)) # Just the last point
                lr = lmodel()
                lr.fit(X_train, Y)
                if linear_model == 'ransac':
                    m = lr.estimator_.coef_[0][0]
                    b = lr.estimator_.intercept_[0]
                else:
                    m = lr.coef_[0][0]
                    b = lr.intercept_[0]
                dt = t[i] - t[i-1]
                x_p[i] = x[i - 1] + v[i - 1]* dt+ b/2 * dt**2 + m/6 * dt**3

    elif kernel == 'a_p_linear':
        assert window_size >= 11, 'The smallest window size for a_linear is 11'
        v, a, a_p = get_kinematics(x)
        for i, data in enumerate(x):
            if i < half_ind:
                x_p[i] = np.nan
            else:
                Y = a_p[i - half_ind + 3: i].reshape((-1, 1))
                X = t[i - half_ind + 3: i + 1]
                X_train = X[:-1].reshape((-1, 1))# All the points except the last one
                X_test = X[-1].reshape((-1, 1)) # Just the last point
                lr = lmodel()
                lr.fit(X_train, Y)
                if linear_model == 'ransac':
                    m = lr.estimator_.coef_[0][0]
                    b = lr.estimator_.intercept_[0]
                    print(m, b)
                else:
                    m = lr.coef_[0][0]
                    b = lr.intercept_[0]
                dt = t[i] - t[i-1]
                x_p[i] = (x[i - 1]) + (v[i - 1]* dt) + (1/2 * a[i - 1] * dt**2) +\
                          (b/6 * dt**3 ) + m/24 * dt**4


    if direction == 'forward':
        return x_p
    else:
        return np.flip(x_p)

def ori_interpolation(XYZ, window_size = 2, direction = 'forward'):
    '''
    Parameters:
    XYZ: the original orientation data
    window_size: the size of the moving window
    kernel: The way to carry out the interpolation.
            Options: 'x_linear': The velocity is assumed a constant and the trajectory will be linear.
                     'v_linear': The velocity is assumed linear.
                     'a_linear': The acceleration is assumed linear.

    '''
    print(window_size)
    if direction == 'backward':
        XYZ = np.flip(XYZ)
    XYZ_p = np.zeros_like(XYZ)
    for i, data in enumerate(XYZ):
        if i < window_size:
            # XYZ_p[i] = np.nan
            pass
        else:
            v1 = XYZ[i - window_size: i]
            v2 = XYZ[i - window_size + 1 : i + 1]
            # print(v1.reshape(-1,3), v2)
            # raise
            r = R.align_vectors(v1, v2)
            rot_matrix = r[0].as_matrix()
            v = XYZ[i - 1]
            XYZ_p[i,:] = rot_matrix @ v
    XYZ_p[:window_size] = np.nan
    if direction == 'forward':
        return XYZ_p
    elif direction == 'backward':
        return np.flip(XYZ_p)


def get_error_pos(x,t,window_size, kernel):
    forward_interpolation = pos_interpolation(x, t, window_size, kernel,'forward')
    back_interpolation = pos_interpolation(x,  t, window_size, kernel, 'backward')
    x_p = (forward_interpolation + back_interpolation) / 2
    # x_p = forward_interpolation
    error = x - x_p
    return error

def get_error_ori(XYZ, window_size):
    forward_interpolation = ori_interpolation(XYZ, window_size = window_size, direction = 'forward')
    back_interpolation = ori_interpolation(XYZ, window_size = window_size, direction =  'backward')
    XYZ_p = (forward_interpolation + back_interpolation) / 2
    XYZ_p = np.nan_to_num(XYZ_p)
    # XYZ_p = np.nan_to_num(forward_interpolation)

    dist = harersine_distance(XYZ, XYZ_p)
    # dist = np.sqrt(np.sum((XYZ - XYZ_p)**2, axis = 1))
    dist[:2] = dist[-2:] = 0
    return dist

def get_error_total(df, obj, window_size, kernel, with_ori = True, normalized = False):

    x = df[obj]['x'].values
    y = df[obj]['y'].values
    z = df[obj]['z'].values
    t = df['time_stamp'].values
    error_x = get_error_pos(x, t, window_size, kernel)
    error_y = get_error_pos(y, t, window_size, kernel)
    error_z = get_error_pos(z, t, window_size, kernel)
    error_pos = np.sqrt(np.nan_to_num(error_x**2) + np.nan_to_num(error_y**2) + np.nan_to_num(error_z**2))
    error_pos_normalized = error_pos / np.linalg.norm(error_pos)
    if with_ori:

        XYZ = df[obj][['X', 'Y', 'Z']].values
        error_ori = get_error_ori(XYZ, window_size = 3)
        error_ori[error_ori> 0.5] = 0
        # error_ori[np.where(error_ori>1)[0]] = 0
        error_ori_normalized = error_ori / np.linalg.norm(error_ori)
        # X = df[obj]['X'].values
        # Y = df[obj]['Y'].values
        # Z = df[obj]['Z'].values
        # t = df['time_stamp'].values
        # error_X = get_error_pos(X, t, window_size, kernel)
        # error_Y = get_error_pos(Y, t, window_size, kernel)
        # error_Z = get_error_pos(Z, t, window_size, kernel)
        # error_ori= np.sqrt(np.nan_to_num(error_X**2) + np.nan_to_num(error_Y**2) + np.nan_to_num(error_Z**2))
        # error_ori_normalized = error_ori / np.linalg.norm(error_ori)
    else:
        error_ori = error_ori_normalized = None
    if normalized:
        return  error_pos_normalized, error_ori_normalized
    else:
        return error_pos, error_ori
