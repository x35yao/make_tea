import numpy as np
from sklearn.linear_model import LinearRegression

def get_velocity(x):
    v = np.zeros_like(x)
    for i, data in enumerate(x):
        if i == 0:
            v[i] = np.nan
        else:
            v[i] = x[i] - x[i - 1]
    return v

def get_acceleration(v):
    a = np.zeros_like(v)
    for i, data in enumerate(v):
        if i == 0 or i == 1:
            a[i] = np.nan
        else:
            a[i] = v[i] - v[i - 1]
    return a


def get_gradient(x, data_type = 'trajectory'):
    assert data_type in ['trajectory', 'velocity', 'acceleration'], 'Data type has to be trajectory, velocity or acceleration'
    x_shift = np.roll(x, 1)
    x_p = x_shift - x
    if data_type == 'trajectory':
        x_p[0] = np.nan
    elif data_type == 'velocity':
        x_p[0:2] = np.nan
    elif data_type == 'accleration':
        x_p[0:3] = np.nan
    return x_p

def get_kinematics(x):
    v = get_gradient(x)
    a = get_gradient(v, data_type = 'velocity')
    a_p = get_gradient(a, data_type = 'acceleration')

    return v, a, a_p

def get_interpolation(x, t, window_size, kernel, direction = 'forward'):
    '''
    Parameters:
    x: the original trajectory data
    window_size: the size of the moving window
    kernel: The way to carry out the interpolation.
            Options: 'x_linear': The velocity is assumed a constant and the trajectory will be linear.
                     'v_linear': The velocity is assumed linear.
                     'a_linear': The acceleration is assumed linear.

    '''
    if direction == 'backward':
        x = np.flip(x)

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
                lr = LinearRegression()
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
                lr = LinearRegression()
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
                lr = LinearRegression()
                lr.fit(X_train, Y)
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
                lr = LinearRegression()
                lr.fit(X_train, Y)
                m = lr.coef_[0][0]
                b = lr.intercept_[0]
                dt = t[i] - t[i-1]
                x_p[i] = (x[i - 1]) + (v[i - 1]* dt) + (1/2 * a[i - 1] * dt**2) +\
                          (b/6 * dt**3 ) + m/24 * dt**4


    if direction == 'forward':
        return x_p
    else:
        return np.flip(x_p)

def get_error(x,t,window_size, kernel):
    forward_interpolation = get_interpolation(x, t, window_size, kernel, 'forward')
    back_interpolation = get_interpolation(x, t, window_size, kernel, 'backward')
    x_p = (forward_interpolation + back_interpolation) / 2
    error = x - x_p
    return error

def get_error_total(df, obj, window_size, kernel):

    x = df[obj]['x'].values
    y = df[obj]['y'].values
    z = df[obj]['z'].values
    t = df['time_stamp'].values

    error_x = get_error(x, t, window_size, kernel)
    error_y = get_error(y, t, window_size, kernel)
    error_z = get_error(z, t, window_size, kernel)

    return np.sqrt(np.nan_to_num(error_x**2) + np.nan_to_num(error_y**2) + np.nan_to_num(error_z**2))
