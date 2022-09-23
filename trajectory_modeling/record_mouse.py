from pynput.mouse import Listener
import numpy as np
import time
import os

file_name = str(time.time()).split('.')[0] + '.npy'
destfolder = './TP_GMM/data/wiping'
data = []
screen_width = 1920
def on_move(x, y):
    print('Pointer moved to {0}'.format(
        (x, y)))
    data.append([x - screen_width, y])


def on_click(x, y, button, pressed):
    print('{0} at {1}'.format(
        'Pressed' if pressed else 'Released',
        (x, y)))
    if not pressed:
        return False

def on_scroll(x, y, dx, dy):
    print('Scrolled {0}'.format(
        (x, y)))

# Collect events until released
with Listener(
        on_move=on_move,
        on_click=on_click,
        on_scroll=on_scroll) as listener:
    listener.join()
data = np.array(data)
np.save(os.path.join(destfolder, file_name), data)