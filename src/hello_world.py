import numpy as np

#

if __name__ == '__main__':
    msg = np.array(['h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd'], dtype=str)
    print(''.join(msg.tolist()))
