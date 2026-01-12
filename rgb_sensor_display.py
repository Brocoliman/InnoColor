import serial
import time
import colour
import numpy as np
from colour.plotting import *
import matplotlib.pyplot as plt


def draw_point(r, g, b):
    
    # Configure plotting styles
    colour_style()

    # Draw CIE 1976 UCS chromaticity diagram.
    plot_chromaticity_diagram_CIE1976UCS(standalone=False)

    # Convert YUV to XY (requires RGB intermediate step)
    rgb_normalized = np.array([[r/255, g/255, b/255]])
    xyz = colour.sRGB_to_XYZ(rgb_normalized)
    xy = colour.XYZ_to_xy(xyz)

    str_d = 'RGB('+str(r)+' ' +str(g)+' '+ str(b) + ')'
    plt.scatter(xy[0][0], xy[0][1], color='red', s=100, label=str_d)
    plt.legend()

    plt.title('CIE 1976 Gamut')
    plt.grid(True)
    plt.show()

# Must be adjusted to actual setup
arduino = serial.Serial('COM12', 9600, timeout=1)

while True:
    try:
        data = arduino.readline().decode('utf-8').strip()
        if data:
            values = data.split(',')  # Parse Arduino CSV data
            r = int(values[0])
            g = int(values[1])
            b = int(values[2])
            print(f"RGB data: {values[0]}, {values[1]}, {values[2]}")
            draw_point(r, g, b)
            time.sleep(3)
    except KeyboardInterrupt:
        arduino.close()
        break