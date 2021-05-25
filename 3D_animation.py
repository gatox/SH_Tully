# =============================================================================
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Tue Jan  5 14:09:55 2021
# 
# @author: edisonsalazar
# """

# =============================================================================
# To see the movement of the particle, change the preferences/console/ghapics/
# bakend from linear to Q5
# 
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

def make_helix(n):
    theta_max = 8 * np.pi
    theta = np.linspace(0, theta_max, n)
    x, y, z = theta, np.sin(theta), np.cos(theta)
    helix = np.vstack((x, y, z))

    return helix

def update_lines(num, dataLines, lines) :
    for line, data in zip(lines, dataLines) :
        line.set_data(data[0:2, num-1:num])
        line.set_3d_properties(data[2,num-1:num])
    return lines

# Attach 3D axis to the figure
fig = plt.figure()
ax = p3.Axes3D(fig)

n = 100
data = [make_helix(n)]

lines = [ax.plot(data[0][0,0:1], data[0][1,0:1], data[0][2,0:1], 'o')[0]]

# Setthe axes properties
ax.set_xlim3d([0.0, 8*np.pi])
ax.set_xlabel('X')

ax.set_ylim3d([-1.0, 1.0])
ax.set_ylabel('Y')

ax.set_zlim3d([-1.0, 1.0])
ax.set_zlabel('Z')

ax.set_title('3D Test')

# Creating the Animation object
ani = animation.FuncAnimation(fig, update_lines, n, fargs=(data, lines),
                              interval=50, blit=False)
plt.show()
