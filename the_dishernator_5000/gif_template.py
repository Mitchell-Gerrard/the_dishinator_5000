import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation

if __name__=='__main__':
    def sin(x,t):
        return np.sin(x*np.log(t))
    nploty=1000
    figure_folder = 'the_dishernator_5000/figures'
    time=np.linspace(1,1000,nploty)
    x=np.linspace(-10,10,10000)
    fig, ax = plt.subplots()
    def update(frame):
     ax.clear()
     ax.set_xlabel('Time')
     ax.set_ylabel('Sin(xt)')
     ax.plot(x,sin(x,time[frame]))  # Plot x and y data for the frame
     ax.set_title(f'The Sins Boi {time[frame]}')  # Update the title based on the frame
     print(f'Frame {frame}/{nploty}')

# Create the animation and save as GIF
    ani = FuncAnimation(fig, update, frames=nploty, repeat=True)

# Save the animation as a GIF
    ani.save(f'{figure_folder}/sin.gif', writer='pillow', fps=30)  # Adjust fps as needed