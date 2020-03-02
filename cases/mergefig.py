import numpy as np
from imageio import get_reader, mimsave

# Create reader object for the gif
gif0 = get_reader('../../tmp/chains_best.gif')
gif1 = get_reader('../../tmp/chains.gif')
gif2 = get_reader('../../tmp/conv.gif')

# If they don't have the same number of frame take the shorter
number_of_frames = min(gif0.get_length(), gif1.get_length(), gif2.get_length())

# Create writer object
images = []

for frame_number in range(number_of_frames):
    img0 = gif0.get_next_data()
    img1 = gif1.get_next_data()
    img2 = gif2.get_next_data()
    new_image = np.hstack((img0, img1, img2))
    images.append(new_image)

mimsave('../../tmp/analyt_full.gif', images, format='GIF', duration=0.25)
