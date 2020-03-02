import numpy as np
from imageio import get_reader, get_writer

# Create reader object for the gif
gif1 = get_reader('../../tmp/chains.gif')
gif2 = get_reader('../../tmp/conv.gif')

# If they don't have the same number of frame take the shorter
number_of_frames = min(gif1.get_length(), gif2.get_length())

# Create writer object
new_gif = get_writer('../../tmp/analyt.gif')

for frame_number in range(number_of_frames):
    img1 = gif1.get_next_data()
    img2 = gif2.get_next_data()
    new_image = np.hstack((img1, img2))
    new_gif.append_data(new_image)

gif1.close()
gif2.close()
new_gif.close()
