import cProfile
import os

# For correct work, install following packages:
# pip install gprof2dot
# brew install graphviz

def profile(func):
    """Decorator for run function profile"""
    def wrapper(*args, **kwargs):
        profile_filename = func.__name__ + '.prof'
        profiler = cProfile.Profile()
        result = profiler.runcall(func, *args, **kwargs)
        profiler.dump_stats(profile_filename)
        
        return result
    
    return wrapper 

# After run function, run following command, where FILE_NAME = name of your .prof file
# PNG_NAME = name of your future .png file
os.system("gprof2dot -f pstats FILE_NAME.prof | dot -Tpng -o PNG_NAME.png")


# Example:

# Append decorator to your function, which you want to anylyze
@profile
def function():
    lst = []
    for i in range(10):
        lst.append(i)
    return lst
# Run your function
function()
# Create .png from .prof file
os.system("gprof2dot -f pstats function.prof | dot -Tpng -o function_picture.png")
# Profit! Open function_picture.png and relax