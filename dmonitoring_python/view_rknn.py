import inspect

# Import the module
from rknnlite.api import RKNNLite

# Get a list of all names defined in the module
names = dir(RKNNLite)

# Iterate over the names and check if each is a function
for name in names:
    # Get the object for the name
    obj = getattr(RKNNLite, name)
    # Check if the object is a function
    if inspect.isfunction(obj):
        print(name)
        
rknn = RKNNLite(verbose=False)
platform = rknn.get_sdk_version()
print(platform)