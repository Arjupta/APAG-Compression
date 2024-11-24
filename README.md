# APAG-Compression
Customised image compression llibrary made as an assignment of Digital Image Processing 

# Setting up
You need to have open-cv installed on your system for which you can do 

`pip install opencv-python`

After installing verify it with this command

`python -c "import cv2; print(cv2.__version__)"`

We also need Matplotlib to visualize the RSME vs BPP plots

`pip install pillow matplotlib`

# Running the script
You can import the `agag_compression.py` and use it's compress and decompress functions in your project. Also we have create an `analysis.py` which compares our compression mechanism with JPEG for a set of test images and then plots the RSME vs BPP curve for them.

Run the analysis like this and see the plot in the `output` directory.

`python analysis.py`

