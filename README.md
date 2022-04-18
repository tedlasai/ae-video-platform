# 4D Temporal Stack Data Browser
This repository is for the GUI codes developed to browse the 4d temporal stack data in general.

## Dataset
The exposure stack data is used as an example to run this data-browsing GUI.

To avoid latency in opening and reading images every time we use the data slider, the images are pre-processed and stored in a four-dimensional tensor where the first dimension is the image index and the rest are the typical image three channels.

### Four-dimensional tensor data:
* [Scene 1 (12x90 images)](https://drive.google.com/file/d/1EndfDPVuNnxLzZ7_2Sx2Dfx6njRJOdYS/view?usp=sharing)
* [Scene 2 (47x65 images)](https://drive.google.com/file/d/10eAPVmmw1DlmimYymXPoOcpFndA6My6g/view?usp=sharing)

<b>This data is processed to an sRGB, encoded with a lossless 8-bit depth, and downscaled with a 0.12 ratio.</b>
<b>Download the data and place it inside the same directory of the GUI codes.</b>

## Code
### Prerequisites
* The code tested with:
	* Python 3.8.3
	* Numpy 1.19.1
	* Pillow 7.0.0
	* Tkinter (Tk) 8.6.8
	* OpenCV 4.4.0
	* Matplotlib 3.1.3
	
	<i>Despite not tested, the code may work with library versions other than the specified.</i>
  
### Installation
* Clone with HTTPS this project to your local machine 
```bash
git clone https://github.com/Abdullah-Abuolaim/4d-data-browser.git
cd ./4d-data-browser/
```

### Prepare the four-dimensional data
If the data is sorted following the Canon naming convention, all you need is to run the code in `prepare_data.py`.

<b>Recall that you need to adjust the `read_path` variable inside the code to the directory where the images are stored in your machine.</b>
