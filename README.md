# 4D Temporal Stack Data Browser
This repository is for the GUI codes developed to browse the 4d temporal stack data in general.

## Code
### Prerequisites
* The code tested with:
	* Python 3.8.3
	* Numpy 1.19.1
	* Pillow 7.0.0
	* Tkinter (Tk) 8.6.8
	* OpenCV 4.4.0
	* Matplotlib 3.1.3
	
	<i>Despite not tested, the code may work with library versions other than the specified</i>
  
### Installation
* Clone with HTTPS this project to your local machine 
```bash
git clone https://github.com/Abdullah-Abuolaim/4d-data-browser.git
cd ./4d-data-browser/
```

## Dataset
The exposure stack data is used as an example to run this data-browsing GUI.
To avoid latency in opening and reading images every time we use the data slider, the images are pre-processed and stored in a four-dimensional tensor where the first dimension is the image index and the rest are the typical image three channels.
