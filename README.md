# Examining Autoexposure for Challenging Scenes

<p align="center">
												<a href="https://techsaico.com/">SaiKiran Tedla*</a>
			&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 	<a href="https://beixuanyang.com/">Beixuan Yang*</a>
			&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;	<a href="https://www.eecs.yorku.ca/~mbrown/">Michael S. Brown</a>
	<br>
	York University
</p>

[//]: # (<img src="./figures/MDP-NIMAT-fast-2.gif" width="100%" alt="teaser gif">)

Detailed information of this project shall be found on the [Project Website](https://ae-video.github.io/).

Reference github repository for the paper [Examining Autoexposure for Challenging Scenes](https://arxiv.org/pdf/2309.04542.pdf). Tedla et al., Proceedings of IEEE International Conference on Computer Vision 2023 ([YouTube presentation](https://www.youtube.com/watch?v=ZeHqNPD1UXg)). If you use our dataset or code, please cite our paper:
```
@inproceedings{Tedla2023,
  title={Examining Autoexposure for Challenging Scenes},
  author={Tedla, SaiKiran and Yang, Beixuan and Brown, Michael S},
  booktitle={Proceedings of IEEE International Conference on Computer Vision},
  year={2023}
}
```
## 4D AutoExposure Dataset
We captured a temporal exposure dataset with four dimensions: time * exposure * height * width.


The [Download Link](https://ln5.sync.com/dl/8b886b5b0/mp2h6pjc-3kkmhqr4-7gjgvpae-czxz58jn) redirects you to the download page of the 4D AE dataset. 

It contains 4 folders named as 'dng', 'sRGB_npy', 'Saliency_map_npy', and 'RAW_npy'.
In the 'dng' folder, 9 sub-folders are listed where each of them has 1500 RAW images (6720 * 4480 pixels) stored in the dng formate.
They are ordered as 100 frame * 15 exposure (15 s, 8 s, 6 s, 4 s, 2 s, 1 s, 1/2 s, 1/4 s, 1/8 s, 1/15 s, 1/30 s, 1/60 s,
1/125 s, 1/250 s, 1/500 s). The 'sRGB_npy' folder contains 9 npy files in the size of 100 frame * 40 exposure * 640 * 960 * 3,
the 'Saliency_map_npy' folder contains 9 npy files with the Saliency maps produced for the "Saliency AE algorithm",
and 'RAW_npy' has 9 files for each scene in the size of 100 frame * 40 exposure * 1120 * 1680. The 40 exposure time includes 15 s, 
13 s, 10 s, 8 s, 6 s, 5 s, 4 s, 3.2 s, 2.5 s, 2 s, 1.6 s, 1.3 s, 1 s, 0.8 s, 0.6 s, 0.5 s, 0.4 s, 0.3 s, 1/4 s, 1/5 s, 1/6 s, 1/8 s, 
1/10 s, 1/13 s, 1/15 s, 1/20 s, 1/25 s, 1/30 s, 1/40 s, 1/50 s, 1/60 s, 1/80 s, 1/100 s, 1/125 s, 1/160 s, 1/200 s, 1/250 s, 1/320 s,
1/400 s, 1/500 s.

[//]: # (## Our New Image Motion Attribute &#40;NIMAT&#41; Effect)

[//]: # (<img src="./figures/nimat.gif" width="100%" alt="NIMAT effect">)

## Code of the AE algorithm examining platform
A Python-based AE evaluation platform is also developed to work with our dataset. 4 AE strategies including 'Global',
'Semantic', 'Saliency'(we proposed), and 'Entropy' We examined using our dataset are also embedded in the released code.

### Prerequisites
* The code tested with:
	* Python 3.10.0 
    * numpy~=1.23.2
    * seaborn~=0.12.2
    * matplotlib~=3.5.3
    * Pillow~=9.2.0
    * opencv-python~=4.6.0.66
    * rawpy~=0.17.2
    * scikit-learn~=1.1.3
    * networkx~=2.8.8
    * scipy~=1.9.1
    * scikit-image~=0.19.3
    * plotly~=5.10.0
    * dash~=2.6.2
    * dbr~=9.6.0
    * imutils~=0.5.4
    * RangeSlider~=2021.7.4
    * ExifRead~=3.0.0

          <i>Despite not tested, the code may work with library versions other than the specified</i>

### Installation
* Clone with HTTPS this project to your local machine 
```bash
git clone https://github.com/tedlasai/4d-data-browser.git
cd ./4d-data-browser/
```

### Testing
* Download the npy files in 'sRGB_npy', 'Saliency_map_npy', and 'RAW_npy' from the [4D AE Dataset](https://ln5.sync.com/dl/8b886b5b0/mp2h6pjc-3kkmhqr4-7gjgvpae-czxz58jn)

* Create three directories under the root of the project to store the downloaded npy files .

* Modify the variables (path and file names) from line 1 to line 9 in the 'constants.py' file as needed. Or, you may modify
  the paths in the 'browser_builder.py' directly if your data is stored in other directories.
 
* Run the 'run_algorithms_in_browser.py'

* Follow the steps in the video.

* Note that we implemented a step size limit function to prevent a large jump of exposure.

* Note that the outlier setting of our user study discussed on our paper is set to be 'over 0.9'. If you would like to reproduce the result,
  please set the outlier slider to be '0-0.9'.

## Contact

Should you have any question/suggestion, please feel free to reach out:

[SaiKiran Tedla](https://techsaico.com/) (tedlasai@yorku.ca).

[Beixuan Yang](https://beixuanyang.com/) (beixuan@yorku.ca).
