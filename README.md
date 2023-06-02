# Opencv-simple_video_editor

## Quick background
Writting this simple GUI for one of my project, you can tune the contrast brightness, gaussian  filter level and Binary therohold here, and see the effect immediately in the window, may help you accelerate your workflow and knowing the best parameter for your current file. 
- Based on Opencv & Qt5
## Can do:
##### Import a video and watch (dragging the slider bar under the preview window):
##### add effect
- tune contrast and brightness
- add Gaussian Blur
- add Binary 
-  functions listed below just can see in preview **will not** be add as a effect on the export video yet, ****tired to write the code at this moment***
- add gamma correction
- add bilateral effect
- add sharpen effect
if you want to save the video after adding the effects simply click **'save'** button

## Config:
~~~bash 
git clone https://github.com/KIllerLB-USTC/Opencv-simple_video_editor.git
~~~
~~~python
# go into the dirct
pip install -r requirements.txt
~~~
## Use
by simple type in 
~~~python
python simple_video_editor.py
~~~
Demo:

<iframe id="ytplayer" type="text/html" width="640" height="360"
  src="https://www.youtube.com/embed/ESAOZNu6yhg"
  frameborder="0"></iframe>

  