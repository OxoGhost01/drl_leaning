# Hi

Welcome to my repo where i post the code i did (or i'm currently working on) to learn DRL.

For that i use [this course](https://huggingface.co/learn/deep-rl-course/unit0/introduction), it's pretty detailed and i prefer texte-based courses rather than video-based ones x)

Fell free to explore, ask something, take the files or use it for something else, as long as you mention me.

## How-to-use

First, make sure you are on linux (it's so hard to find some packages on windows, and to get it work, so just use linux, it's the best for developpment).

Then, executes theses commands in your terminal (you can find them [here](https://huggingface.co/learn/deep-rl-course/unit1/hands-on#install-dependencies-and-create-a-virtual-screen-) too):
```
# if you don't have python installed (the commands are made for debian-based distros, using APT):
sudo apt install python3 python3-full python3-venv

# then install some packages :
sudo apt install swig cmake

# create a virtual environnement in your project folder:
python3 -m venv .venv

# install python dependencies:
pip install -r https://raw.githubusercontent.com/huggingface/deep-rl-class/main/notebooks/unit1/requirements-unit1.txt

# (optional) and if you want to follow the course, you'll need theses:
apt install python3-opengl
apt install ffmpeg
apt install xvfb
pip install pyvirtualdisplay
pip install moviepy
```
Have fun !