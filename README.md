# classifier_pacman
For this project, I will implement a naive bayesian classifier. I will use this classifier in some code that has to make a decision. The code will be controlling Pacman, in the classic game, and the decision will be about how Pacman chooses to move. However, it probably won’t help Pacman to make particularly good decisions.

## Environment

My code is in Python 2.7.

## Background

The Pacman code that we will be using for the 7CCSMML1 coursework was developed at UC Berkeley for their AI course. The folk who developed this code then kindly made it available to everyone. The homepage for the Berkeley AI Pacman projects is here:

http://ai.berkeley.edu/


## Code to randomly control Pacman

python pacman.py --pacman RandomAgent

## Navie bayesian classifier to control Pacman

This class provides some simple data handling. It reads data from a file called good-moves.txt and turns it into arrays target and data which are similar to the ones you have used with scikit-learn. When we test your code, it will have to be able to read data in the same format as good-moves.txt, from a file called good-moves.txt.

python pacman.py --pacman ClassifierAgent


### Author

Minyu Luo


### Version

1.0
