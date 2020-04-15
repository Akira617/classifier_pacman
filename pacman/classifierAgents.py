# classifierAgents.py
# parsons/07-oct-2017
#
# Version 1.0
#
# Some simple agents to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agents here are extensions written by Simon Parsons, based on the code in
# pacmanAgents.py

from pacman import Directions
from game import Agent
import api
import random
import game
import util
import sys
import os
import csv
import numpy as np
from sklearn.model_selection import train_test_split

# ClassifierAgent
#
# An agent that runs a classifier to decide what to do.
class ClassifierAgent(Agent):

    # Constructor. This gets run when the agent starts up.
    def __init__(self):
        print 'Initialising'

    # Take a string of digits and convert to an array of
    # numbers. Exploits the fact that we know the digits are in the
    # range 0-4.
    #
    # There are undoubtedly more elegant and general ways to do this,
    # exploiting ASCII codes.
    def convertToArray(self, numberString):
        numberArray = []
        for i in range(len(numberString) - 1):
            if numberString[i] == '0':
                numberArray.append(0)
            elif numberString[i] == '1':
                numberArray.append(1)
            elif numberString[i] == '2':
                numberArray.append(2)
            elif numberString[i] == '3':
                numberArray.append(3)
            elif numberString[i] == '4':
                numberArray.append(4)

        return numberArray

    # This gets run on startup. Has access to state information.
    #
    # Here we use it to load the training data.
    def registerInitialState(self, state):

        # open datafile, extract content into an array, and close.
        self.datafile = open('good-moves.txt', 'r')
        content = self.datafile.readlines()
        self.datafile.close()

        # Now extract data, which is in the form of strings, into an
        # array of numbers, and separate into matched data and target
        # variables.
        self.data = []
        self.target = []
        # Turn content into nested lists
        for i in range(len(content)):
            lineAsArray = self.convertToArray(content[i])
            dataline = []
            for j in range(len(lineAsArray) - 1):
                dataline.append(lineAsArray[j])

            self.data.append(dataline)
            targetIndex = len(lineAsArray) - 1
            self.target.append(lineAsArray[targetIndex])

        # data and target are both arrays of arbitrary length.
        #
        # data is an array of arrays of integers (0 or 1) indicating state.
        #
        # target is an array of imtegers 0-3 indicating the action
        # taken in that state.
        

    # def main():
    #     self.registerInitialState(state)

    # Build naive model by using the good-move.txt
    def buildNBmodel(self):
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.target, test_size=0.20)
        X_train = np.array(X_train)
        y_train = np.array(y_train).reshape(-1, 1)
        # laplace smooth     k = 1  
        k = 1
        # Initialising the Prior and Likelihood
        Prior = np.empty(4)
        Likelihood = np.empty([4, 25])

        # calculate the prior probability
        direction, counts = np.unique(y_train, return_counts=True)
        # laplace smooth     
        Prior = (counts+k) / float(len(y_train)+4) 

        # calculate the likelihood
        data = np.concatenate((X_train, y_train), axis=1)
        for i in range(4):
            for j in range(25):
                newdata = data[data[:, -1] == i]
        		# laplace smooth     
                condition = float(newdata[:, j].sum()+k) / float(counts[i]+2)
                Likelihood[i, j] = condition

        self.Prior = Prior
        self.Likelihood = Likelihood

        return Prior, Likelihood

    # def main():
    #     self.buildNBmodel()

    # test Naive models by appling new sets of features and give a specific action
    def testNBmodel(self, features, Prior, Likelihood):

        # convert new features into newvalue
        newvalue = np.empty([4, 25])
        for i in range(len(features)):
            newvalue[:, i] = features[i] * Likelihood[:, i]
        newvalue[newvalue == 0] = 1

        # initialize likelihood and Posterior
        likelihood = np.ones(4)
        Posterior = np.empty(4)

        for i in range(4):
            for j in range(len(newvalue[0])):
                likelihood[i] = likelihood[i] * newvalue[i, j]
            Posterior[i] = likelihood[i] * Prior[i]

        action = np.argmax(Posterior)

        return action

    # def main():
    #     self.testNBmodel(features, self.Prior, self.Likelihood)

    # Tidy up when Pacman dies
    def final(self, state):

        print "I'm done!"

    # Turn the numbers from the feature set into actions:
    def convertNumberToMove(self, number):
        if number == 0:
            return Directions.NORTH
        elif number == 1:
            return Directions.EAST
        elif number == 2:
            return Directions.SOUTH
        elif number == 3:
            return Directions.WEST

    # Here we just run the classifier to decide what to do
    def getAction(self, state):

        # How we access the features.
        features = api.getFeatureVector(state)  # test
        Likelihood, Prior = self.buildNBmodel()

        actionum = self.testNBmodel(features, self.Prior, self.Likelihood)
        # convert actionum into direction
        action = self.convertNumberToMove(actionum)

        # Get the actions we can try.
        legal = api.legalActions(state)
        # getAction has to return a move. Here we pass "STOP" to the
        # API to ask Pacman to stay where they are. We need to pass
        # the set of legal moves to teh API so it can do some safety
        # checking.
        if action in legal:
            return api.makeMove(action, legal)
        else:
            return api.makeMove(random.choice(legal), legal)
    #
    # def main():
    #     self.getAction(state)
