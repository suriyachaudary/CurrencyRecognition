#!/bin/bash 
# This script compile and run currency-train program. This script creates temporary text files containing path to images for each class.
# The script expects path to directory containing training set. Images belonging to same class should be in same directory in the training set.

vocabSize=100;
numImagesToTrain=4;
mkdir train_files;
for f in $1/ten/*.jpg; do echo $f;done > train_files/ten.txt
for f in $1/twenty/*.jpg; do echo $f;done > train_files/twenty.txt
for f in $1/fifty/*.jpg; do echo $f;done > train_files/fifty.txt
for f in $1/hundred/*.jpg; do echo $f;done > train_files/hundred.txt
for f in $1/fivehundred/*.jpg; do echo $f;done > train_files/fivehundred.txt
for f in $1/thousand/*.jpg; do echo $f;done > train_files/thousand.txt

g++ -std=c++0x `pkg-config --cflags --libs opencv` currency.cpp -o currency-train -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_objdetect -lopencv_nonfree -lopencv_features2d -lopencv_flann;
./currency-train train_files/ $vocabSize $numImagesToTrain

rm -r train_files;