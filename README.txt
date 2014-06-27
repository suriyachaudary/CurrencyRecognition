The code is divided into 2 parts
    
1. Training code - This code can be used to obtain necessary files for instance retrieval :
    (a) vocabulary and vocabulary size used
    (b) inverted index and size of each entry in inverted index
    (c) annotation for each training image and number of training images used
    (d) location and vocabulary assignment information for keypoints in each training image

2. Test code - This code perform
    (a) read files obtained from training step into memory.
    (b) predict labels of images.

(provided script assume JPEG image. For other type of image the user is required to properly modify the scripy 'run.sh' for training as well as test code.)

-----------------------------------------------------------------------------------------------------------------------------------

1. Training code - In directory 'currency_train' there are 4 files
    (a) currency.cpp - implementation training part of Currency Recognition on Mobile Phones paper.
    (b) train_BOW_IR_utils.cpp - utility functions for performing Bag of Word retrieval.
    (c) global.h - global variables and parameters are defined here.
    (d) run.sh - bash script that
                 (0) set vocabulary size and number of training images to be used for training
                 (i) scan training directory and create a class-wise list of training images. 
                 (ii) compile and run the code. (iii) place all necessary files into directory 'currency_train_output' and
                 (iv) remove temporary files.

'training directory' should have directory structure
base_directory
+ ------ ten/
+ ------ twenty/
+ ------ fifty/
+ ------ hundred/
+ ------ fivehundred/
+ ------ thousand/


How to run the code - 
$ sh run.sh 'path to training directory'

e.g.
$ sh run.sh /home/bond/currency_dataset/train/

Once the code finishes, it create a directory 'currency_train_output' with following files and directory
'currency_train_output'
-------- keypoints/
-------- vocabulary.bin
-------- allIndex.bin
-------- labels.txt
-------- indicesSize.txt
-------- dataFile.txt


2. Test code - In directory 'currency_IR' there are 3 files
    (a) testCurrency.cpp - implements the retrieval and recognition part of Currency Recognition on Mobile Phones paper.
    (b) testCurrency_BOW_IR_utils.cpp - utility functions for performing Bag of Word retrieval.
    (c) run.sh - bash script that (i) scan training directory and create a class-wise list of test images. (ii) compile and run the code. (iii)                  remove temporary files. NOTE - the script is used for predicting the labels in batch.

