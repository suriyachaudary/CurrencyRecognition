This code is implementation of the paper : 
Suriya Singh, Shushman Choudhury, Kumar Vishal and C.V. Jawahar. "Currency Recognition on Mobile Phones". Proceedings of the 22nd International Conference on Pattern Recognition, 24-28 Aug 2014, Stockholm, Sweden. 

Project Page :
http://researchweb.iiit.ac.in/~suriya.singh/Currency2014ICPR/Currency2014ICPR.html


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

--------------------------------------------------------------------------

1. Training code - In directory 'currency_train' there are 4 files
    (a) currency.cpp - implementation training part of Currency Recognition on Mobile Phones paper.
    (b) train_BOW_IR_utils.cpp - utility functions for performing Bag of Word retrieval.
    (c) global.h - global variables and parameters are defined here.
    (d) run.sh - bash script that
                 (0) set vocabulary size and number of training images to be used for training
                 (i) scan training directory and create a class-wise list of training images. 
                 (ii) compile and run the code.
                 (iii) place all necessary files into directory 'currency_train_output' and
                 (iv) remove temporary files.

'directory conatining training images' should have directory structure
base_directory
+ ------ ten/
+ ------ twenty/
+ ------ fifty/
+ ------ hundred/
+ ------ fivehundred/
+ ------ thousand/


How to run the code - 
$ sh run.sh 'path to directory conatining training images'

e.g.
$ sh run.sh /home/bond/currency_dataset/train/

Once the code finishes, it create a directory 'currency_train_output' with following files and directory
'currency_train_output'
+ ------ keypoints/
  ------ vocabulary.bin
  ------ allIndex.bin
  ------ labels.txt
  ------ indicesSize.txt
  ------ dataFile.txt

--------------------------------------------------------------------------

2. Test code - In directory 'currency_IR' there are 3 files
    (a) testCurrency.cpp - implements the retrieval and recognition part of Currency Recognition on Mobile Phones paper.
    (b) testCurrency_BOW_IR_utils.cpp - utility functions for performing Bag of Word retrieval.
    (c) run.sh - bash script that 
                 (i) scan training directory and create a class-wise list of test images.
                 (ii) compile and run the code.
                 (iii) remove temporary files.

NOTE - the script is used for predicting the labels in batch.

'directory conatining test images' should have directory structure
base_directory
+ ------ ten/
+ ------ twenty/
+ ------ fifty/
+ ------ hundred/
+ ------ fivehundred/
+ ------ thousand/

How to run the code - 
$ sh run.sh 'path to directory containing training output' 'path to directory conatining test images'

e.g.
$ sh run.sh ../currency_train/currency_train_output/ /home/bond/currency_dataset/test_images/

Once the code finishes, it output the results like classification accuracy and average time.

NOTE - In order to predict label one by onem the user need to make changes in test code. Call testCurrency() function with path to image as argument. Compile and run to get the predicted label.

--------------------------------------------------------------------------

To manually compile the train code

$ g++ `pkg-config --cflags --libs opencv` currency.cpp -o currency-train -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_objdetect -lopencv_nonfree -lopencv_features2d -lopencv_flann;

To manually run the train code, the user has to create  classwise text files which contains list of training images. And run by using
$ ./currency-train train_files/ $vocabSize $numImagesToTrain

where train_files/ is path to text files which contains list of training images.


To manually compile the test code

g++ `pkg-config --cflags --libs opencv` testCurrency.cpp -o currency-test -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_objdetect -lopencv_nonfree -lopencv_features2d -lopencv_flann -lopencv_calib3d;

To manually run the train code, the user has to create  classwise text files which contains list of training images. And run by using
$ ./currency-test PATH/dataFile.txt PATH/vocabulary.bin PATH/labels.txt PATH/indicesSize.txt PATH/allIndex.bin $topKValue PATH/keypoints/ train_files/

where train_files/ is path to text files which contains list of training images and PATH is path to 'currency_train_output' or path to directory containing training output.