#include <stdio.h>
#include <iostream>
#include <fstream>
#include "../currency_train/global.h"
#include "opencv2/opencv.hpp"
#include "opencv2/nonfree/features2d.hpp"

using namespace cv;
using namespace std;


BOWImgDescriptorExtractor bowDE(new SiftDescriptorExtractor(),new FlannBasedMatcher());


Mat readVocab(char* vocabularyBinaryFile, const int vocabSize)
{ 
    /**
    *  Read vocabulary from file and return a Mat of type CV_8UC1.
    *
    *  Parameters: # vocabularyBinaryFile - path to vocabulary file.
    *              # vocabSize - Size of vocanulary.
    **/
  
    unsigned  char array[vocabSize][128];
    fstream binaryFile(vocabularyBinaryFile,ios::binary|ios::in);
  
    if(!binaryFile.is_open())
    {
        printf("\nERROR..!! Couldn't open vocabulary file");
        return Mat();
    }
  
    binaryFile.read((char *)array,sizeof(array)) ;//read binary file
    Mat vocabulary = Mat(vocabSize, 128, CV_8UC1, &array );
    binaryFile.close();

    return vocabulary;
}

void setVocabulary(Mat &vocabulary)
{
    vocabulary.convertTo(vocabulary,CV_32F);
    bowDE.setVocabulary(vocabulary);  
}

void readLabels(char *labelFile, const int numOfTrainImg, int labels[])
{
    /**
    *  Read annotations from annotation file and store it in labels array.
    *
    *  Parameters: # labelFile - path to the annotation file.
    *              # numOfTrainImg- number of images used for training.
    *              # labels[] - annotations are stored in this integer array. 
    **/
 
    FILE *filePointer=fopen(labelFile,"r");
  
    if(filePointer==NULL)
    {
        printf("\nERROR..!! Couldn't opencv annotations file");
        return;
    }
  
    for(int i=0; i<numOfTrainImg; i++)
    {
        fscanf(filePointer,"%d",&labels[i]);
    }

    fclose(filePointer);
}

void readSize(char *indicesSizeFile,const int vocabSize,int indicesSize[])
{
    /** 
    *  Read size of each inverted index from file and store in indicesSize integer array
    *
    *  Parameters: # indicesSizeFile- path to file that contains size of each inverted index.
    *              # indicesSize[] - Array in which size of inverted index is stored.
    **/
  
    FILE *filePointer =fopen(indicesSizeFile, "r");
      
    if(filePointer==NULL)
    {
        printf("\nERROR..!! Couldn't open indicesSize file");
        return;
    }
  
    for(int i=0;i<vocabSize;i++)
    {
        fscanf(filePointer,"%d",&indicesSize[i]);
    }

    fclose(filePointer);
}

vector<invertedIndex> readInvertedIndex(char *allIndexFile,int indicesSize[], const int vocabSize)
{
    /** 
    *  Read inverted index from binary file and return vector of inverted index.
    *  Each inverted index i contains imageIndex and tf-idf score of images that contain visual word i.
    *
    *  Parameter: # allIndexFile - path to inverted index file.
    *             # indicesSize[] - Array in which size of each inverted index is stored.   
    *             # vocabSize - size of vocabulary.
    **/   
  
    vector<invertedIndex> allIndex;  
    fstream binaryFile(allIndexFile,  ios::binary|ios::in);
  
    for(int i=0;i< vocabSize;i++)
    {
        invertedIndex temp;
        for(int j=0;j<indicesSize[i];j++)
        {
            int imgIndex;
            float weightedHistValue;
            binaryFile.read((char *)&imgIndex, sizeof(imgIndex)) ;
            binaryFile.read((char *)&weightedHistValue, sizeof(weightedHistValue)) ;
            temp.imgIndex.push_back(imgIndex);
            temp.weightedHistValue.push_back(weightedHistValue);
        }

       allIndex.push_back(temp);	
    }

    binaryFile.close();

    return allIndex; 
}

Mat grabcutSegmentation(Mat &input)
{
    /**
    *  implements the grabcut algorithm. It takes BGR image as an argument and it return binary mask. 
    *
    *  Parameter: # input- image to be segmented.
    **/
  
    Mat img;
    resize(input,img,Size(90,input.rows*(90.0/input.cols)));
    int count=0;
    int thresh=0.9*img.rows*img.cols;
    Rect rectangle(5,5,img.cols-10,img.rows-10);
    Mat mask; 
    Mat bgModel,fgModel; 
    bgModel=Mat(), fgModel=Mat();//initialize  mat
      
    grabCut(img,mask,rectangle,bgModel,fgModel,1,GC_INIT_WITH_RECT); //perform grabcut algorithm
      
    for(int i=0;i<mask.rows;i++)
    {
        for(int j=0;j<mask.cols;j++)
        {
            if(mask.at<uchar>(i,j)==0||mask.at<uchar>(i,j)==2)
            {
                mask.at<uchar>(i,j)=2;
                count++;
            }
        }
    }
    
    if(count>0 && count<thresh) 
    {
        grabCut(img, mask,rectangle,bgModel,fgModel,1,GC_INIT_WITH_MASK );
    }

    else
    {
        for(int i=0.2*mask.rows;i<0.8*mask.rows;i++)
        {
            for(int j=0.2*mask.cols;j<0.8*mask.cols;j++)
            {
                mask.at<uchar>(i,j)=3;
            }
        }
    }
    
    cv::compare(mask,cv::GC_PR_FGD,mask,cv::CMP_EQ);
    resize(mask,mask,input.size(),0,0,CV_INTER_LINEAR); 
    if(DISPLAY)
    {
        Mat out;
        input.copyTo(out, mask);
        imshow("img", out) ;
        waitKey(1);    
    }
    return mask;
}

vector<KeyPoint> removeKeyPoints(Mat mask,vector<KeyPoint> keyPoints)
{
    /**
    *  remove the keypoints which takes the mask value greater than zero or corresponding foreground pixels. 
    *
    *  Parameters: * mask- 8-bit single-channel mask.
    *              * keyPoints-interesting points on the object can be extracted to provide a "feature description" of the object
    **/
  
    vector<KeyPoint> tempKeyPoints;
    for(int i=0;i<keyPoints.size();i++)
    {
        if(mask.at<unsigned char>(keyPoints[i].pt)>0)
            tempKeyPoints.push_back(keyPoints[i]);
    }  
  
    return tempKeyPoints;  
}

void getDotProduct(vector<invertedIndex> &allIndex, Mat &imgHistogram, const int numOfTrainImg, float dotProduct[])
{
    /** 
    *  performs dot product between histogram of test image and all training images
    *
    *  Parameters: # allIndex- vector of inverted index.
    *              # numOfTrainImg- Total number of images trained.
    *              # dotProduct[]- Array in which dotProduct is stored.
    **/ 
    
    for(int i=0;i<numOfTrainImg;i++)
        dotProduct[i]=0;
    
    for(int i=0;i<imgHistogram.cols;i++)
    {
        for (int j=0;j<allIndex[i].imgIndex.size();j++)
        {
           dotProduct[allIndex[i].imgIndex[j]] +=  imgHistogram.at<float>(0,i) * allIndex[i].weightedHistValue[j];
        }
    }  
}

int argmax(float array[], const int arraySize)
{
    /** 
    *   implement argmax
    *
    **/   
  
    int k = 0;
    float max = array[k];
    for(int i = 0; i < arraySize; i++)
    {
        if(array[i] > max) 
        {
            max = array[i];
            k = i;
        }
    }
  
    return k;
}

void retrieveTopKImages(float dotProduct[],const int topKValue,const int numOfTrainImg,int indices[])
{
    /** 
    *  retreives top K image indices which corresponds to largest dotProduct value
    *
    *  Parameters: # dotProduct[]- dotProduct value of each train image BOW histogram with that of test image.
    *              # topKValue- number of images to be retrieved.
    *              # numOfTrainImg- number of images used for training.
    *              # indices[]- indices of K training images which corresponds to largest dotProduct value
    **/   
  
     for(int i=0;i<topKValue;i++)
        indices[i]=-1;   
    
    for(int i=0;i<topKValue;i++)
    { 
        indices[i] = argmax(dotProduct, numOfTrainImg);
        dotProduct[indices[i]]=-1;
    }   
}

void rerankUsingGeometryVerification(int retrievedImages[],const int topKValue,vector<vector<int> > &pointIdxsOfClusters,vector<KeyPoint> &keypoints,char  *keyPointsPath,int geoScore[])
{
    /** 
    *  Rerank the retrieved images based on geometric verification. Geometric verification is carried out by fitting fundamental matrix
    *  using keypoints that have been assigned to same visual word as correspondences.
    *  The keypoint location of of retrieved images are read from text files.
    *  
    *  Parameters: * retrievedImages[]- indices of images corrsponding to highest K dotProduct values.
    *              * topKValue- number of images retrieve\d for geometric verification.
    *              * pointIdxsOfClusters-  vocabulary assignment details for each keypoints of test images.
    *              * keypoints- keypoints of test images.
    *              * keyPointsPath- path to directory that contains keypoint location of training images.
    *              * geoScore[]- array to which calculated score is stored. This score is number of keypoints in retreived images that are geometrically consistent with test image.
    **/   
  
    for(int i=0; i<topKValue; i++)
    { 
        geoScore[i]=0;
        Mat pointInTestImg, pointInRetreivedImg;
        char keyPointsFile[200];
        sprintf(keyPointsFile,"%s/%d.txt",keyPointsPath,retrievedImages[i]);

        FILE *keyPointsFilePointer;
        keyPointsFilePointer= fopen(keyPointsFile,"r");
        int vocabularyId=-1,x=-1,y=-1;
        
        if(keyPointsFilePointer==NULL)
        {
            printf("\nFile %s not found", keyPointsFile);
            break;
        }
    
        for(int j=0;j<pointIdxsOfClusters.size();j++)
        {    
            fseek ( keyPointsFilePointer , 0 , SEEK_SET );
            for(int k=0;k<pointIdxsOfClusters[j].size();k++)
            {
                if(feof(keyPointsFilePointer))
	            {
	                break;
	            }
	           Mat temp=Mat(1,2,CV_32F);
               temp.at<float>(0,0)=keypoints[pointIdxsOfClusters[j][k]].pt.x;
               temp.at<float>(0,1)=keypoints[pointIdxsOfClusters[j][k]].pt.y;
               while(!feof(keyPointsFilePointer))
               {
                    fscanf(keyPointsFilePointer,"%d\t%d\t%d",&vocabularyId,&x,&y); 
          
                    if(vocabularyId>j)
                    {
                        break;
                    }   
          
                    if(vocabularyId==j)
                    {
                        Mat temp2=Mat(1,2,CV_32F);
                        temp2.at<float>(0,0)=x;
                        temp2.at<float>(0,1)=y;
                        pointInTestImg.push_back(temp);
                        pointInRetreivedImg.push_back(temp2);
                        // break;
                    }
                }
            }
        }
        fclose(keyPointsFilePointer);
   
        if(pointInTestImg.rows>=8)
        {
            Mat out;
            findFundamentalMat(pointInTestImg,pointInRetreivedImg,FM_RANSAC,3.0,0.99, out );
            for(int j=0;j<out.rows;j++)
            {
                geoScore[i] = geoScore[i] +out.at<uchar>(j,0);//calculate score
            }
        }
    }  
}

void getVote(int geoScore[], int topKValue, int labels[], int retrievedImages[], int vote[])
{
    /** 
    *  calculates number of votes for each class using score from geometric verification of each retrieved image
    *  
    *  Parameters: # geoScore[]- score from geometric verification of each retrieved image.
    *              # topKValue- number of images to retrieve for geo-metric verification.
    *              # labels[]- annotations of training images
    *              # retrievedImages[]- indices of images corrsponding to highest K dotProduct values.
    *              # vote- vote for each class.
    **/
 
    for(int i=0;i<6;i++)
        vote[i]=0;
   
    for(int i=0;i<topKValue;i++)
    {
        vote[labels[retrievedImages[i]]] = vote[labels[retrievedImages[i]]] + geoScore[i];
    }
}
 
int argmax(int array[])
{
    /** 
    *   implement argmax
    *
    **/ 
  
    int k = 0;
    int max = array[k];
    for (int i = 0; i < 6; i++)
    {
        if (array[i] > max)
        {
            max = (int)array[i];
            k = i;
        }
    }
  
    return k;
}

int testCurrency(char *pathToTestImg, vector<invertedIndex> &allIndex, int labels[], char *keyPointsPath, const int numOfTrainImg, const int topKValue)
{
    /** 
    *   This function detects keypoints, remove irrelevant keypoints, then vocabulary assignment to each keypoint and compute BOW histogram for the image.
    *   Following that it perform inverted index search and then geometrically verify the top K images.
    *   Finally, voting is done and predicted label is returned.
    *   
    *   Parameter: # pathToTestImg - path to the image.
    *              # allIndex- vecotr of inverted index.
    *              # labels[]- annotations of training images
    *              # keyPointsPath- path to directory containing the keypoints location of each image.
    *              # numOfTrainImg- number of images used for training.
    *              # topKValue- number of images to retrieve for geo-metric verification.
    **/
  
    Mat img = imread(pathToTestImg,1);
    printf("\nReading : %s",pathToTestImg);
  
    if(!img.data)
    {
        printf("\nCould not find %s",pathToTestImg);
        return -3;
    }
  
    if(img.cols>TEST_WIDTH)
    {
        resize(img,img,Size((int)TEST_WIDTH,(int)(img.rows*TEST_WIDTH/img.cols)));
    }

    if(DISPLAY)
    {
        imshow("img", img);
        waitKey(1);
    }

    
  
    /******************** grabcut segmentation ********************/
    Mat mask= grabcutSegmentation(img);
  
    /******************** detect keypoints ********************/
    cvtColor(img, img, CV_BGR2GRAY);
    int clo = clock();
    vector<KeyPoint> keypoints;
    detector.detect(img, keypoints);
    if(keypoints.size() < 100)
    {
        return -2;
    }

    printf("\nkeypoint Time = %f", (float)(clock()-clo)/CLOCKS_PER_SEC);   

    printf("\n Keypoints detected before removing keypoints : %d", keypoints.size() );

    vector<KeyPoint> keypoints_removed = removeKeyPoints(mask, keypoints);
    if(keypoints_removed.size() < 100)
    {
        swap(keypoints_removed, keypoints);
    }

    printf("\n Keypoints detected after removing keypoints : %d",keypoints_removed.size() );
    
    Mat imgHistogram;
    vector<vector<int> > pointIdxsOfClusters;
    
    clo = clock();
    bowDE.compute(img, keypoints_removed, imgHistogram, &pointIdxsOfClusters); //Computes an image BOW histogram.
    printf("\nvocab assignment Time = %f", (float)(clock()-clo)/CLOCKS_PER_SEC);   
    
    /******************** calculate dot product *********************/ 
    clo = clock();
    float *dotProduct = new float[numOfTrainImg];
    getDotProduct(allIndex, imgHistogram, numOfTrainImg, dotProduct);
    printf("\ndot Time = %f", (float)(clock()-clo)/CLOCKS_PER_SEC);   
    
    /******************** retrieve top K images *********************/
    clo = clock();
    int *retrievedImages = new int[topKValue];
    retrieveTopKImages(dotProduct, topKValue, numOfTrainImg, retrievedImages);
    
    printf("\narg max Time = %f", (float)(clock()-clo)/CLOCKS_PER_SEC);   

    for(int i=0;i<10;i++)
        printf("\nIndices=%d", retrievedImages[i]);
  
    /******************* Rerank using geometric verification **********************/
    clo = clock();
    int *geoScore = new int[topKValue];
    rerankUsingGeometryVerification(retrievedImages,topKValue, pointIdxsOfClusters, keypoints_removed, keyPointsPath, geoScore);
    printf("\ngeo Time = %f", (float)(clock()-clo)/CLOCKS_PER_SEC);   

    for(int i=0;i<topKValue;i++)
        printf("\nScore=%d", geoScore[i]);
 
    /******************** vote *********************/
    int vote[6]={0};
    getVote(geoScore, topKValue, labels, retrievedImages, vote);
    for(int i=0;i<6;i++)
        printf("\n vote[%d]=%d",i,vote[i]);
   
    /******************** label *********************/
    int label= argmax(vote);
    
    delete[] dotProduct;
    delete[] retrievedImages;
    delete[] geoScore;
    
    return label;
}

void readFiles(char *pathToTxtFile, vector<invertedIndex> &allIndex, int labels[], char *keyPointsPath, const int numOfTrainImg, const int topKValue)
{
    /** 
    *   read image and call testCurrency function to identify the denomination of thr bill in the image.
    *   Also calculate accuracy and other statistics.
    *
    *   Parameter: # pathToTxtFile- path to directory containing the text files
    *              # allIndex- vecotr of inverted index.
    *              # labels[]- annotations of training images
    *              # keyPointsPath- path to directory containing the keypoints location of each image.
    *              # numOfTrainImg- number of images used for training.
    *              # topKValue- number of images to retrieve for geo-metric verification.
    **/
    if(DISPLAY)
    {
        namedWindow("img", WINDOW_NORMAL );
    }

    int imgCounter=0;
    int numCorrect=0;

    for(int i=0;i<6;i++)
    {
        char txtFileName[200];
        sprintf(txtFileName,"%s/%s.txt", pathToTxtFile, txtFiles[i]);
        FILE *filePointer = fopen(txtFileName,"r");
        printf("\n*******************************************************************");
        printf("\nReading %s", txtFileName);
     
        if(filePointer==NULL)
        {
            printf("\nERROR..!! File '%s' not found", txtFileName);
            break;
        }
     
        while(!feof(filePointer))
        {
            char pathToImage[500];
            fscanf(filePointer,"%s",pathToImage);

            int clo = clock();
            int label = testCurrency(pathToImage, allIndex, labels, keyPointsPath, numOfTrainImg, topKValue);
            printf("\nLabel = %d", label);
            printf("\nTime = %f", (float)(clock()-clo)/CLOCKS_PER_SEC);
            if(label==i)
                numCorrect++;

            imgCounter++;
            printf("\nAccuracy = %f", (float)numCorrect/imgCounter);
        }
    }     

    printf("\nTotal images read : %d", imgCounter);
    printf("\nTotal correct predicted : %d", numCorrect);
    printf("\nAccuracy = %f", (float)numCorrect/imgCounter);

}
