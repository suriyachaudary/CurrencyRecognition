#include <stdio.h>
#include <iostream>
#include "global.h"
#include <fstream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/flann/flann.hpp"

using namespace cv;
using namespace std;



Mat extractSift(char *pathToTxtFile, const int numImagesToTrain)
{
    /**
    *  This function reads the first 'numImagesToTrain' images whose paths are given in text files. The image read is resized to a standard size.
    *  The keypoints are detected using SiftFeatureDetector( MAXIMAL_KEYPOINTS ) and described using SiftDescriptorExtractor. 
    *  A Mat of size NumberofFeatures x FeatureDimensions is returned. Using SIFT, this function returns a Mat of size NumberofFeatures x 128.
    *
    *  Parameter: #  pathToTxtFile - path to a directory which contains text files. Each text file corresponds to one class and contains path to images.
    *             #  numImagesToTrain - Maximal number of images to read from each txt files. First 'numImagesToTrain' many images are read.
    **/
    
    if(DISPLAY)
    {
        namedWindow("img", WINDOW_NORMAL );
    }

    Mat siftFeature;

    int imgCounter=0;
    
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
            return Mat();
        }
     
        while(imgCounter < numImagesToTrain && !feof(filePointer))
        {
            char pathToImage[500];
            fscanf(filePointer,"%s",pathToImage);
  
            Mat img = imread(pathToImage,0);
            printf("\nReading : %s",pathToImage);
  
            if(!img.data)
            {
                printf("\nCould not find %s",pathToImage);
                continue;
            }
  
            if(img.cols>WIDTH)
            {
                resize(img,img,Size((int)WIDTH,(int)(img.rows*WIDTH/img.cols)));
            }

            if(DISPLAY)
            {
                imshow("img", img);
                waitKey(1);
            }

            Mat descriptor;
            vector<KeyPoint> keypoints;
            
            detector.detect(img, keypoints);
            extract.compute(img, keypoints,descriptor);
            siftFeature.push_back(descriptor);
            imgCounter++;
        }
     
        imgCounter = 0;
        fclose(filePointer);
        printf("\n*******************************************************************");
    }
  
    return siftFeature;
}

Mat kMeansCluster(Mat &data,int clusterSize)
{ 
    /**
    *  implements k-means clustering algorithm. The function returns the cluster centers of type CV_8UC1.
    *  
    *  Parameters: # data- Each sample in rows. Type should be CV_32FC1
    *              # clusterSize- Number of clusters.
    **/

    TermCriteria termCriteria(CV_TERMCRIT_ITER,100,0.01);
    int nAttempts=3;
    int flags=KMEANS_PP_CENTERS;
  
    Mat clusterCenters, temp;
    kmeans(data, clusterSize, temp, termCriteria, nAttempts, flags, clusterCenters);
    clusterCenters.convertTo(clusterCenters,CV_8U);
    printf("here\n");
      
    return clusterCenters;
}

Mat hiKMeansCluster(Mat &data,int clusterSize)
{   
    /**
    *  implements Hierarchical k-means clustering algorithm. The function returns the cluster centers of type CV_8UC1.
    *
    *  Parameters: # data- Each sample in rows. Type should be CV_32FC1.
    *              # clusterSize- Number of clusters.
    **/
    
    Mat clusterCenters=Mat(clusterSize,data.cols,CV_32F);
    cvflann::KMeansIndexParams kParams = cvflann::KMeansIndexParams(2, 100*clusterSize, cvflann::FLANN_CENTERS_KMEANSPP,0.2);
    int numClusters =cv::flann::hierarchicalClustering<cvflann::L2<float> >(data, clusterCenters, kParams);
    clusterCenters = clusterCenters.rowRange(cv::Range(0,numClusters));
    clusterCenters.convertTo(clusterCenters,CV_8U);
  
    return clusterCenters;
}

void writeToYMLFile(Mat &dataToWrite, char *fileName)
{
    /**
    *  writes the data to the .yml file.
    *
    *  Parameters: # dataToWrite - matrix to write in .yml file.
    *              # fileName - name of yml file.
    **/
  
    char ymlFileName[100];
    sprintf(ymlFileName,"%s.yml",fileName);
    FileStorage fileStorage(ymlFileName, FileStorage::WRITE);
    
    fileStorage << fileName << dataToWrite;
    
    fileStorage.release();
}


void writeToBinaryFile(Mat &dataToWrite , char *fileName)
{
    /**
    *  writes the data to the .bin file .
    *  
    *  Parameters: # dataToWrite - matrix to write in binbary file. Type CV_8UC1.
    *              # fileName - name of binary file.
    **/
  
    fstream binaryFile(fileName,ios::binary|ios::out);
    if(!binaryFile.is_open())
    {
        printf("\nerror in opening: %s", fileName);
        return;
    }

    binaryFile.write((char *)dataToWrite.data, dataToWrite.rows*dataToWrite.cols) ;
    
    binaryFile.close();
}


Mat getBowHist(Mat &vocabulary,char *pathToTxtFile, const int numImagesToTrain)
{
    /**
    *  0. Each image is indexed (0-based) in same order as they are read from txt files.
    *  1. This function compute Bag of Words histogram for each image read from txt files. It returns a Mat of size NumberofImagesRead x VocabularySize and type CV_32FC1.
    *  2. The function also writes label of each file into labels.txt file. line i in labels.txt is label for ith image.
    *  3. It also writes location of keypoints of each image into ($imgIndex).txt . Each line in this file is a 3-tuple (vocab-id, x, y) where 
    *     keypoint at location (x,y) in image $imgIndex has been assigned to visual word 'vocab-id'.
    *  
    *  Parameters: # vocabulary- visual words. Size ClusterSizex128, type CV_8UC1.
    *              # pathToTxtFile - path to a directory which contains text files. Each text file corresponds to one class and contains path to images.
    *              # numImagesToTrain - Maximal number of images to read from each txt files. First 'numImagesToTrain' many images are read.
    **/

    if(DISPLAY)
    {
        namedWindow("img", WINDOW_NORMAL );
    }

    BOWImgDescriptorExtractor bowDE(new SiftDescriptorExtractor(),new FlannBasedMatcher());
    vocabulary.convertTo(vocabulary,CV_32F);                //convert vocabulary fron CV_8U to CV_32F
    bowDE.setVocabulary(vocabulary);
    
    Mat allHist;
    
    int imgCounter = 0;
    
    int imgIndex=0;
  
    FILE *labelFilePointer=fopen("labels.txt","w");
  
    if(labelFilePointer==NULL)
    {
        printf("\nERROR..!! Couldn't open 'Labels.txt'");
        return Mat();
    }
  
    for(int i=0;i<6;i++)
    { 

        char txtFileName[200];
        sprintf(txtFileName,"%s/%s.txt",pathToTxtFile,txtFiles[i]);
        FILE *filePointer = fopen(txtFileName, "r");
        printf("\n*******************************************************************");
        printf("\nReading %s", txtFileName);
     
        if(filePointer==NULL)
        {
            printf("\nERROR..!! File '%s' not found", txtFileName);
            return Mat();
        }
     
        while(imgCounter<numImagesToTrain && !feof(filePointer))
        { 

            fprintf(labelFilePointer,"%d\n",i);
            char pathToImage[500];
            fscanf(filePointer,"%s",pathToImage);//read the pathToImage of ith file  
       
            Mat img = imread(pathToImage,0);
            printf("\nReading : %s",pathToImage);
  
            if(!img.data)
            {
                printf("\nCould not find %s",pathToImage);
                continue;
            }
  
            if(img.cols>WIDTH)
            {
                 resize(img,img,Size((int)WIDTH,(int)(img.rows*WIDTH/img.cols)));
            }

            if(DISPLAY)
            {
                imshow("img", img);
                waitKey(1);
            }

            vector<KeyPoint> keypoints;
            vector<vector<int> > pointIdxsOfClusters;   
            Mat imgHistogram;
    
            detector.detect(img, keypoints);
            bowDE.compute(img, keypoints,imgHistogram,&pointIdxsOfClusters);    
    
            char keypointFileName[200];
            sprintf(keypointFileName,"%s/%d.txt", KEYPOINTS_DIRECTORY,imgIndex);

            FILE *keypointFile=fopen(keypointFileName,"w");
        
            for(int k=0;k<pointIdxsOfClusters.size();k++)
            {
                for(int j=0;j<pointIdxsOfClusters[k].size();j++)
                {
                    fprintf(keypointFile,"%d %d %d\n",k, (int)keypoints[pointIdxsOfClusters[k][j]].pt.x, (int)keypoints[pointIdxsOfClusters[k][j]].pt.y);
                }
            }
            
            fclose(keypointFile);
               
            imgIndex++;
            imgCounter++;
            allHist.push_back(imgHistogram);
        }
        
        imgCounter=0;
        fclose(filePointer);
        printf("\n*******************************************************************");
    }

    fclose(labelFilePointer);     
    return allHist; 
}

Mat tfIdfWeighting(Mat &allHist)
{
    /**
    *  This function perform 'term frequency-inverse document frequency' (tf-idf) weighting.
    *  It returns a Mat of size allHist.rows x allHist.cols and type CV_32FC1.
    *
    *  Parameter: # allHist- contains the histogram of all images. Type CV_32FC1.
    **/
  
    Mat weightedAllHist= Mat::zeros(allHist.rows, allHist.cols, CV_32F);

    int *numImagesInDb = new int[allHist.cols];
    for(int j=0;j<allHist.cols;j++)
    {
        numImagesInDb[j]=0;
    }
  
    for(int i=0;i<allHist.rows;i++) 
    {
        for(int j=0;j<allHist.cols;j++)
        {
            if(allHist.at<float>(i,j)>0) 
            {
                numImagesInDb[j]=numImagesInDb[j] + 1;
            }
        }
    }  
  
    for(int i=0;i<allHist.rows;i++)
    { 
        for(int j=0;j<allHist.cols;j++)
        {
            if(numImagesInDb[j] > 0)
            { 
                weightedAllHist.at<float>(i,j)=allHist.at<float>(i,j)*log(((float)(allHist.rows))/numImagesInDb[j]);
            } 
            else
            {
                weightedAllHist.at<float>(i,j)=allHist.at<float>(i,j);
            }
        }
    } 
  
  delete[] numImagesInDb;
  
  return   weightedAllHist;
}

vector<invertedIndex> getInvertedIndex(Mat weightedAllHist)
{
    /**
    *  Create an inverted index from weightedAllHist. Each element in invertedIndex corresponds to a visual word.
    *  Each invertedIndex i contains variable number of 2-tuple (imgIndex, tf-idf value) where imgIndex and tf-idf value is weightedAllHist[imgIndex][i].  
    *  This is a compact representation of the weighted BOW histogram.
    *  It returns a vector of structure of type invertedIndex.
    *  The function also write size of each invertedIndex into a text file indicesSize.txt
    *  
    *  Parameter: # weightedAllHist- tf-idf weighted BOW histogram of each training image.
    **/
  
    vector<invertedIndex> allIndex;
    FILE *indicesSizeFilePointer=fopen("indicesSize.txt","w");
  
    for(int i=0;i<weightedAllHist.cols;i++)
    {
        invertedIndex tempIndex;
        for(int j=0;j<weightedAllHist.rows;j++)
        {
            if(weightedAllHist.at<float>(j,i)>0)
            { 
                tempIndex.imgIndex.push_back(j);
                tempIndex.weightedHistValue.push_back(weightedAllHist.at<float>(j,i));
            }
        } 
    
        allIndex.push_back(tempIndex);
    
        fprintf(indicesSizeFilePointer,"%d\n",tempIndex.imgIndex.size() );
    }
  
    fclose(indicesSizeFilePointer);
  
    return allIndex;    
}

void writeToBinaryFile(vector<invertedIndex> allIndex , char *fileName)
{
    /**
    *  writes the inverted index into binary file
    *
    *  Parameter: * allIndex- vector of invertedIndex type to write in .bin file.
    *             * fileToWrite- name of file to which vector is to write in .bin file  
    **/

    fstream binaryFile(fileName, ios::out | ios::binary);
    if(!binaryFile.is_open())
    {
        printf("\nerror in opening: %s", fileName);
        return;
    }

    for(int i=0;i<allIndex.size();i++)
    {
     
        for(int j=0;j<allIndex[i].imgIndex.size();j++)
        {
            int a = allIndex[i].imgIndex[j];
            float b = allIndex[i].weightedHistValue[j];
            binaryFile.write((char *)&a,4) ;
            binaryFile.write((char *)&b,4) ;
        }
    }
  
  binaryFile.close();
}

