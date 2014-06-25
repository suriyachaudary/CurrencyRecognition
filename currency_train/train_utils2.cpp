#include <stdio.h>
#include <iostream>
#include <fstream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/flann/flann.hpp"

using namespace cv;
using namespace std;

SiftFeatureDetector detector( 500 );
SiftDescriptorExtractor extract;

#define IMAGES_TO_TRAIN 400
#define KEYPOINTS_DIRECTORY "keypoints/"

BOWImgDescriptorExtractor bowDE(new SiftDescriptorExtractor(),new FlannBasedMatcher());

struct invertedIndex
{
  vector<int> imgIndex;
  vector<float> weightedHistValue;
};

Mat extractSift(char *pathToFile)
{
  /**
  *  reads the full path of every image in  .txt file of all labels one by one. It detects keypoints in the 
  *  image using detector.detect() function and using extractor.compute() function descriptor is calculated. 
  *  Descriptor is stored in siftFeature.
  *
  *  Parameter: *  pathToFile - path to file containing full path of the images.
  **/
  
  Mat siftFeature,descriptor;
  vector<KeyPoint> keypoints;
  char fileName[200];
  char pathToImage[200];
  int limit=0;
  FILE *filePointer;
  char files[6][30]={"ten","twenty","fifty","hundred","fivehundred","thousand"};
  
  for(int i=0;i<6;i++)
  {	
     sprintf(fileName,"%s/%s.txt",pathToFile,files[i]);
     filePointer=fopen(fileName,"r");
     printf("\n file %d :%s",i,fileName);
     
     if(filePointer==NULL)
     {
       printf("File %s not found\n",files[i]);
       continue;
     }
     
     while(limit<IMAGES_TO_TRAIN)
     {
        descriptor=Mat();
	fscanf(filePointer,"%s",pathToImage);
	
	if(feof(filePointer))
	{
	 break;
	}
	Mat img=imread(pathToImage,0);
	printf("\n Reading the image %s",pathToImage);
	
	if(!img.data)
	{
           printf("\n could not find the image in the path %s",pathToImage);
	   continue;
	}
	
	if(img.cols>480)
	{
	   resize(img,img,Size(480,img.rows*(480.0/(int)img.cols)));
	}
	detector.detect(img, keypoints);
	extract.compute(img, keypoints,descriptor);
	siftFeature.push_back(descriptor);
	limit++;
     }
     limit=0;
     fclose(filePointer);
     
  }
  
  return siftFeature;
  
}

Mat kMeansCluster(Mat data,int clusterSize)
{ 
  /**
  *  implements k-means clustering algorithm.
  *  
  *  Parameters: * data- takes data of type Mat
  *              * clusterSize- number of clusters used.
  **/
  
  Mat bestLabels,clusterCenters;
  TermCriteria termCriteria(CV_TERMCRIT_ITER,100,0.01);
  int nAttempts=1;
  int flags=KMEANS_PP_CENTERS;
  kmeans(data, clusterSize,bestLabels,termCriteria,nAttempts,flags,clusterCenters);
  clusterCenters.convertTo(clusterCenters,CV_8U);
  
  return clusterCenters;
  
}

/*Mat hiKMeansCluster(Mat data,int clusterSize)
{   
  /**
  *  implements Hierarchical k-means clustering algorithm.
  *
  *  Parameters: * data- takes data of type Mat
  *              * clusterSize- number of clusters used.
  *
    
  Mat clusterCenters=Mat(clusterSize,data.cols,CV_32F);
  cvflann::KMeansIndexParams kParams = cvflann::KMeansIndexParams(2,10000, cvflann::FLANN_CENTERS_KMEANSPP,0.2);
  int numClusters =cv::flann::hierarchicalClustering<cvflann::L2<float> >(data, clusterCenters, kParams);
  clusterCenters = clusterCenters.rowRange(cv::Range(0,numClusters));
  clusterCenters.convertTo(clusterCenters,CV_8U);
  printf("\nCluster centers : %d",numClusters);
  
  return clusterCenters;
  
}*/

void writeToFile(Mat matrix , char *fileToWrite)
{
  /**
  *  writes the data to the .yml file.
  *
  *  Parameters: * matrix- matrix to write in .yml file
  *              * fileToWrite- name of file to which matrix is to write in .yml file
  **/
  
  char ymlFileName[100];
  sprintf(ymlFileName,"%s.yml",fileToWrite);
  FileStorage fileStorage(ymlFileName, FileStorage::WRITE);
  fileStorage << fileToWrite << matrix;
  fileStorage.release();
   
}
void writeToBinaryFile(Mat matrix , char *fileToWrite)
{
  /**
  *  writes the data to the .bin file .
  *  
  *  Parameters: * Matrix- matrix to write in .bin file
  *              * fileToWrite- name of file to which matrix is to write in .bin file  
  **/
  
  unsigned char array[matrix.rows][matrix.cols];
  
  for(int i=0;i<matrix.rows;i++)
  {
    for(int j=0;j<matrix.cols;j++)
    {
      array[i][j]= matrix.at<uchar>(i,j);
    }
  }      
  char binaryFileName[200];
  sprintf(binaryFileName,"%s.bin",fileToWrite);
  fstream binaryFile(binaryFileName,ios::binary|ios::out);
  if(!binaryFile.is_open())
  {
    printf("\n error in opening %s",binaryFileName);
  }
  binaryFile.write((char *)array,sizeof(array)) ;
  binaryFile.close();
  
}

Mat getBowHist(Mat vocabulary,char *pathToFile)
{
  /**
  *  reads the full path of every image in .txt file of all labels one by one. It detects keypoints 
  *  in the image using detector.detect() function and it computes imgHistogram and pointIdxsOfClusters using 
  *  bowDE.compute().If pointIdxsOfCluster.size() is greater than zero then it opens the file named numOfImgCount and 
  *  it writes clusterId(which is i), keypoints[pointIdxsOfClusters].pt.x,(int)keypoints[pointIdxsOfClusters].pt.y
  *  (coordinates of keypoints) to the file.LabelFile is opened in write mode to write the label of each image. 
  *  
  *  Parameters: * vocabulary- Bag of visual words.
  *              * pathToFile- path to file containing full path of the images.
  **/
  
  bowDE.setVocabulary(vocabulary);
  vector<KeyPoint> keypoints;
  vector<vector<int> > pointIdxsOfClusters;
  Mat imgHistogram,allHist;
  char fileName[200];
  char pathToImage[200];
  int limit=0;
  FILE *labelFilePointer;
  FILE *filePointer;
  FILE *keypointFile;  
  char keypointFileName[200];
  int numOfImgCount=0;
  
  labelFilePointer=fopen("label.txt","a");
  
  if(labelFilePointer==NULL)
  {
    printf("label file  not found\n");
  }
  
  char files[6][30]={"ten","twenty","fifty","hundred","fivehundred","thousand"};
  
  for(int i=0;i<6;i++)
  {	
     sprintf(fileName,"%s/%s.txt",pathToFile,files[i]);
     filePointer=fopen(fileName,"r");
     printf("\n file %d :%s",i,fileName);
     
     if(filePointer==NULL)
     {
        printf("\n File %s not found",files[i]);
	continue;
     }
     
     while(limit<IMAGES_TO_TRAIN)
     { 
       
       imgHistogram=Mat();
       fscanf(filePointer,"%s",pathToImage);//read the pathToImage of ith file 	
       
       if(feof(filePointer))
         break;
       Mat img=imread(pathToImage,0);
       printf("\n Reading the image %s",pathToImage);
       
       if(!img.data)
       {
           printf("\n could not find the image in the path %s",pathToImage);
	   continue;
	}
	
	if(img.cols>480)
	{
	  resize(img,img,Size(480,img.rows*(480.0/(int)img.cols)));
	}
	
	detector.detect(img, keypoints);
	bowDE.compute(img, keypoints,imgHistogram,&pointIdxsOfClusters);    
	
	for(int i=0;i<imgHistogram.cols;i++)
	sprintf(keypointFileName,"%s%d.txt",KEYPOINTS_DIRECTORY,numOfImgCount);

        keypointFile=fopen(keypointFileName,"w");
        
        for(int k=0;k<pointIdxsOfClusters.size();k++)
        {
          if(pointIdxsOfClusters[k].size()>0)
          {
            for(int j=0;j<pointIdxsOfClusters[k].size();j++)
            {
               fprintf(keypointFile,"%d\t%d\t%d\n",k,(int)keypoints[pointIdxsOfClusters[k][j]].pt.x,(int)keypoints[pointIdxsOfClusters[k][j]].pt.y);
            
            }
          }
        }
        fprintf(labelFilePointer,"%d\n",i);
        numOfImgCount++;
        fclose(keypointFile);
        allHist.push_back(imgHistogram);
        limit++;
     }
     limit=0;
     fclose(filePointer);
 }
 
  fclose(labelFilePointer);     
  
  return allHist; 
  
}

Mat tfIdfWeighting(Mat allHist)
{
  /**
  *  perform 'term frequency-inverse document frequency' (tf-idf) weighting.
  *  It uses the formula (nid/nd)*log(N/ni), where nid is the number of occurrences of word i in document d, 
  *  nd is the total number of words in the document d,N is the total number of documents in the whole database,
  *  ni is the number of images in database for weighting. 
  *
  *  Parameter: * allHist- contains the histogram of all images.
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
         numImagesInDb[j]=numImagesInDb[j] + 1;
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
  *  perform inverted index for weightedAllHist. Inverted index contains imgIndex and its corresponding weightedHistValue
  *  These values will be stored in allIndex. 
  *
  *  Parameter: * weightedAllHist- contains weighted histogram of all images
  **/
  
  vector<invertedIndex> allIndex;
  FILE *indicesFilePointer=fopen("indices.txt","w");
  
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
    fprintf(indicesFilePointer,"%d\n",tempIndex.imgIndex.size() );
    allIndex.push_back(tempIndex);
  }
  
  fclose(indicesFilePointer);
  
  return allIndex;    
  
}

void writeToBinaryFile(vector<invertedIndex> allIndex , char *fileToWrite)
{
  /**
  *  writes the data to the .bin file. 
  *
  *  Parameter: * allIndex- vector to write in .bin file.
  *             * fileToWrite- name of file to which vector is to write in .bin file  
  **/
  char binaryFileName[200];
  sprintf(binaryFileName,"%s.bin",fileToWrite);
  fstream binaryFile(binaryFileName, ios::out | ios::binary);
  
  for(int i=0;i<allIndex.size();i++)
  {
     invertedIndex temp = allIndex[i];
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

