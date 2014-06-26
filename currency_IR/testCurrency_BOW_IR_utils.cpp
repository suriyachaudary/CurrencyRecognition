#include <stdio.h>
#include <iostream>
#include <fstream>
#include "../currency_train/global.h"
#include "opencv2/opencv.hpp"
#include "opencv2/nonfree/features2d.hpp"

using namespace cv;
using namespace std;

#define numOfTestImgs 509


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

Mat grabcutSegmentation(Mat input)
{
    /**
    *  implements the grabcut algorithm.It takes image as an argument and it return mask. 
    *
    *  Parameter: * img- image to be segmented.
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
        for(int i=0;i<mask.rows;i++)
        {
            for(int j=0;j<mask.cols;j++)
            {
                if(i>0.2*mask.rows && i<0.8*mask.rows)
                {
                    if(j>0.2*mask.cols && j<0.8*mask.cols)
                    {
                        mask.at<uchar>(i,j)=3;
                    }
                }
            }
        }
    }
    
    cv::compare(mask,cv::GC_PR_FGD,mask,cv::CMP_EQ);
    resize(mask,mask,Size(input.cols,input.rows),0,0,CV_INTER_LINEAR); 
  
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

void getDotProduct(vector<invertedIndex> allIndex,Mat imgHistogram,const int numOfTrainImg,float dotProduct[])
{
    /** 
    *  performs dot product between imgHistogram which is computed output image descriptor which takes cols as clusterSize 
    *  and weightedHistogram value for each image index and it returns dotProduct to retreive top k images.
    *
    *  Parameters: * allIndex- Vector containing img index and its weighted histogram value.
    *              * numOfTrainImg- Total number of images trained.
    *              * dotProduct[]- Array in which dotProduct is stored.
    **/ 
    
    for(int i=0;i<numOfTrainImg;i++)
        dotProduct[i]=0;
    
    for(int i=0;i<imgHistogram.cols;i++)
    {
        for (int j=0;j<allIndex[i].imgIndex.size();j++)
        {
           dotProduct[allIndex[i].imgIndex[j]] +=  imgHistogram.at<float>(0,i) * allIndex[i].weightedHistValue[j];//dotProduct calculation
        }
    }  
}

int argmax(float array[],const int numOfTrainImg)
{
    /** 
    *  identifies index which takes largest dotProduct value and it returns the  index value to retrieveTopKImages function.
    *
    *   Parameters: * array[]- dotProduct array is passed here to identify largest dotProduct value and its corresponding index.
    *               * numOfTrainImg- Total number of images trained.
    **/   
  
    int k = 0;
    float max = array[k];
    for (int i = 0; i < numOfTrainImg; i++)
    {
        if (array[i] > max) 
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
    *  retreives top K indices which takes largest dotProduct value and it stores the value to indices array and it returns to 
    *   identify score value. 
    *
    *  Parameters: * array[]- dotProduct array is passed here to identify top k  dotProduct value and its corresponding indices.
    *              * topKValue- number of top values to be considered.
    *              * numOfTrainImg- Total number of images trained.
    *              * indices[]- array to which top K indices which is having high dotProduct value is stored.
    **/   
  
     for(int i=0;i<topKValue;i++)
        indices[i]=-1;   
    
    for(int i=0;i<topKValue;i++)
    { 
        indices[i]=argmax(dotProduct,numOfTrainImg);
        dotProduct[indices[i]]=-1;
    }   
}

void rerankUsingGeometryVerification(int retrievedImages[],const int topKValue,vector<vector<int> > pointIdxsOfClusters,vector<KeyPoint> keyPoints,char  *keyPointsPath,int geoScore[])
{
    /** 
    *  perform reranking of indices using geometry verification.It reads k number of file which are top k retrieved indices 
    *  named file presents in keypoints directory.pointIdxsOfClusters.size() is the size of indices of keypoints that belong to the cluster.
    *  The x and y values of indices of keypoints is stored in temp, and after checking the condition temp is stored in pointInTestImg and
    *  Mat temp2 is created if vocabularyId and j are equal then x and y values of indices of keypoints is stored in temp2.temp2 is pushed 
    *  to pointInRetrievedImg. If matches(poinInTestImg.rows) is greater than 4 then fundamentalMat is created using findFundamentalMat()
    *  function which takes pointInTestImg,pointInRetreivedImg,FM_RANSAC,3.0,0.99, geo as an argument. 
    *  
    *  Parameters: * retrievedImages[]- array which contains the indices of top K dotProduct values.
    *              * topKValue- number of top values to be considered.
    *              * pointIdxsOfClusters-  indices of keypoints that belong to the cluster.
    *              * keyPoints- interesting points on the object can be extracted to provide a "feature description" of the object.
    *              * keyPointsPath- directory  which contains number of  imgcounts file.
    *              * geoScore[]- array to which calculated score is stored.
    **/   
  
    for(int i=0;i<topKValue;i++)
    { 
        geoScore[i]=0;
        Mat pointInTestImg, pointInRetreivedImg;
        char keyPointsFile[200];
        sprintf(keyPointsFile,"%s/%d.txt",keyPointsPath,retrievedImages[i]);
        printf("\n keyPointsFilename=%s",keyPointsFile);
        FILE *keyPointsFilePointer;
        keyPointsFilePointer= fopen(keyPointsFile,"r");
        int vocabularyId=-1,x=-1,y=-1;
        
        if(keyPointsFilePointer==NULL)
        {
            printf("\n File %d not found",retrievedImages[i]);
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
               temp.at<float>(0,0)=keyPoints[pointIdxsOfClusters[j][k]].pt.x;
               temp.at<float>(0,1)=keyPoints[pointIdxsOfClusters[j][k]].pt.y;
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
                        break;
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

void getVote(int geoScore[],int topKValue,int labels[],int retrievedImages[],int vote[])
{
    /** 
    *  calculates number of votes obtained  for each label using geoScore value obtained from rerankingUsingGeometricVerification and 
    *   it is returned to identify label.
    *  
    *  Parameters: * geoScore[]- contains score value of top K indices.
    *              * topKValue- number of top values to be considered.
    *              * labels[]- array in which label of each train image is stored.
    *              * retrievedImages[]- array in which top K indices is stored.
    *              * vote- array in which vote of each label is stored.
    **/
 
    for(int i=0;i<6;i++)
        vote[i]=0;
   
    for(int i=0;i<topKValue;i++)
        vote[labels[retrievedImages[i]]] = vote[labels[retrievedImages[i]]] + geoScore[i];
}
 
int argmax(int array[])
{
    /** 
    *  identifies index which takes highest vote value and it returns the index value to the label in  test function.
    *   
    *  Parameters: * array[]- vote array is passed to identify the index which is having highest vote.
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

int test(char *pathToTestImg,vector<invertedIndex> allIndex,int labels[],char *keyPointsPath,const int numOfTrainImg,const int topKValue)
{
    /** 
    *   detects keypoints , perform grabcut segmentation, computes an image descriptor using the set visual vocabulary,
    *   following that it calculates dotProduct,retrieve top K images and re-ranking is performed with geometric verification and 
    *   voting is placed and finally label is calculated.
    *   
    *   Parameter: * pathToTestImg - path to file containing the image.
    *              * allIndex- Vector containing img index and its weighted histogram value. 
    *              * labels[]- array in which label of each train image is stored.
    *              * keyPointsPath- directory  which contains number of  imgcounts file.
    *              * numOfTrainImg- Total number of images trained.
    *              * topKValue- number of top values to be considered.
    **/
  
    vector<KeyPoint> keyPoints;
    vector<vector<int> > pointIdxsOfClusters;
    Mat imgHistogram;
    int label;
    Mat img = imread(pathToTestImg,1);
    printf("\n Reading the image %s",pathToTestImg);

    if(!img.data)
    {
        printf("\n could not find the image in the path %s",pathToTestImg);
    }   
  
    resize(img,img,Size(480,360));
  
    /******************** grabcut segmentation ********************/
    Mat mask= grabcutSegmentation(img);
  
    /******************** detect keypoints ********************/
    cvtColor(img, img, CV_BGR2GRAY);
    detector.detect(img, keyPoints);
    printf("\n Keypoints detected before removing keypoints : %d",keyPoints.size() );
    if(keyPoints.size()>10)
        keyPoints= removeKeyPoints(mask,keyPoints);
    printf("\n Keypoints detected after removing keypoints : %d",keyPoints.size() );
      
    bowDE.compute(img, keyPoints,imgHistogram,&pointIdxsOfClusters); //Computes an image histogram using the set visual vocabulary.
       
    /******************** calculate dot product *********************/ 
    float dotProduct[numOfTrainImg];
    getDotProduct(allIndex,imgHistogram,numOfTrainImg,dotProduct);
  
    /******************** retrieve top K images *********************/
    int retrievedImages[topKValue];
    retrieveTopKImages(dotProduct,topKValue,numOfTrainImg,retrievedImages);
    for(int i=0;i<10;i++)
        printf("\n indices=%d",retrievedImages[i]);
  
    /******************* Rerank using geometric verification **********************/
    int geoScore[topKValue];
    rerankUsingGeometryVerification(retrievedImages,topKValue,pointIdxsOfClusters,keyPoints,keyPointsPath,geoScore);
    for(int i=0;i<topKValue;i++)
        printf("\n score=%d",geoScore[i]);
 
    /******************** vote *********************/
    int vote[6];
    getVote(geoScore,topKValue,labels,retrievedImages,vote);
    for(int i=0;i<6;i++)
        printf("\n vote[%d]=%d",i,vote[i]);
   
    /******************** label *********************/
    label= argmax(vote);
   
    return label;
}

void readFiles(char *pathToFile,vector<invertedIndex> allIndex,int labels[],char *keyPointsPath,const int numOfTrainImg,const int topKValue)
{
  /** 
  *   read files containing the full path of the images and calculates label of each image and finally accuracy is calculated.
  *   
  *   Parameter: * pathToFile - path to file containing full path of the images.
  *              * allIndex- Vector containing img index and its weighted histogram value. 
  *              * labels[]- array in which label of each train image is stored.
  *              * keyPointsPath- directory  which contains number of  imgcounts file.
  *              * numOfTrainImg- Total number of images trained.
  *              * topKValue- number of top values to be considered.
  **/
  
  char fileName[200];
  int label;
  int numOfLabelsCrct=0;
  int totNumOfImgs=0;
  float accuracy;
  FILE *filePointer;
  char pathToTestImg[200];
  char files[6][30]={"ten","twenty","fifty","hundred","fivehundred","thousand"};
  for(int i=0;i<6;i++)
  {	
     sprintf(fileName,"%s/%s.txt",pathToFile,files[i]);
     filePointer=fopen(fileName,"r");//read the ith file
     printf("\n file %d :%s",i,fileName);
     if(filePointer==NULL)
     {
       printf("\n File %s not found",files[i]);
       continue;
     }
     while(!feof(filePointer) )
     {          
       fscanf(filePointer,"%s",pathToTestImg);//read images in ith file    
       label = test(pathToTestImg,allIndex,labels,keyPointsPath,numOfTrainImg,topKValue);
       totNumOfImgs++;
       printf("\n label=%d",label);
       if(label==i)
         numOfLabelsCrct++;
     }
  }     
           
  /******************** accuracy *********************/  
  printf("\ntotNumOfImgs=%d",  totNumOfImgs);
  printf("numOfLabelCrct=%d",numOfLabelsCrct);
  accuracy =  ((float)numOfLabelsCrct/numOfTestImgs );
  accuracy=accuracy*100.0;
  printf("\n accuracy= %f",accuracy);
}
