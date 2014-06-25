#include "train_utils.cpp"

using namespace cv;
using namespace std;


int main(int argc,char **argv)
{
   
    /******************** extract features ********************/
    Mat siftFeature;
    const int vocabSize = (int)strtol(argv[2], NULL, 0);
    const int numImagesToTrain = (int)strtol(argv[3], NULL, 0);
  
    printf("\nvocabSize selected : %d", vocabSize);
    printf("\nMaximal number of images form each class to be used for training : %d", numImagesToTrain);
    printf("\nExtracting features");
    
    siftFeature=extractSift(argv[1], numImagesToTrain);
    
    printf("\nFeatures extracted : %d", siftFeature.rows);
  
    /******************** cluster the features ********************/
    printf("\nClustering start");
    
    // Mat vocabulary=kMeansCluster(siftFeature, vocabSize);
    Mat vocabulary=hiKMeansCluster(siftFeature,vocabSize);
    
    printf("\nClustering done : %d", vocabulary.rows);
  
    /******************** write vocabulary to .yml and .bin file ********************/
    if(writeToYML)
    {
        printf("\nWriting Vocabulary to the yml file");
        writeToYMLFile(vocabulary, (char *)"vocabulary");
    }
  
    printf("\nWriting Vocabulary to the binary file");
    writeToBinaryFile(vocabulary, (char *)"vocabulary");
    

    vocabulary.convertTo(vocabulary,CV_32F);                //convert vocabulary fron CV_8U to CV_32F
  
  /********** get Histogram of visual words ********************/
  printf("\nHistogram of vocabulary words");
  Mat allHist=getBowHist(vocabulary,argv[1]);
  
  printf("\nWriting Histogram of vocabulary words to YML file");
  writeToYMLFile(allHist,(char *)"allHist");
  
  // /******************** Weighted Histogram of allHist ********************/
  // printf("\n Weighted Histogram");
  // Mat weightedAllHist=tfIdfWeighting(allHist);
  
  // /******************** perform inverted index ********************/
  // printf("\n Indexing");
  // vector<invertedIndex> allIndex = getInvertedIndex(weightedAllHist);
  // printf("\n Total inverted indices = %d", allIndex.size());
  // printf("\n writing to the inverted index file");
  // writeToBinaryFile(allIndex,(char *)"allIndex");
    
  // /********************  write total num of train images(weightedAllHist.rows) and vocabSize to dataFile ********************/
  // FILE *filePointer=fopen("dataFile.txt","w");
  // if(filePointer==NULL)
  // {
  //   printf("\n data File  not found");
  // }
  // fprintf(filePointer,"%d\n%d",weightedAllHist.rows,vocabSize);
  // fclose(filePointer);
  // printf("\n ********************finish********************");
  return 0;
}

