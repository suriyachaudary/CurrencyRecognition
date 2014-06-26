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
    siftFeature.release();
  
    /******************** write vocabulary to .yml and .bin file ********************/
    if(writeToYML)
    {
        printf("\nWriting Vocabulary to the yml file");
        writeToYMLFile(vocabulary, (char *)"vocabulary");
    }
  
    printf("\nWriting Vocabulary to the binary file");
    writeToBinaryFile(vocabulary, (char *)"vocabulary.bin");
    

    
  
    /********** get BOW histogram for each image ********************/
    printf("\ngetting BOW histogram for each image");
    Mat allHist=getBowHist(vocabulary, argv[1], numImagesToTrain);
    
    vocabulary.release();

    /******************** Weighted Histogram of allHist ********************/
    printf("\ntf-idf scoring");
    Mat weightedAllHist=tfIdfWeighting(allHist);
    
    allHist.release();

    if(writeToYML)
    {
        printf("\nWriting tf-idf weighted BOW histogram for each image to YML file");
        writeToYMLFile(weightedAllHist,(char *)"weightedAllHist");
    }

  
    /******************** perform inverted index ********************/
    printf("\nGetting inverted index");
    vector<invertedIndex> allIndex = getInvertedIndex(weightedAllHist);
    weightedAllHist.release();
    printf("\nTotal inverted indices = %d", allIndex.size());
    printf("\nWriting inverted index to binary file");
    writeToBinaryFile(allIndex,(char *)"allIndex.bin");
    
    /********************  write total num of train images and vocabSize to data file ********************/
    printf("\nWrite total num of train images and vocabSize to data file");
    FILE *filePointer=fopen("dataFile.txt","w");
    if(filePointer==NULL)
    {
        printf("\nCouldn't open 'datafile.txt'");
        return 0;
    }

    fprintf(filePointer,"%d\n%d", weightedAllHist.rows, vocabSize);
    fclose(filePointer);
    printf("\n\n********************finish********************\n\n");
    return 0;
}

