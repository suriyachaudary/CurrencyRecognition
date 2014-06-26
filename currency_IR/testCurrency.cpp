#include "testCurrency_BOW_IR_utils.cpp"

using namespace cv;

int main(int argc,char **argv)
{

    /******************** read data file containing number of images used for training and vocabSize ********************/

    int numOfTrainImg;
    int vocabSize;
    FILE *datafilePointer=fopen(argv[1],"r");

    if(datafilePointer==NULL)
    {
        printf("\nERROR..!! data file  not found");
        return 0;
    }

    fscanf(datafilePointer,"%d%d",&numOfTrainImg,&vocabSize);
    fclose(datafilePointer);

    printf("\nNumber of images used for training : %d",numOfTrainImg);
    printf("\nVocabulary Size : %d\n", vocabSize);
   
    /******************** read vocabulary from file ********************/

      Mat vocabulary=readVocab(argv[2], (const int)vocabSize);
    printf("\nRead vocabulary : %dx%d", vocabulary.rows, vocabulary.cols);

    setVocabulary(vocabulary);

    /******************** read annotations from file ********************/

    printf("\nReading annotations from file");
    int *labels = new int[numOfTrainImg];
    readLabels(argv[3], numOfTrainImg, labels);

    /******************** read size of each invertedIndex ********************/

    printf("\nReading size of each invertedIndex");
    
    int *indicesSize = new int[vocabSize];
    readSize(argv[4],vocabSize, indicesSize);
   
    /******************** reading inverted indices ********************/

    printf("\nReading inverted index from file");

    vector<invertedIndex> allIndex = readInvertedIndex(argv[5], indicesSize, vocabSize );
    printf("\nIndices read  : %d", allIndex.size());

    int topKValue=(int)strtol(argv[6], NULL, 0);

    printf("\nBOW retrieval parameters in use :");
    printf("\nVocabulary size  : %dx%d", vocabulary.rows, vocabulary.cols);
    printf("\nNumber of images used for training : %d", numOfTrainImg);
    printf("\nNumber of images to retrieve for geo-metric verification : %d", topKValue);
  
    /******************** identifying labels ********************/
    // readFiles(argv[5], allIndex, labels, argv[6],numOfTrainImg,topKValue);
   
    printf("\n\n********************finish********************\n\n");
      
  return 0;
  
}
