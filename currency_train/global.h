#include <opencv2/opencv.hpp>
#include "opencv2/nonfree/features2d.hpp"

const bool DISPLAY  = 0;
const bool writeToYML = 0;
const char KEYPOINTS_DIRECTORY[50] = "keypoints";

const float WIDTH = 480.0;
const float TEST_WIDTH = 640.0;
const int MAXIMAL_KEYPOINTS = 500;
const char txtFiles[6][30] = {"ten", "twenty", "fifty", "hundred", "fivehundred", "thousand"};


cv::SiftFeatureDetector detector( MAXIMAL_KEYPOINTS );
cv::SiftDescriptorExtractor extract;

struct invertedIndex
{
  std::vector<int> imgIndex;
  std::vector<float> weightedHistValue;
};
