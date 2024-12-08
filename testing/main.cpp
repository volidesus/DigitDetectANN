#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <winsock2.h>

using namespace cv;
using namespace cv::ml;
using namespace std;

bool readMNIST(const string& imageFile, const string& labelFile, Mat& images, Mat& labels) {
    ifstream imageStream(imageFile, ios::binary);
    ifstream labelStream(labelFile, ios::binary);

    if (!imageStream.is_open() || !labelStream.is_open()) {
        cerr << "Failed to open MNIST files: " << imageFile << ", " << labelFile << endl;
        return false;
    }

    int magicNumber, numImages, numRows, numCols;
    imageStream.read((char*)&magicNumber, 4);
    imageStream.read((char*)&numImages, 4);
    imageStream.read((char*)&numRows, 4);
    imageStream.read((char*)&numCols, 4);

    magicNumber = ntohl(magicNumber);
    numImages = ntohl(numImages);
    numRows = ntohl(numRows);
    numCols = ntohl(numCols);

    if (magicNumber != 2051) {
        cerr << "Invalid image file magic number: " << magicNumber << endl;
        return false;
    }

    int labelMagicNumber, numLabels;
    labelStream.read((char*)&labelMagicNumber, 4);
    labelStream.read((char*)&numLabels, 4);

    labelMagicNumber = ntohl(labelMagicNumber);
    numLabels = ntohl(numLabels);

    if (labelMagicNumber != 2049 || numImages != numLabels) {
        cerr << "Invalid label file magic number or mismatch: " << labelMagicNumber << endl;
        return false;
    }

    images = Mat(numImages, numRows * numCols, CV_32F);
    labels = Mat(numImages, 1, CV_32S);

    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < numRows * numCols; j++) {
            unsigned char pixel;
            imageStream.read((char*)&pixel, 1);
            images.at<float>(i, j) = pixel / 255.0f;
        }
    }

    for (int i = 0; i < numLabels; i++) {
        unsigned char label;
        labelStream.read((char*)&label, 1);
        labels.at<int>(i, 0) = (int)label;
    }

    return true;
}

int main() {
    string imageFile = "t10k-images.idx3-ubyte";
    string labelFile = "t10k-labels.idx1-ubyte";
    string modelPath = "trained_digit_model.xml";

    Mat testImages, testLabels;

    if (!readMNIST(imageFile, labelFile, testImages, testLabels)) {
        return -1;
    }

    Ptr<ANN_MLP> model = ANN_MLP::load(modelPath);
    if (model->empty()) {
        cerr << "Failed to load the model from " << modelPath << endl;
        return -1;
    }
    cout << "Model loaded successfully!" << endl;

    cout << "Testing on " << testImages.rows << " samples..." << endl;

    int correctCount = 0;
    for (int i = 0; i < testImages.rows; i++) {
        Mat sample = testImages.row(i);
        Mat output;

        model->predict(sample, output);
        Point maxLoc;
        minMaxLoc(output, nullptr, nullptr, nullptr, &maxLoc);

        int predictedLabel = maxLoc.x;
        int trueLabel = testLabels.at<int>(i, 0);

        if (predictedLabel == trueLabel) {
            correctCount++;
        }
    }

    double accuracy = static_cast<double>(correctCount) / testImages.rows * 100.0;
    cout << "Testing completed!" << endl;
    cout << "Accuracy: " << accuracy << "%" << endl;

    return 0;
}
