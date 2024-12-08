#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <winsock2.h>
#include <fstream>
#include <iostream>

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
    labels = Mat(numImages, 10, CV_32F);

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
        labels.row(i) = Scalar::all(0);
        labels.at<float>(i, label) = 1.0f;
    }

    return true;
}

int main() {
    string imageFile = "training-images.idx3-ubyte";
    string labelFile = "training-labels.idx1-ubyte";
    Mat images, labels;

    if (!readMNIST(imageFile, labelFile, images, labels)) {
        return -1;
    }

    cout << "MNIST dataset loaded successfully. Training data: " 
         << images.rows << " samples, " << images.cols << " features." << endl;

    Ptr<ANN_MLP> ann = ANN_MLP::create();
    Mat layers = (Mat_<int>(1, 3) << 784, 128, 10);

    ann->setLayerSizes(layers);
    ann->setActivationFunction(ANN_MLP::SIGMOID_SYM, 0.0, 0.0);
    ann->setTrainMethod(ANN_MLP::BACKPROP, 0.01);
    ann->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, 1e-6));

    cout << "Training ANN..." << endl;
    ann->train(images, ROW_SAMPLE, labels);
    cout << "Training completed!" << endl;

    string modelFile = "trained_digit_model.xml";
    ann->save(modelFile);
    cout << "Model saved to " << modelFile << endl;

    return 0;
}