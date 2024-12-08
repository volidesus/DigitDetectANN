#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>

using namespace cv;
using namespace cv::ml;

Mat canvas;
bool drawing = false;
Point prevPoint;

void drawDigit(int event, int x, int y, int flags, void* userdata) {
    if (event == EVENT_LBUTTONDOWN) {
        drawing = true;
        prevPoint = Point(x, y);
    }
    else if (event == EVENT_MOUSEMOVE) {
        if (drawing) {
            line(canvas, prevPoint, Point(x, y), Scalar(255), 10);
            prevPoint = Point(x, y);
        }
    }
    else if (event == EVENT_LBUTTONUP) {
        drawing = false;
    }
}

int main() {
    std::string modelPath = "trained_digit_model.xml";
    Ptr<ANN_MLP> model = ANN_MLP::load(modelPath);
    if (model->empty()) {
        std::cerr << "Failed to load the model from " << modelPath << std::endl;
        return -1;
    }
    std::cout << "Model loaded successfully!" << std::endl;

    canvas = Mat::zeros(300, 300, CV_8UC1);

    namedWindow("MNIST Neural Network", WINDOW_NORMAL);
    setMouseCallback("MNIST Neural Network", drawDigit);

    while (true) {
        Mat tempCanvas = canvas.clone();

        Mat processedImg;
        resize(tempCanvas, processedImg, Size(28, 28));
        processedImg.convertTo(processedImg, CV_32F, 1.0 / 255.0);

        Mat input;
        processedImg.reshape(1, 1).copyTo(input);

        Mat output;
        model->predict(input, output);

        Point maxLoc;
        double confidence;
        minMaxLoc(output, 0, &confidence, 0, &maxLoc);
        int predictedLabel = maxLoc.x;

        putText(tempCanvas, "Press R to reset", Point(10, 268), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255), 1);
        std::string predictionText = "Prediction: " + std::to_string(predictedLabel) + " - " + std::to_string(static_cast<int>(confidence * 100.0)) + "%";
        putText(tempCanvas, predictionText, Point(10, 290), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255), 1);

        imshow("MNIST Neural Network", tempCanvas);

        char key = (char)waitKey(10);
        if (key == 27) break;
        if (key == 'r') canvas.setTo(Scalar(0));
    }

    destroyAllWindows();
    return 0;
}
