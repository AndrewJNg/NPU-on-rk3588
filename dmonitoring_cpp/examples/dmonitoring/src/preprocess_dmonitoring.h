#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

cv::Mat BGR2YUV420(std::string image_path) {
    cv::Size IMG_SIZE_in(1440, 960);
    cv::Mat img_bgr = cv::imread(image_path);
    cv::resize(img_bgr, img_bgr, IMG_SIZE_in);

    cv::Mat img_yuv;
    cv::cvtColor(img_bgr, img_yuv, cv::COLOR_BGR2YUV);
    std::vector<cv::Mat> channels;
    cv::split(img_yuv, channels);
    cv::Mat y_channel = channels[0];
    cv::Mat output = y_channel.reshape(1, 1382400);
    return output;
}

std::vector<cv::Mat> combine_inputs(std::string IMG_PATH = "dataset/ecam.jpeg") {
    cv::Mat img = BGR2YUV420(IMG_PATH);

    cv::Mat calib_input = cv::Mat::zeros(1, 3, CV_32F);
    std::vector<cv::Mat> result;
    result.push_back(img);
    result.push_back(calib_input);
    return result;
}

// int main() {
//     std::vector<cv::Mat> value = combine_inputs();
//     std::cout << value[0].size() << std::endl;
//     std::cout << value[1].size() << std::endl;
//     return 0;
// }

