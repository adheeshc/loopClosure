#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <chrono>

#include <fbow/fbow.h>

int main() {
    std::cout << "Reading database..." << std::endl;
    fbow::Vocabulary vocab;
    vocab.readFromFile("../output/vocab.yml.gz");

    if (vocab.size() == 0) {
        std::cerr << "Vocab doesnt exist" << std::endl;
        return 1;
    }

    std::cout << "Reading images..." << std::endl;
    std::vector<cv::Mat> images;
    for (int i = 0; i < 10;i++) {
        std::string path = "../data/" + std::to_string(i + 1) + ".png";
        images.push_back(cv::imread(path));
    }

    // detect ORB features
    


    return 0;
}