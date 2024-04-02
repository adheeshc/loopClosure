#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <chrono>

#include <fbow/fbow.h>
#include <fbow/vocabulary_creator.h>

int main() {
    std::cout << "Reading images..." << std::endl;
    std::vector<cv::Mat> images;
    for (int i = 0; i < 10;i++) {
        std::string path = "../data/" + std::to_string(i + 1) + ".png";
        images.push_back(cv::imread(path));
    }

    // detect ORB features
    cv::Ptr<cv::Feature2D> detector = cv::ORB::create();
    std::vector<cv::Mat> descriptors;
    for (cv::Mat& img : images) {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptor;
        detector->detectAndCompute(img, cv::Mat(), keypoints, descriptor);
        descriptors.emplace_back(descriptor);
    }

    //create vocabulary
    std::cout << "Creating vocabulary with FBow" << std::endl;
    fbow::VocabularyCreator::Params params;
    params.k = 10;
    params.L = 6;
    params.nthreads = 1;
    params.maxIters = 0;
    params.verbose = false;
    fbow::VocabularyCreator vocCreator;
    fbow::Vocabulary vocab;
    std::string descriptorsName = "ORBDescriptors";

    vocCreator.create(vocab, descriptors, descriptorsName, params);
    
    std::cout << "vocabulary size: " << vocab .size()<< std::endl;
    vocab.saveToFile("../output/vocab.yml.gz");
    std::cout << "Done!" << std::endl;

    return 0;
}