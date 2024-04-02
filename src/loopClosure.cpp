#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <chrono>

#include <fbow/fbow.h>

double euclideanDistance(const fbow::fBow& vec1, const fbow::fBow& vec2) {
    double sum = 0.0;
    // Iterate through the first vector
    for (const auto& elem : vec1) {
        auto it = vec2.find(elem.first);
        double val2 = (it != vec2.end()) ? it->second : 0.0;
        double diff = elem.second - val2;
        sum += diff * diff;
    }
    // Also consider elements that are only in vec2
    for (const auto& elem : vec2) {
        if (vec1.find(elem.first) == vec1.end()) {
            sum += elem.second * elem.second;
        }
    }
    return sqrt(sum);
}

//The Bhattacharyya distance is a measure of the similarity of two probability distributions
double bhattacharyyaDistance(const fbow::fBow& vec1, const fbow::fBow& vec2) {
    double sum = 0.0;
    for (auto it1 = vec1.begin(), it2 = vec2.begin(); it1 != vec1.end() && it2 != vec2.end(); ++it1, ++it2) {
        sum += sqrt(it1->second * it2->second);
    }
    return -log(sum);
}

//Cosine similarity measures the cosine of the angle between two vectors, comparing BoW histograms 
// it considers the orientation of the vectors and not their magnitude.

double cosineSimilarity(const fbow::fBow& vec1, const fbow::fBow& vec2) {
    double dotProduct = 0.0, normA = 0.0, normB = 0.0;
    for (auto it1 = vec1.begin(), it2 = vec2.begin(); it1 != vec1.end() && it2 != vec2.end(); ++it1, ++it2) {
        dotProduct += it1->second * it2->second;
        normA += it1->second * it1->second;
        normB += it2->second * it2->second;
    }
    return dotProduct / (sqrt(normA) * sqrt(normB));
}

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
        cv::Mat img = cv::imread(path);
        if (img.empty()) {
            std::cerr << "Failed to load image: " << path << std::endl;
            continue;
        }
        images.push_back(img);
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


    for (int i = 0; i < images.size(); i++) {
        fbow::fBow v1 = vocab.transform(descriptors[i]);

        for (int j = 0; j < images.size(); j++) {
            fbow::fBow v2 = vocab.transform(descriptors[j]);
            double score = fbow::fBow::score(v1, v2);
            std::cout << "image " << i << " vs image " << j << " : " << score << std::endl;
        }
        std::cout << std::endl;
    }


    fbow::fBow vv;

    for (int i = 0;i < 1;i++) {
        vv = vocab.transform(descriptors[0]);
    }

    std::cout << vv.begin()->first << " " << vv.begin()->second << std::endl;
    std::cout << vv.rbegin()->first << " " << vv.rbegin()->second << std::endl;
    // for (auto v : vv)
    //     std::cout << v.first << " ";
    // std::cout << std::endl;


    return 0;
}