#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <stdio.h>
#include <torch/torch.h> 
#include <torch/script.h>
#include <vector>
#include <fstream>

using namespace cv;
using namespace std; 
using namespace torch;

torch::jit::Module load_model(string model_name);
vector<string> load_classes();
vector<int> topKIndices(const vector<float>& vec, int k);

int main()
{
    torch::jit::script::Module module;
    Mat img = imread("../sample_images/car.jpg", IMREAD_COLOR);
    vector<string> listOfClass = load_classes();
    module = load_model("../net.pt");
    vector<torch::jit::IValue> input;
    vector<double> mean = {0.406, 0.456, 0.485};
    vector<double> std = {0.225, 0.224, 0.229};
    cv::resize(img, img, cv::Size(128, 128));
    img.convertTo(img, CV_32FC3, 1.0f / 255.0f);
    torch::Tensor img_tensor = torch::from_blob(img.data, {1, 128, 128, 3});
    img_tensor = img_tensor.permute({0, 3, 1, 2});
    img_tensor = torch::data::transforms::Normalize<>(mean, std)(img_tensor);
    //img_tensor = img_tensor.to(torch::kCUDA);
    input.push_back(img_tensor);
    auto pred = module.forward(input).toTensor().detach();//.to(torch::kCPU);
    auto probability = pred[0].softmax(0);
    std::vector<float> probability_vector(probability.data_ptr<float>(),
                                      probability.data_ptr<float>() + probability.numel());
    vector<int> top3 = topKIndices(probability_vector, 3);
    cout << "Top 3 recognition:" << endl;
    for (int idx : top3) {
        cout << listOfClass[idx] << endl;
    }

    return 0;
}

vector<int> topKIndices(const vector<float>& vec, int k) {
    // Create a vector of indices from 0 to vec.size() - 1
    std::vector<int> indices(vec.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Sort the indices based on the values in vec
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [&vec](int i1, int i2) {
                          return vec[i1] > vec[i2]; // Sort in descending order
                      });

    // Return the top K indices
    return vector<int>(indices.begin(), indices.begin() + k);
}

vector<string> load_classes()
{
    vector<string> listOfClass;
    ifstream infile("../class.txt");
    string single_word;
    while (getline(infile, single_word)){
        listOfClass.push_back(single_word);
        }
        infile.close();
    return listOfClass;
}

torch::jit::Module load_model(string model_name)
{
    torch::jit::Module module = torch::jit::load(model_name);
    //module.to(torch::kCUDA);
    module.eval();
    return module;
}
