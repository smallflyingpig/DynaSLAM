/***
 * 
*/

#ifndef __TORCHSEG_H
#define __TORCHSEG_H
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <vector>
namespace DynaSLAM{

class SegmentTorch{
private:
    torch::jit::script::Module module;
    int width = 512; int height = 512;
    std::vector<std::string> name_list;
    torch::Device device = torch::Device(torch::kCPU);
    std::vector<std::string> coco_pred_name_list = {"person", "bicycle", "car", "motorcycle", "airplane", 
                                    "bus", "train", "truck", "boat", "bird", "cat", "dog", 
                                    "horse", "sheep", "cow", "elephant", "bear", "zebra", 
                                    "giraffe" };
    std::vector<std::string>  coco_classes_name = {"BG", "person", "bicycle", "car", "motorcycle", "airplane",
                            "bus", "train", "truck", "boat", "traffic light",
                            "fire hydrant", "stop sign", "parking meter", "bench", "bird",
                            "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
                            "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
                            "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                            "kite", "baseball bat", "baseball glove", "skateboard",
                            "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                            "fork", "knife", "spoon", "bowl", "banana", "apple",
                            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
                            "donut", "cake", "chair", "couch", "potted plant", "bed",
                            "dining table", "toilet", "tv", "laptop", "mouse", "remote",
                            "keyboard", "cell phone", "microwave", "oven", "toaster",
                            "sink", "refrigerator", "book", "clock", "vase", "scissors",
                            "teddy bear", "hair drier", "toothbrush"};

public:
    SegmentTorch(std::vector<std::string> && class_name_list);
    ~SegmentTorch() {};
    void Initialize(int gpu_id, int input_width, int input_height, std::string pretrained_path);
    cv::Mat PredictOneImage(cv::Mat& image, std::vector<std::string> && pred_name_list);
};
    

}
//cv::Mat image = cv::imread("your path to voc_person_seg\\val\\2007_004000.jpg");
//Segmentor<FPN> segmentor;
//segmentor.Initialize(0,512,512,{"background","person"},
//                      "resnet34","your path to resnet34.pt");
//segmentor.LoadWeight("segmentor.pt"/*the saved .pt path*/);
//segmentor.Predict(image,"person"/*class name for showing*/);
#endif 