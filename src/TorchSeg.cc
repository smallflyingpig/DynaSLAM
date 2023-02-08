/**
 * This file is part of DynaSLAM
*/

#include "TorchSeg.h"
#include <vector>
#include <iostream>

namespace DynaSLAM{

    SegmentTorch::SegmentTorch(std::vector<std::string> && class_name_list)
        :name_list(class_name_list)
    {
        std::cout << "cuda available: " << torch::cuda::is_available() << std::endl;
        std::cout << "cudnn available: " << torch::cuda::cudnn_is_available() << std::endl;
        std::cout << "device count: " << torch::cuda::device_count() << std::endl;
    };

    void SegmentTorch::Initialize(
        int gpu_id, int input_width, int input_height, std::string pretrained_path)
    {
        width = input_width;
        height = input_height;
        int gpu_num = torch::getNumGPUs();
	    if (gpu_id >= gpu_num) {
            std::cout<< "GPU id exceeds max number of gpus";
        } else {
            if(gpu_id>=0){
              device = torch::Device(torch::kCUDA, gpu_id);
            } else {
              device = torch::Device("cpu");
            }
        }

        try {
          // 使用以下命令从文件中反序列化脚本模块: torch::jit::load().
          module = torch::jit::load(pretrained_path);
          module.to(device);
        }
        catch (const c10::Error& e) {
          std::cerr << "error loading the model\n";
        }

    };
    
    cv::Mat SegmentTorch::PredictOneImage(cv::Mat& srcImg, std::vector<std::string> && pred_name_list){
        cv::Mat image = srcImg.clone();
	
	at::Tensor class_index = at::ones({static_cast<signed long>(pred_name_list.size()),}, at::kInt).mul(-1);
	for (auto j = 0; j < pred_name_list.size(); j++) {
		bool find_flag = false;
        for (auto i = 0; i < name_list.size(); i++) {
	    	if (name_list[i] == pred_name_list[j]) {
		    	class_index[j] = i;
				find_flag = true;
			    break;
		    }
	    }
		if (!find_flag) std::cout<< pred_name_list[j] + "not in the name list"; 
	}
	
	int image_width = image.cols;
	int image_height = image.rows;
	cv::resize(image, image, cv::Size(width, height));
	torch::Tensor tensor_image = torch::from_blob(image.data, { 1, height, width,3 }, torch::kByte);
	tensor_image = tensor_image.to(device);
	tensor_image = tensor_image.permute({ 0,3,1,2 });
	tensor_image = tensor_image.to(torch::kFloat);
	tensor_image = tensor_image.div(255.0);

    at::Tensor output;
	try
	{
		output = module.forward({ tensor_image }).toTensor();
	}
	catch (const std::exception& e)
	{
		std::cout << e.what();
	}
	// at::Tensor output = model->forward({ tensor_image });
	output = torch::softmax(output, 1);

	image = cv::Mat::ones(cv::Size(width, height), CV_8UC1);
    std::cout << "the shape of the output: " << output.sizes() << std::endl;
	at::Tensor re_classes = output[0].index({class_index.toType(torch::kLong)}).sum(0).clamp(0,1).mul(255.0).toType(torch::kByte);
	std::cout << "the shape of the return: " << re_classes.sizes() << std::endl;
	at::Tensor re = re_classes.to(at::kCPU).detach();

	memcpy(image.data, re.data_ptr(), width * height * sizeof(unsigned char));
	cv::resize(image, image, cv::Size(image_width, image_height));

	
	return image;
    };
    
}