/************************************************
* Author: MaybeShewill-CV
* File: lanenetModel.h
* Date: 2019/11/5 下午5:19
************************************************/

#ifndef MNN_LANENET_MODEL_H
#define MNN_LANENET_MODEL_H

#include <memory>
#include <string>

#include <opencv2/opencv.hpp>

#include <Interpreter.hpp>
#include <Session.hpp>
#include <Tensor.hpp>
#include "config_parser.h"

using beec::config_parse_utils::ConfigParser;

namespace lane_detection {

class LaneNet {

public:
    /**
     * Remove default construction funciton
     */
    LaneNet() = delete;

    /***
     * Destruction function
     */
    ~LaneNet();

    LaneNet(const ConfigParser& config);

    /***
    * Not allow copy here
    * @param transformer
    */
    LaneNet(const LaneNet& transformer) = delete;

    /***
     * Not allow copy here
     * @param transformer
     * @return
     */
    LaneNet &operator=(const LaneNet& transformer) = delete;

    /***
     * Detect lanes on image using lanenet model
     * @param input_image : input image
     * @param binary_seg_result : binary segmentation result [0, 255] ---> [foreground, background]
     * @param pix_embedding_result : pixel embedding result
     */
    void detect(const cv::Mat& input_image, cv::Mat& binary_seg_result, cv::Mat& pix_embedding_result);

    /***
     * Return if model is successfully initialized
     * @return
     */
    bool is_successfully_initialized() {
        return _m_successfully_initialized;
    }


private:
    // MNN Lanenet model file path
    std::string _m_lanenet_model_file_path = "";
    // MNN Lanenet model interpreter
    std::unique_ptr<MNN::Interpreter> _m_lanenet_model = nullptr;
    // MNN Lanenet model session
    MNN::Session* _m_lanenet_session = nullptr;
    // MNN Lanenet model input tensor
    MNN::Tensor* _m_input_tensor_host = nullptr;
    // MNN Lanenet model binary output tensor
    MNN::Tensor* _m_binary_output_tensor_host = nullptr;
    // MNN Lanenet model pixel embedding output tensor
    MNN::Tensor* _m_pix_embedding_output_tensor_host = nullptr;
    // MNN Lanenet input graph node tensor size
    cv::Size _m_input_node_size_host;
    // successfully init model flag
    bool _m_successfully_initialized = false;

    /***
     * Preprocess image, resize image and scale image into [-1.0, 1.0]
     * @param input_image
     * @param output_image
     */
    void preprocess(const cv::Mat& input_image, cv::Mat& output_image);
};

}

#endif //MNN_LANENET_MODEL_H
