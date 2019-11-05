/************************************************
* Author: MaybeShewill-CV
* File: lanenetModel.cpp
* Date: 2019/11/5 下午5:19
************************************************/

#include "lanenet_model.h"

#include <glog/logging.h>

#include <AutoTime.hpp>

namespace lane_detection {
/******************Public Function Sets***************/

/***
 *
 * @param config
 */
LaneNet::LaneNet(const beec::config_parse_utils::ConfigParser &config) {
    using config_content = std::map<std::string, std::string>;

    config_content config_section;
    try {
        config_section = config["LaneNet"];
    } catch (const std::out_of_range& e) {
        LOG(ERROR) << e.what();
        LOG(ERROR) << "Can not get LaneNet section content in config file, please check again";
        _m_successfully_initialized = false;
        return;
    }

    if (config_section.find("model_file_path") == config_section.end()) {
        LOG(ERROR) << "Can not find \"model_file_path\" field in config section";
        _m_successfully_initialized = false;
        return;
    } else {
        _m_lanenet_model_file_path = config_section["model_file_path"];
    }

    _m_lanenet_model = std::unique_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(
                           _m_lanenet_model_file_path.c_str()));
    if (nullptr == _m_lanenet_model) {
        LOG(ERROR) << "Construct lanenet mnn interpreter failed";
        _m_successfully_initialized = false;
        return;
    }

    MNN::ScheduleConfig mnn_config;
    mnn_config.type = MNN_FORWARD_CPU;
    mnn_config.numThread = 4;

    MNN::BackendConfig backend_config;
    backend_config.precision = MNN::BackendConfig::Precision_High;
    backend_config.power = MNN::BackendConfig::Power_High;
    mnn_config.backendConfig = &backend_config;

    _m_lanenet_session = _m_lanenet_model->createSession(mnn_config);
    if (nullptr == _m_lanenet_session) {
        LOG(ERROR) << "Construct laneNet mnn session failed";
        _m_successfully_initialized = false;
        return;
    }

    std::string input_node_name = "lanenet/input_tensor";
    std::string pix_embedding_output_name = "lanenet/final_pixel_embedding_output";
    std::string binary_output_name = "lanenet/final_binary_output";
    _m_input_tensor_host = _m_lanenet_model->getSessionInput(
                               _m_lanenet_session, input_node_name.c_str());
    _m_binary_output_tensor_host = _m_lanenet_model->getSessionOutput(
                                       _m_lanenet_session, binary_output_name.c_str());
    _m_pix_embedding_output_tensor_host = _m_lanenet_model->getSessionOutput(
            _m_lanenet_session, pix_embedding_output_name.c_str());
    _m_input_node_size_host.width = _m_input_tensor_host->width();
    _m_input_node_size_host.height = _m_input_tensor_host->height();

    for (auto i : _m_binary_output_tensor_host->shape()) {
        LOG(INFO) << i;
    }

    _m_successfully_initialized = true;
    return;
}

/***
 * Destructor
 */
LaneNet::~LaneNet() {
    _m_lanenet_model->releaseModel();
    _m_lanenet_model->releaseSession(_m_lanenet_session);
}

/***
 * Detect lanes on image using lanenet model
 * @param input_image
 * @param binary_seg_result
 * @param pix_embedding_result
 */
void LaneNet::detect(const cv::Mat &input_image, cv::Mat &binary_seg_result, cv::Mat &pix_embedding_result) {

    // preprocess
    cv::Mat input_image_copy;
    input_image.copyTo(input_image_copy);
    {
        AUTOTIME
        preprocess(input_image, input_image_copy);
    }

    // run session
    MNN::Tensor input_tensor_user(_m_input_tensor_host, MNN::Tensor::DimensionType::TENSORFLOW);
    {
        AUTOTIME
        auto input_tensor_user_data = input_tensor_user.host<float>();
        auto input_tensor_user_size = input_tensor_user.size();
        ::mempcpy(input_tensor_user_data, input_image_copy.data, input_tensor_user_size);

        _m_input_tensor_host->copyFromHostTensor(&input_tensor_user);
        _m_lanenet_model->runSession(_m_lanenet_session);
    }

    // output graph node
    MNN::Tensor binary_output_tensor_user(
        _m_binary_output_tensor_host, MNN::Tensor::DimensionType::TENSORFLOW);
    MNN::Tensor pix_embedding_output_tensor_user(
        _m_pix_embedding_output_tensor_host, MNN::Tensor::DimensionType::TENSORFLOW);
    _m_binary_output_tensor_host->copyToHostTensor(&binary_output_tensor_user);
    _m_pix_embedding_output_tensor_host->copyToHostTensor(&pix_embedding_output_tensor_user);

    auto binary_output_data = binary_output_tensor_user.host<float>();
    cv::Mat binary_output_mat(_m_input_node_size_host, CV_32FC1, binary_output_data);
    binary_output_mat *= 255;
    binary_output_mat.convertTo(binary_seg_result, CV_8UC1);

    auto pix_embedding_output_data = pix_embedding_output_tensor_user.host<float>();
    cv::Mat pix_embedding_output_mat(
        _m_input_node_size_host, CV_32FC4, pix_embedding_output_data);
    pix_embedding_output_mat.convertTo(pix_embedding_result, CV_8UC4);
}

/***************Private Function Sets*******************/

/***
 * Resize image and scale image into [-1.0, 1.0]
 * @param input_image
 * @param output_image
 */
void LaneNet::preprocess(const cv::Mat &input_image, cv::Mat& output_image) {

    if (input_image.type() != CV_32FC3) {
        input_image.convertTo(output_image, CV_32FC3);
    }

    if (output_image.size() != _m_input_node_size_host) {
        cv::resize(output_image, output_image, _m_input_node_size_host);
    }

    cv::divide(output_image, cv::Scalar(127.5, 127.5, 127.5), output_image);
    cv::subtract(output_image, cv::Scalar(1.0, 1.0, 1.0), output_image);

    return;
}
}
