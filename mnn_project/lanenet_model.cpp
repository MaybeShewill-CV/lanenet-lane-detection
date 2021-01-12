/************************************************
* Copyright 2019 Baidu Inc. All Rights Reserved.
* Author: MaybeShewill-CV
* File: lanenetModel.cpp
* Date: 2019/11/5 下午5:19
************************************************/

#include "lanenet_model.h"

#include <omp.h>

#include <glog/logging.h>
#include <boost/lexical_cast.hpp>

#include <AutoTime.hpp>
#include "dbscan.hpp"

namespace beec_task {
namespace lane_detection {

/******************Public Function Sets***************/

/***
 * Constructor. Using config file to setup lanenet model. Mainly defined object are as follows:
 * 1.Init mnn model file path
 * 2.Init lanenet model pixel embedding feature dims
 * 3.Init dbscan cluster search radius eps threshold
 * 4.Init dbscan cluster min pts which are supposed to belong to a core object.
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

    if (config_section.find("dbscan_neighbor_radius") == config_section.end()) {
        LOG(ERROR) << "Can not find \"dbscan_neighbor_radius\" field in config section";
        _m_successfully_initialized = false;
        return;
    } else {
        _m_dbscan_eps = boost::lexical_cast<float>(config_section["dbscan_neighbor_radius"]);
    }

    if (config_section.find("dbscan_core_object_min_pts") == config_section.end()) {
        LOG(ERROR) << "Can not find \"dbscan_core_object_min_pts\" field in config section";
        _m_successfully_initialized = false;
        return;
    } else {
        _m_dbscan_min_pts = boost::lexical_cast<uint>(config_section["dbscan_core_object_min_pts"]);
    }

    if (config_section.find("pix_embedding_feature_dims") == config_section.end()) {
        LOG(ERROR) << "Can not find \"pix_embedding_feature_dims\" field in config section";
        _m_successfully_initialized = false;
        return;
    } else {
        _m_lanenet_pix_embedding_feature_dims = boost::lexical_cast<uint>(config_section["pix_embedding_feature_dims"]);
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
void LaneNet::detect(const cv::Mat &input_image, cv::Mat &binary_seg_result, cv::Mat &instance_seg_result) {

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

    // gather pixel embedding features
    std::vector<cv::Point> coords;
    std::vector<DBSCAMSample> pixel_embedding_samples;
    gather_pixel_embedding_features(binary_seg_result, pix_embedding_output_mat,coords, pixel_embedding_samples);

    // simultaneously random shuffle embedding vector and coord vector inplace
    simultaneously_random_shuffle<cv::Point, DBSCAMSample >(coords, pixel_embedding_samples);

    // normalize pixel embedding features
    normalize_sample_features(pixel_embedding_samples, pixel_embedding_samples);

    // cluster samples
    std::vector<std::vector<uint> > cluster_ret;
    std::vector<uint> noise;
    {
        AUTOTIME
        cluster_pixem_embedding_features(pixel_embedding_samples, cluster_ret, noise);
    }

    // visualize instance segmentation
    instance_seg_result = cv::Mat(_m_input_node_size_host, CV_8UC3, cv::Scalar(0, 0, 0));
    {
        AUTOTIME
        visualize_instance_segmentation_result(cluster_ret, coords, instance_seg_result);
    }
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

/***
 * Gather pixel embedding features via binary segmentation result
 * @param binary_mask
 * @param pixel_embedding
 * @param coords
 * @param embedding_features
 */
void LaneNet::gather_pixel_embedding_features(const cv::Mat &binary_mask, const cv::Mat &pixel_embedding,
        std::vector<cv::Point> &coords,
        std::vector<DBSCAMSample> &embedding_samples) {

    CHECK_EQ(binary_mask.size(), pixel_embedding.size());
    auto image_rows = _m_input_node_size_host.height;
    auto image_cols = _m_input_node_size_host.width;

    for (auto row = 0; row < image_rows; ++row) {
        auto binary_image_row_data = binary_mask.ptr<uchar>(row);
        auto embedding_image_row_data = pixel_embedding.ptr<cv::Vec4f>(row);
        for (auto col = 0; col < image_cols; ++col) {
            auto binary_image_pix_value = binary_image_row_data[col];
            if (binary_image_pix_value == 255) {
                coords.emplace_back(cv::Point(col, row));
                Feature embedding_features;
                for (auto index = 0; index < 4; ++index) {
                    embedding_features.push_back(embedding_image_row_data[col][index]);
                }
                DBSCAMSample sample(embedding_features, CLASSIFY_FLAGS::NOT_CALSSIFIED);
                embedding_samples.push_back(sample);
            }
        }
    }
}

/***
 *
 * @param embedding_samples
 * @param cluster_ret
 */
void LaneNet::cluster_pixem_embedding_features(std::vector<DBSCAMSample> &embedding_samples,
        std::vector<std::vector<uint> > &cluster_ret, std::vector<uint>& noise) {

    if (embedding_samples.empty()) {
        LOG(INFO) << "Pixel embedding samples empty";
        return;
    }

    // dbscan cluster
    auto dbscan = DBSCAN<DBSCAMSample, float>();
    dbscan.Run(&embedding_samples, _m_lanenet_pix_embedding_feature_dims, _m_dbscan_eps, _m_dbscan_min_pts);
    cluster_ret = dbscan.Clusters;
    noise = dbscan.Noise;
}

/***
 * Visualize instance segmentation result
 * @param cluster_ret
 * @param coords
 */
void LaneNet::visualize_instance_segmentation_result(
    const std::vector<std::vector<uint> > &cluster_ret,
    const std::vector<cv::Point> &coords,
    cv::Mat& intance_segmentation_result) {

    LOG(INFO) << "Cluster nums: " << cluster_ret.size();

    std::map<int, cv::Scalar> color_map = {
        {0, cv::Scalar(0, 0, 255)},
        {1, cv::Scalar(0, 255, 0)},
        {2, cv::Scalar(255, 0, 0)},
        {3, cv::Scalar(255, 0, 255)},
        {4, cv::Scalar(0, 255, 255)},
        {5, cv::Scalar(255, 255, 0)},
        {6, cv::Scalar(125, 0, 125)},
        {7, cv::Scalar(0, 125, 125)}
    };

    omp_set_num_threads(4);
    for (int class_id = 0; class_id < cluster_ret.size(); ++class_id) {
        auto class_color = color_map[class_id];
        #pragma omp parallel for
        for (auto index = 0; index < cluster_ret[class_id].size(); ++index) {
            auto coord = coords[cluster_ret[class_id][index]];
            auto image_col_data = intance_segmentation_result.ptr<cv::Vec3b>(coord.y);
            image_col_data[coord.x][0] = class_color[0];
            image_col_data[coord.x][1] = class_color[1];
            image_col_data[coord.x][2] = class_color[2];
        }
    }
}

/***
 * Calculate the mean feature vector among a vector of DBSCAMSample samples
 * @param input_samples
 * @return
 */
Feature LaneNet::calculate_mean_feature_vector(const std::vector<DBSCAMSample> &input_samples) {

    if (input_samples.empty()) {
        return Feature();
    }

    auto feature_dims = input_samples[0].get_feature_vector().size();
    auto sample_nums = input_samples.size();
    Feature mean_feature_vec;
    mean_feature_vec.resize(feature_dims, 0.0);
    for (const auto& sample : input_samples) {
        for (auto index = 0; index < feature_dims; ++index) {
            mean_feature_vec[index] += sample[index];
        }
    }
    for (auto index = 0; index < feature_dims; ++index) {
        mean_feature_vec[index] /= sample_nums;
    }

    return mean_feature_vec;
}

/***
 *
 * @param input_samples
 * @param mean_feature_vec
 * @return
 */
Feature LaneNet::calculate_stddev_feature_vector(
        const std::vector<DBSCAMSample> &input_samples,
        const Feature& mean_feature_vec) {

    if (input_samples.empty()) {
        return Feature();
    }

    auto feature_dims = input_samples[0].get_feature_vector().size();
    auto sample_nums = input_samples.size();

    // calculate stddev feature vector
    Feature stddev_feature_vec;
    stddev_feature_vec.resize(feature_dims, 0.0);
    for (const auto& sample : input_samples) {
        for (auto index = 0; index < feature_dims; ++index) {
            auto sample_feature = sample.get_feature_vector();
            auto diff = sample_feature[index] - mean_feature_vec[index];
            diff = std::pow(diff, 2);
            stddev_feature_vec[index] += diff;
        }
    }
    for (auto index = 0; index < feature_dims; ++index) {
        stddev_feature_vec[index] /= sample_nums;
        stddev_feature_vec[index] = std::sqrt(stddev_feature_vec[index]);
    }

    return stddev_feature_vec;
}

/***
 * Normalize input samples' feature. Each sample's feature is normalized via function as follows:
 * feature[i] = (feature[i] - mean_feature_vector[i]) / stddev_feature_vector[i].
 * @param input_samples
 * @param output_samples
 */
void LaneNet::normalize_sample_features(const std::vector<DBSCAMSample> &input_samples,
                                        std::vector<DBSCAMSample> &output_samples) {
    // calcualte mean feature vector
    Feature mean_feature_vector = calculate_mean_feature_vector(input_samples);

    // calculate stddev feature vector
    Feature stddev_feature_vector = calculate_stddev_feature_vector(input_samples, mean_feature_vector);

    std::vector<DBSCAMSample> input_samples_copy = input_samples;
    for (auto& sample : input_samples_copy) {
        auto feature = sample.get_feature_vector();
        for (auto index = 0; index < feature.size(); ++index) {
            feature[index] = (feature[index] - mean_feature_vector[index]) / stddev_feature_vector[index];
        }
        sample.set_feature_vector(feature);
    }
    output_samples = input_samples_copy;
}

/***
 * simultaneously random shuffle two vector inplace. The two input source vector should have the same size.
 * @tparam T
 * @param src1
 * @param src2
 */
template <typename T1, typename T2>
void LaneNet::simultaneously_random_shuffle(std::vector<T1> src1, std::vector<T2> src2) {

    CHECK_EQ(src1.size(), src2.size());
    if (src1.empty() || src2.empty()) {
        return;
    }

    // construct index vector of two input src
    std::vector<uint> indexes;
    indexes.reserve(src1.size());
    std::iota(indexes.begin(), indexes.end(), 0);
    std::random_shuffle(indexes.begin(), indexes.end());

    // make copy of two input vector
    std::vector<T1> src1_copy(src1);
    std::vector<T2> src2_copy(src2);

    // random two source input vector via random shuffled index vector
    for (uint i = 0; i < indexes.size(); ++i) {
        src1[i] = src1_copy[indexes[i]];
        src2[i] = src2_copy[indexes[i]];
    }
}

}
}
