/************************************************
* Copyright 2019 Baidu Inc. All Rights Reserved.
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
#include "dbscan.hpp"

namespace beec_task {
namespace lane_detection {

using beec::config_parse_utils::ConfigParser;
using DBSCAMSample = DBSCAMSample<float>;
using Feature = Feature<float>;

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

    /***
     * Constructor. Using config file to setup lanenet model. Mainly defined object are as follows:
     * 1.Init mnn model file path
     * 2.Init lanenet model pixel embedding feature dims
     * 3.Init dbscan cluster search radius eps threshold
     * 4.Init dbscan cluster min pts which are supposed to belong to a core object.
     * @param config : ConfigParser object
     */
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
     * @param instance_seg_result : instance segmentation result
     */
    void detect(const cv::Mat& input_image, cv::Mat& binary_seg_result, cv::Mat& instance_seg_result);

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
    // lanenet pixel embedding feature dims
    uint _m_lanenet_pix_embedding_feature_dims=4;
    // Dbscan eps threshold
    float _m_dbscan_eps = 0.0;
    // dbscan min pts threshold
    uint _m_dbscan_min_pts = 0;
    // successfully init model flag
    bool _m_successfully_initialized = false;

    /***
     * Preprocess image, resize image and scale image into [-1.0, 1.0]
     * @param input_image
     * @param output_image
     */
    void preprocess(const cv::Mat& input_image, cv::Mat& output_image);

    /***
     * Gathet embedding features via binary segmentation mask
     * @param binary_mask
     * @param pixel_embedding
     * @param coords
     * @param embedding_features
     */
    void gather_pixel_embedding_features(const cv::Mat& binary_mask, const cv::Mat& pixel_embedding,
                                         std::vector<cv::Point>& coords, std::vector<DBSCAMSample>& embedding_samples);

    /***
     * Cluster pixel embedding features via DBSCAN
     * @param embedding_samples
     * @param cluster_ret
     */
    void cluster_pixem_embedding_features(std::vector<DBSCAMSample>& embedding_samples,
                                          std::vector<std::vector<uint> >& cluster_ret, std::vector<uint>& noise);

    /***
     * Visualize instance segmentation result
     * @param cluster_ret
     * @param coords
     * @param instance_segmentation_result
     */
    static void visualize_instance_segmentation_result(const std::vector<std::vector<uint> >& cluster_ret,
            const std::vector<cv::Point>& coords, cv::Mat& instance_segmentation_result);

    /***
     * Normalize input samples' feature. Each sample's feature is normalized via function as follows:
     * feature[i] = (feature[i] - mean_feature_vector[i]) / stddev_feature_vector[i].
     * @param input_samples : vector of samples whose feature vector need to be normalized
     * @param output_samples : normalized result
     */
    static void normalize_sample_features(const std::vector<DBSCAMSample >& input_samples,
                                          std::vector<DBSCAMSample >& output_samples);

    /***
     * Calculate the mean feature vector among a vector of DBSCAMSample samples
     * @param input_samples : vector of DBSCAMSample samples
     * @return : mean feature vector
     */
    static Feature calculate_mean_feature_vector(const std::vector<DBSCAMSample >& input_samples);

    /***
     * Calculate the stddev feature vector among a vector of DBSCAMSample samples
     * @param input_samples : vector of DBSCAMSample samples
     * @param mean_feature_vec : mean feature vector
     * @return : stddev feature vector
     */
    static Feature calculate_stddev_feature_vector(
            const std::vector<DBSCAMSample >& input_samples,
            const Feature& mean_feature_vec);

    /***
     * simultaneously random shuffle two vector inplace. The two input source vector should have the same size.
     * @tparam T
     * @param src1
     * @param src2
     */
    template <typename T1, typename T2>
    static void simultaneously_random_shuffle(std::vector<T1> src1, std::vector<T2> src2);
};

}
}

#endif //MNN_LANENET_MODEL_H
