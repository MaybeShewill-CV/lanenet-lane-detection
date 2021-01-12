//
// Created by Nezha on 2016/12/15.
//

/**
* REAMME *
* @author: ZJ Jiang (Nezha)
* @github: https://github.com/CallmeNezha/SimpleDBSCAN
* @describe: This is a simple DBSCAN clustering method implement
*/

// Copyright 2019 Baidu Inc. All Rights Reserved.
// Author: MaybeShewill-CV
// File: dbscan.hpp
// Date: 2019/11/6 下午8:26

#ifndef MNN_DBSCAN_HPP
#define MNN_DBSCAN_HPP

// Option brute force or use kdtree(by default)
#define BRUTEFORCE false

#include <vector>
#include <queue>
#include <set>
#include <memory>
#include <functional>

#if !BRUTEFORCE
/*
* @author: Scott Deming, John Tsiombikas
* @github:https://github.com/sdeming/kdtree
*/
#include "kdtree.h"
#endif

typedef unsigned int uint;

enum CLASSIFY_FLAGS {
    NOISE = -2,
    NOT_CALSSIFIED = -1
};

/***
 * Feature vector. T can be float ,double ,int or any other number type but MUST
 * SUPPORT implicitly convert to double type
 */
template <typename T>
using Feature = std::vector<T>;

/***
 * Dbscan sample which will be clustered. A sample must have a feature vector and class id.
 * @tparam T
 */
template <typename T>
class DBSCAMSample final {
public:
    /***
     * Default constructor not supplied here
     */
    DBSCAMSample() = delete;

    /***
     * Default destructor
     */
    ~DBSCAMSample() = default;

    /***
     * Constructor
     * @param feature : feature vector
     * @param class_id : class id
     */
    DBSCAMSample(const Feature<T>& feature, uint class_id = CLASSIFY_FLAGS::NOT_CALSSIFIED);

    /***
     * Get sample's feature value via index
     * @param idx : feature index
     * @return : T feature value if in range[0, feature_vec.size()) else 0.0
     */
    T operator[](int idx) const;

    /***
     * Get feature vector's value
     * @return
     */
    Feature<T> get_feature_vector() const;

    /***
     * Set feature's value via feature from input parameter
     * @param feature : feature vector
     */
    void set_feature_vector(const Feature<T>& feature);

    /***
     * Get sample's class id
     * @return
     */
    uint get_class_id();

    /***
     * Set sample's class id
     */
    void set_class_id(uint class_id);

private:
    // sample features
    Feature<T> _m_feature = Feature<T>();
    // class id
    uint _m_class_id = CLASSIFY_FLAGS::NOT_CALSSIFIED;
};

template <typename T>
DBSCAMSample<T>::DBSCAMSample(const Feature<T> &feature, uint class_id) {
    _m_feature = feature;
    _m_class_id = class_id;
}

template <typename T>
T DBSCAMSample<T>::operator[](int idx) const {
    // if idx is not in range [0, feature_vector.size() - 1] return 0.0
    if (idx >= _m_feature.size()) {
        return 0.0;
    } else {
        return _m_feature[idx];
    }
}

template <typename T>
Feature<T> DBSCAMSample<T>::get_feature_vector() const {
    return _m_feature;
}

template <typename T>
void DBSCAMSample<T>::set_feature_vector(const Feature<T> &feature) {
    if (_m_feature.size() != feature.size()) {
        _m_feature.resize(feature.size(), 0.0);
        for (auto index = 0; index < feature.size(); ++index) {
            _m_feature[index] = feature[index];
        }
    } else {
        for (auto index = 0; index < feature.size(); ++index) {
            _m_feature[index] = feature[index];
        }
    }
}

template <typename T>
uint DBSCAMSample<T>::get_class_id() {
    return _m_class_id;
}

template <typename T>
void DBSCAMSample<T>::set_class_id(uint class_id) {
    _m_class_id = class_id;
}


//! type T must be a vector-like container and MUST SUPPORT operator[] for iteration
//! Float can be float ,double ,int or any other number type but MUST SUPPORT implicitly convert to double type
template <typename T, typename Float>
class DBSCAN final {

    enum ERROR_TYPE {
        SUCCESS = 0
        , FAILED
        , COUNT
    };

    using TVector      = std::vector<T>;
    using DistanceFunc = std::function<Float(const T&, const T&)>;

public:
    DBSCAN() { }
    ~DBSCAN() { }

    /**
    * @describe: Run DBSCAN clustering alogrithm
    * @param: V {std::vector<T>} : data
    * @param: dim {unsigned int} : dimension of T (a vector-like struct)
    * @param: eps {Float} : epsilon or in other words, radian
    * @param: min {unsigned int} : minimal number of points in epsilon radian, then the point is cluster core point
    * @param: disfunc {DistanceFunc} : !!!! only used in bruteforce mode. Distance function recall. Euclidian distance is recommanded, but you can replace it by any metric measurement function
    * @usage: Object.Run() and get the cluster and noise indices from this->Clusters & this->Noise.
    * @pitfall: If you set big eps(search range) and huge density V, then kdtree will be a bottleneck of performance
    * @pitfall: You MUST ensure the data's identicality (TVector* V) during Run(), because DBSCAN just use the reference of data passed in.
    * @TODO: customize kdtree algorithm or rewrite it ,stop further searching when minimal number which indicates cluster core point condition is satisfied
    */
    int Run(TVector* V, const uint dim, const Float eps, const uint min, const DistanceFunc& disfunc = [](const T& t1, const T& t2)->Float { return 0; });


private:
    std::vector<uint> regionQuery(const uint pid) const;
    void              addToCluster(const uint pid, const uint cid);
    void              expandCluster(const uint cid, const std::vector<uint>& neighbors);
    void              addToBorderSet(const uint pid) {
        this->_borderset.insert(pid);
    }
    void              addToBorderSet(const std::vector<uint>& pids) {
        for (uint pid : pids) this->_borderset.insert(pid);
    }
    bool              isInBorderSet(const uint pid) const {
        return this->_borderset.end() != this->_borderset.find(pid);
    }
    void              buildKdtree(const TVector* V);
    void              destroyKdtree();

public:
    std::vector<std::vector<uint>>  Clusters;
    std::vector<uint>               Noise;

private:
    //temporary variables used during computation
    std::vector<bool>   _visited;
    std::vector<bool>   _assigned;
    std::set<uint>      _borderset;
    uint                _datalen;
    uint                _minpts;
    Float               _epsilon;
    uint                _datadim;

    DistanceFunc        _disfunc;
#if !BRUTEFORCE
    kdtree*             _kdtree;
#endif //!BRUTEFORCE

    std::vector<T>*     _data;  //Not owner, just holder, no responsible for deallocate


};

template<typename T, typename Float>
int DBSCAN<T, Float>::Run(
    TVector*                V
    , const uint            dim
    , const Float           eps
    , const uint            min
    , const DistanceFunc&   disfunc
) {

    // Validate
    if (V->size() < 1) return ERROR_TYPE::FAILED;
    if (dim < 1) return ERROR_TYPE::FAILED;
    if (min < 1) return ERROR_TYPE::FAILED;

    // initialization
    this->_datalen = (uint)V->size();
    this->_visited = std::vector<bool>(this->_datalen, false);
    this->_assigned = std::vector<bool>(this->_datalen, false);
    this->Clusters.clear();
    this->Noise.clear();
    this->_minpts = min;
    this->_data = V;
    this->_disfunc = disfunc;
    this->_epsilon = eps;
    this->_datadim = dim;

#if BRUTEFORCE
#else
    this->buildKdtree(this->_data);
#endif // !BRUTEFORCE


    for (uint pid = 0; pid < this->_datalen; ++pid) {
        // Check if point forms a cluster
        this->_borderset.clear();
        if (!this->_visited[pid]) {
            this->_visited[pid] = true;

            // Outliner it maybe noise or on the border of one cluster.
            const std::vector<uint> neightbors = this->regionQuery(pid);
            if (neightbors.size() < this->_minpts) {
                continue;
            }
            else {
                uint cid = (uint)this->Clusters.size();
                this->Clusters.push_back(std::vector<uint>());
                // first blood
                this->addToBorderSet(pid);
                this->addToCluster(pid, cid);
                this->expandCluster(cid, neightbors);
            }
        }
    }

    for (uint pid = 0; pid < this->_datalen; ++pid) {
        if (!this->_assigned[pid]) {
            this->Noise.push_back(pid);
        }
    }

#if BRUTEFORCE
#else
    this->destroyKdtree();
#endif // !BRUTEFORCE

    return ERROR_TYPE::SUCCESS;

}




template<typename T, typename Float>
void DBSCAN<T, Float>::destroyKdtree() {
    kd_free(this->_kdtree);
}

#if !BRUTEFORCE
template<typename T, typename Float>
void DBSCAN<T, Float>::buildKdtree(const TVector* V)
{
    this->_kdtree = kd_create((int)this->_datadim);
    std::unique_ptr<double[]> v(new double[this->_datadim]);
    for (uint r = 0; r < this->_datalen; ++r) {
        // kdtree only support double type
        for (uint c = 0; c < this->_datadim; ++c) {
            v[c] = (double)(*V)[r][c];
        }
        kd_insert(this->_kdtree, v.get(), (void*)&(*V)[r]);
    }
}
#endif

template<typename T, typename Float>
std::vector<uint> DBSCAN<T, Float>::regionQuery(const uint pid) const {

    std::vector<uint> neighbors;

#if BRUTEFORCE //brute force  O(n^2)
    for (uint i = 0; i < this->_data->size(); ++i)
        if (i != pid &&  this->_disfunc((*this->_data)[pid], (*this->_data)[i]) < this->_epsilon)
            neighbors.push_back(i);
#else //kdtree
    std::unique_ptr<double[]> v(new double[this->_datadim]);
    for (uint c = 0; c < this->_datadim; ++c) {
        v[c] = (double)((*this->_data)[pid][c]);
    }

    kdres* presults = kd_nearest_range(this->_kdtree, v.get(), this->_epsilon);
    while (!kd_res_end(presults)) {
        /* get the data and position of the current result item */
        T* pch = (T*)kd_res_item(presults, v.get());
        uint pnpid = (uint)(pch - &(*this->_data)[0]);
        if(pid != pnpid) neighbors.push_back(pnpid);
        /* go to the next entry */
        kd_res_next(presults);
    }
    kd_res_free(presults);

#endif // !BRUTEFORCE

    return neighbors;
}

template<typename T, typename Float>
void DBSCAN<T, Float>::expandCluster(const uint cid, const std::vector<uint>& neighbors) {

    std::queue<uint> border; // it has unvisited , visited unassigned pts. visited assigned will not appear
    for (uint pid : neighbors) border.push(pid);
    this->addToBorderSet(neighbors);

    while(border.size() > 0) {
        const uint pid = border.front();
        border.pop();

        if (!this->_visited[pid]) {

            // not been visited, great! , hurry to mark it visited
            this->_visited[pid] = true;
            const std::vector<uint> pidneighbors = this->regionQuery(pid);

            // Core point, the neighbors will be expanded
            if (pidneighbors.size() >= this->_minpts) {
                this->addToCluster(pid, cid);
                for (uint pidnid : pidneighbors) {
                    if (!this->isInBorderSet(pidnid)) {
                        border.push(pidnid);
                        this->addToBorderSet(pidnid);
                    }
                }
            }
        }
    }

}

template<typename T, typename Float>
void DBSCAN<T, Float>::addToCluster(const uint pid, const uint cid) {
    this->Clusters[cid].push_back(pid);
    this->_assigned[pid] = true;
}

#endif //MNN_DBSCAN_HPP
