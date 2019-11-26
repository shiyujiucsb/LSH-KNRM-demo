/*
 * Implementation of KNRM ranking model.
*/

#include <bitset>
#include <random>
#include <chrono>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <limits>
#include <algorithm>
#include <thread>
#include <cstdint>

#include "Eigen/Dense"

using RMatrixXd =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VectorXd = Eigen::Matrix<double, 1, Eigen::Dynamic, Eigen::RowMajor>;

// Number of bits in LSH fingerprint.
constexpr int N_LSH_BITS = 256;
using LSHFingerprint = std::bitset<N_LSH_BITS>;

// Embedding dimension.
constexpr int DIM = 200;

namespace nn4ir {

const unsigned SEED =
               std::chrono::system_clock::now().time_since_epoch().count();

VectorXd Tanh(const VectorXd & input){
    VectorXd voutput = VectorXd::Zero(input.size());
    for(int i = 0 ; i < voutput.size(); ++ i){
        double exp_val = exp( - 2.0 * input[i] );
        voutput[i] = ( 1.0 - exp_val) / ( 1.0 + exp_val);
    }
    return std::move(voutput);
}

/*
 * Compute LSH fingerprint.
 * */
LSHFingerprint ComputeLSHFingerprint(
                             const VectorXd & vec,
                             const std::vector<VectorXd> lsh_random_vectors) {
    assert(lsh_random_vectors.size() == N_LSH_BITS);
    LSHFingerprint result(0);
    for (const VectorXd& r : lsh_random_vectors) {
        if (vec.dot(r) > 0) {
          result.set(0);
        }
        result <<= 1;
    }
    return std::move(result);
}

/*
 * Initialize LSH random vectors.
 * */
std::vector<VectorXd> InitLshRandomVectors() {
    std::default_random_engine generator(SEED);
    std::normal_distribution<double> distri(0.0, 1.0);

    std::vector<VectorXd> vectors;
    vectors.reserve(N_LSH_BITS);
    for (int i = 0; i < N_LSH_BITS; i++) {
        VectorXd lsh_rand_vec(DIM);
        for (int j = 0; j < DIM; j++) {
            lsh_rand_vec[j] = distri(generator);
        }
        vectors.emplace_back(lsh_rand_vec / lsh_rand_vec.norm());
    }

    return std::move(vectors);
}

/*
 * Compute ranking score based on original embeddings.
 * */
double ComputeRankingScore(
                const std::unordered_map<int, VectorXd>& id2embedding_mm,
                const std::vector<int>& query_term_ids,
                const std::vector<int>& doc_term_ids,
                const std::vector<RMatrixXd>& vW1,
                const VectorXd& vW2,
                const std::vector<VectorXd>& vW3){
    assert(vW1.size() == vW3.size());

    int64_t a,b,c,d,e;
    constexpr double eps = 1e-4;
    double score = 0;
    const int nQSize = query_term_ids.size();
    const int nVecDim = vW2.size();
    const int vW1Size = vW1.size();
    assert(vW1Size > 0);
    const int iDocLen = doc_term_ids.size();

    if(iDocLen == 0 || nQSize == 0) {
      return std::numeric_limits<double>::lowest();
    }
    RMatrixXd mm = RMatrixXd::Constant(nQSize,vW1[0].rows(),0);
    const int iHalfSize = vW1[0].rows()-1;
    for(const int query_term_id : query_term_ids){
        const VectorXd& query_term_vec = id2embedding_mm.at(query_term_id);
        for(const int doc_term_id : doc_term_ids){
            const VectorXd& doc_term_vec = id2embedding_mm.at(doc_term_id);
            double simi = query_term_vec.dot(doc_term_vec);
            for (int i = 0; i < iHalfSize; i++) {
                const double diff = -1.0 + 1.0 / iHalfSize
                                    + i * (2.0 / iHalfSize) - simi;
                mm(a, i) += exp(-100.0 * (diff) * (diff));
            }
            mm(a, iHalfSize) += exp(-1e6 * (1.0 - simi) * (1.0 - simi));
        }
    }
    double weight_sum = 0.0;
    VectorXd vQWeight = VectorXd::Zero(nQSize); //zi
    for(a = 0 ; a < nQSize; ++ a){
        double cqw = 2.0;
        cqw = exp(vW2(0)); 
        vQWeight(a) = cqw; 
        weight_sum += vQWeight(a);
    }

    assert(weight_sum != 0);
    vQWeight /= weight_sum;

    std::vector<VectorXd> vHi(vW1Size + 1);
    std::vector<VectorXd> sigma(vW1Size);
    vHi[0].resize(vW1[0].rows());
    for(a = 1 ; a < vW1Size + 1; ++ a){
        vHi[a].resize(vW1[a-1].cols()); 
        sigma[a-1].resize(vHi[a].size());
    }
    for(a = 0; a < mm.rows(); ++ a){
        vHi[0].setZero();
        for(int b = 0 ; b < vHi[0].size(); ++ b){
            vHi[0](b) = log10(mm(a,b) + 1);
        }
        //feed forward
        for(b = 1 ; b < vW1Size + 1; ++ b){ 
            vHi[b].setZero();
            vHi[b] = vHi[b-1] * vW1[b-1] + vW3[b-1];
            vHi[b] = Tanh(vHi[b]);
        }
        score += vHi[vW1Size](0) * vQWeight(a);
    }
    return score;
}

} // namespace nn4ir
