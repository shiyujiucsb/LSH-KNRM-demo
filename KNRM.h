/*
 * Implementation of KNRM ranking model.
*/

#ifndef _KNRM_H
#define _KNRM_H

#include <bitset>
#include <random>
#include <chrono>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <memory>
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
constexpr int DIM = 300;

// Number of soft kernels.
constexpr int N_KERNELS = 30;

namespace nn4ir {

const unsigned SEED =
               std::chrono::system_clock::now().time_since_epoch().count();

VectorXd Tanh(const VectorXd& input);

/*
 * Compute LSH fingerprint.
 * */
LSHFingerprint ComputeLSHFingerprint(
                             const VectorXd& vec,
                             const std::vector<VectorXd>& lsh_random_vectors);

/*
 * Initialize random vectors with length one.
 * */
std::vector<VectorXd> InitRandomVectors(int n_vectors);

/*
 * Initialize LSH matrix for histogram-to-kernel pooling transition.
 * */
RMatrixXd InitLSHMatrix();

/*
 * Compute ranking score based on original embeddings.
 * */
double ComputeRankingScore(
                const std::vector<VectorXd>& id2embedding_mm,
                const std::vector<int>& query_term_ids,
                const std::vector<int>& doc_term_ids,
                const std::vector<RMatrixXd>& vW1,
                const VectorXd& vW2,
                const std::vector<VectorXd>& vW3);

/*
 * Compute ranking score based on LSH fingerprints.
 * */
double ComputeRankingScoreFromLSH(
                const std::vector<LSHFingerprint>& id2lsh_mm,
                const RMatrixXd& lsh_matrix,
                const std::vector<int>& query_term_ids,
                const std::vector<int>& doc_term_ids,
                const std::vector<RMatrixXd>& vW1,
                const VectorXd& vW2,
                const std::vector<VectorXd>& vW3);

} // namespace nn4ir

#endif

