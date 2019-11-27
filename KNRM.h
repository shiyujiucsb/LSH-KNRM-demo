/*
 * Demo: Use LSH to speed up the evaluation of KNRM ranking model.
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

constexpr int QUERY_LEN = 3;

constexpr int DOC_LEN = 10000;

constexpr int VOCAB_SIZE = 100000;


namespace nn4ir {

const unsigned int SEED =
    std::chrono::system_clock::now().time_since_epoch().count();

VectorXd Tanh(const VectorXd& input);

/*
 * Compute LSH fingerprint.
 */
LSHFingerprint ComputeLSHFingerprint(
    const VectorXd& vec,
    const std::vector<VectorXd>& lsh_random_vectors);

/*
 * Initialize num_vec random vectors with length DIM.
 */
std::vector<VectorXd> InitRandomVectors(int num_vec);

/*
 * Initialize randoms vectors for LSH fingerprint generation.
 */
inline std::vector<VectorXd> InitLSHBaseVectors() {
  return InitRandomVectors(N_LSH_BITS);
}

inline std::vector<int> InitRandomQuery() {
  srand(nn4ir::SEED);
  std::vector<int> query(QUERY_LEN);
  std::generate(query.begin(), query.end(),
                [](){
                  return rand() % VOCAB_SIZE; 
                });
  return query;
}

inline std::vector<int> InitRandomDocument() {
  srand(nn4ir::SEED);
  std::vector<int> doc(DOC_LEN);
  std::generate(doc.begin(), doc.end(),
                [](){
                  return rand() % VOCAB_SIZE; 
                });
  return doc;
}

/*
 * Initialize LSH matrix for histogram-to-kernel pooling transition.
 */
RMatrixXd InitLSHMatrix();

/*
 * Compute ranking score based on original embeddings.
 */
double ComputeRankingScore(
    const std::vector<VectorXd>& id2embedding_mm,
    const std::vector<int>& query_term_ids,
    const std::vector<int>& doc_term_ids);

/*
 * Compute ranking score based on LSH fingerprints.
 */ 
double ComputeRankingScoreFromLSH(
    const std::vector<LSHFingerprint>& id2lsh_mm,
    const std::vector<int>& query_term_ids,
    const std::vector<int>& doc_term_ids);

} // namespace nn4ir

#endif
