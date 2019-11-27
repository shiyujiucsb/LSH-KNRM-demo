#include "LSH_KNRM.h"

namespace nn4ir {

VectorXd Tanh(const VectorXd& input){
  VectorXd voutput = VectorXd::Zero(input.size());
  for(int i = 0 ; i < voutput.size(); ++i){
    const double exp_val = exp( - 2.0 * input[i] );
    voutput[i] = (1.0 - exp_val) / (1.0 + exp_val);
  }
  return voutput;
}

LSHFingerprint ComputeLSHFingerprint(
    const VectorXd& vec,
    const std::vector<VectorXd>& lsh_base_vectors) {
  assert(lsh_base_vectors.size() == N_LSH_BITS);
  LSHFingerprint result(0);
  for (const VectorXd& r : lsh_base_vectors) {
    if (vec.dot(r) > 0) {
      result.set(0);
    }
    result <<= 1;
  }
  return result;
}

std::vector<VectorXd> InitRandomVectors(int num_vec) {
  std::default_random_engine generator(SEED);
  std::normal_distribution<double> distri(0.0, 1.0);

  std::vector<VectorXd> vectors;
  vectors.reserve(num_vec);
  for (int i = 0; i < num_vec; i++) {
    VectorXd rand_vec(DIM);
    for (int j = 0; j < DIM; j++) {
      rand_vec[j] = distri(generator);
    }
    vectors.emplace_back(rand_vec / rand_vec.norm());
  }

  return vectors;
}

RMatrixXd InitLSHMatrix() {
  RMatrixXd M(N_KERNELS, N_LSH_BITS + 1);
  for (int i = 0; i <= N_LSH_BITS; i++) {
    double simi = cos(i * 1.0 / N_LSH_BITS * 3.1415926);
    for (int j = 0; j < (N_KERNELS - 1); j++) {
      const double diff = -1.0 + 1.0 / (N_KERNELS - 1)
                          + j * (2.0 / (N_KERNELS - 1)) - simi;
      M(j, i) = exp(-100.0 * (diff) * (diff));
    }
    M(N_KERNELS - 1, i) += exp(-1e6 * (1.0 - simi) * (1.0 - simi));
  }

  return std::move(M);
}

double ComputeRankingScore(const std::vector<VectorXd>& id2embedding_mm,
                           const std::vector<int>& query_term_ids,
                           const std::vector<int>& doc_term_ids) {
  /*
   * Randomly initialize neural network parameters of KNRM.
   * Again, this is only for demonstrating time cost reduction using LSH.
   * Feel free to replace this model and use a pre-trained model.
   */

  // How many layers.
  constexpr int vW1_size = 2;
  // Size of each layer including input, which equals to the kernel size.
  constexpr int vW_size_info[] = {N_KERNELS, 5, 1};

  std::vector<RMatrixXd> vW1(vW1_size);
  std::vector<VectorXd> vW3(vW1_size);
  const VectorXd vW2 = VectorXd::Random(1) / 100.0;

  for (int i = 0; i < vW1_size; i++) {
    vW1[i] = RMatrixXd::Random(vW_size_info[i], vW_size_info[i+1]) / 10.0;
    vW3[i] = VectorXd::Random(vW_size_info[i+1]) / 10.0;
  }

  /*
   * Begin online evaluation for input query and document.
   */

  const auto original_start = std::chrono::high_resolution_clock::now();

  int64_t a,b,c,d,e;
  constexpr double eps = 1e-4;
  double score = 0;
  const int nQSize = query_term_ids.size();
  const int vW1Size = vW1.size();
  const int iDocLen = doc_term_ids.size();

  if (iDocLen == 0 || nQSize == 0) {
    return std::numeric_limits<double>::lowest();
  }

  RMatrixXd mm = RMatrixXd::Constant(nQSize,vW1[0].rows(),0);
  const int iHalfSize = vW1[0].rows()-1;
  for (int a = 0; a < nQSize; ++a) {
    const VectorXd& query_term_vec = id2embedding_mm.at(query_term_ids[a]);
    for (const int doc_term_id : doc_term_ids) {
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
  for (a = 0 ; a < nQSize; ++a) {
    double cqw = 2.0;
    cqw = exp(vW2(0)); 
    vQWeight(a) = cqw; 
    weight_sum += vQWeight(a);
  }

  assert(weight_sum != 0);
  vQWeight /= weight_sum;

  std::vector<VectorXd> vHi(vW1Size + 1);
  vHi[0].resize(vW1[0].rows());
  for (a = 1; a < vW1Size + 1; ++a) {
    vHi[a].resize(vW1[a-1].cols()); 
  }
  for (a = 0; a < mm.rows(); ++a) {
    vHi[0].setZero();
    for (int b = 0 ; b < vHi[0].size(); ++ b) {
      vHi[0](b) = log10(mm(a,b) + 1);
    }
    //feed forward
    for (b = 1 ; b < vW1Size + 1; ++b) { 
      vHi[b].setZero();
      vHi[b] = vHi[b-1] * vW1[b-1] + vW3[b-1];
      vHi[b] = Tanh(vHi[b]);
    }
    score += vHi[vW1Size](0) * vQWeight(a);
  }

  const auto original_elapse = std::chrono::high_resolution_clock::now()
                               - original_start;

  const int64_t original_time =
      std::chrono::duration_cast<std::chrono::microseconds>(original_elapse)
      .count();
  std::cout << "Original time cost in ms: "
            << original_time / 1000.0 << std::endl;

  return score;
}

double ComputeRankingScoreFromLSH(const std::vector<LSHFingerprint>& id2lsh_mm,
                                  const std::vector<int>& query_term_ids,
                                  const std::vector<int>& doc_term_ids) {
  /*
   * Randomly initialize neural network parameters of KNRM.
   * Again, this is only for demonstrating time cost reduction using LSH.
   * Feel free to replace this model and use a pre-trained model.
   */

  // How many layers.
  constexpr int vW1_size = 2;
  // Size of each layer including input, which equals to the kernel size.
  constexpr int vW_size_info[] = {N_KERNELS, 5, 1};

  std::vector<RMatrixXd> vW1(vW1_size);
  std::vector<VectorXd> vW3(vW1_size);
  const VectorXd vW2 = VectorXd::Random(1) / 100.0;

  for (int i = 0; i < vW1_size; i++) {
    vW1[i] = RMatrixXd::Random(vW_size_info[i], vW_size_info[i+1]) / 10.0;
    vW3[i] = VectorXd::Random(vW_size_info[i+1]) / 10.0;
  }

  // Since we know kernel size, lambda, sigma in a KNRM model,
  // we can pre-compute kernel values for each LSH bucket.
  // This is a key point for time cost reduction with LSH.
  const RMatrixXd lsh_matrix = InitLSHMatrix();

  /*
   * Begin online evaluation for input query and document.
   */
  
  const auto lsh_start = std::chrono::high_resolution_clock::now();

  int64_t a,b,c,d,e;
  constexpr double eps = 1e-4;
  double score = 0;
  const int nQSize = query_term_ids.size();
  const int vW1Size = vW1.size();
  const int iDocLen = doc_term_ids.size();

  if (iDocLen == 0 || nQSize == 0) {
    return std::numeric_limits<double>::lowest();
  }
  RMatrixXd mm = RMatrixXd::Constant(nQSize,vW1[0].rows(), 0);
  const int iHalfSize = vW1[0].rows() - 1;
  for (int a = 0; a < nQSize; ++a) {
    const LSHFingerprint& query_term_vec = id2lsh_mm.at(query_term_ids[a]);
    VectorXd hamming_count = VectorXd::Zero(N_LSH_BITS + 1);
    for (const int doc_term_id : doc_term_ids) {
      const LSHFingerprint& doc_term_vec = id2lsh_mm.at(doc_term_id);
      const LSHFingerprint x = query_term_vec ^ doc_term_vec;
     hamming_count[x.count()] += 1;
    }
    mm.row(a) += lsh_matrix * hamming_count.transpose();
  }
  double weight_sum = 0.0;
  VectorXd vQWeight = VectorXd::Zero(nQSize); //zi
  for (a = 0; a < nQSize; ++a) {
    double cqw = 2.0;
    cqw = exp(vW2(0)); 
    vQWeight(a) = cqw; 
    weight_sum += vQWeight(a);
  }

  assert(weight_sum != 0);
  vQWeight /= weight_sum;

  std::vector<VectorXd> vHi(vW1Size + 1);
  vHi[0].resize(vW1[0].rows());
  for (a = 1; a < vW1Size + 1; ++a) {
    vHi[a].resize(vW1[a-1].cols()); 
  }
  for (a = 0; a < mm.rows(); ++a) {
    vHi[0].setZero();
    for (int b = 0 ; b < vHi[0].size(); ++b) {
      vHi[0](b) = log10(mm(a,b) + 1);
    }
    //feed forward
    for (b = 1 ; b < vW1Size + 1; ++b) {
      vHi[b].setZero();
      vHi[b] = vHi[b-1] * vW1[b-1] + vW3[b-1];
      vHi[b] = Tanh(vHi[b]);
    }
    score += vHi[vW1Size](0) * vQWeight(a);
  }

  const auto lsh_elapse = std::chrono::high_resolution_clock::now()
                          - lsh_start;
  const int64_t lsh_time =
      std::chrono::duration_cast<std::chrono::microseconds>(lsh_elapse)
      .count();
  std::cout << "LSH time cost in ms: "
            << lsh_time / 1000.0 << std::endl;

  return score;
}

} // namespace nn4ir
