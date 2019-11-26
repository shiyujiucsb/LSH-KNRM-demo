#include "KNRM.h"

int main() {
  const std::vector<VectorXd> lsh_random_vectors =
                                     nn4ir::InitRandomVectors(N_LSH_BITS);
  const RMatrixXd lsh_matrix = nn4ir::InitLSHMatrix();

  constexpr int QUERY_LEN = 3;
  constexpr int DOC_LEN = 10000;
  constexpr int VOCAB_SIZE = 10000;

  srand(nn4ir::SEED);
  std::vector<int> query(QUERY_LEN);
  std::generate(query.begin(), query.end(), [](){
		                 return rand() % VOCAB_SIZE; });
  std::vector<int> doc(DOC_LEN);
  std::generate(doc.begin(), doc.end(), [](){
		                 return rand() % VOCAB_SIZE; });

  std::vector<VectorXd> embeddings = nn4ir::InitRandomVectors(VOCAB_SIZE);

  std::vector<LSHFingerprint> lsh_fingerprints;
  lsh_fingerprints.reserve(VOCAB_SIZE);
  for (int i = 0; i < VOCAB_SIZE; i++) {
    lsh_fingerprints.emplace_back(nn4ir::ComputeLSHFingerprint(
                                                      embeddings[i],
                                                      lsh_random_vectors));
  }

  constexpr int vW1_size = 2;
  constexpr int vW_size_info[] = {N_KERNELS, 5, 1};
  std::vector<RMatrixXd> vW1(vW1_size);
  std::vector<VectorXd> vW3(vW1_size);
  const VectorXd vW2 = VectorXd::Random(2) / 100.0;
  for (int i = 0; i < vW1_size; i++) {
    vW1[i] = RMatrixXd::Random(vW_size_info[i], vW_size_info[i+1]) / 10.0;
    vW3[i] = VectorXd::Random(vW_size_info[i+1]) / 10.0;
  }

  const auto original_start = std::chrono::high_resolution_clock::now();
  nn4ir::ComputeRankingScore(embeddings, query, doc, vW1, vW2, vW3);
  const auto original_elapse = std::chrono::high_resolution_clock::now()
                                  - original_start;
  const int64_t original_time =
      std::chrono::duration_cast<std::chrono::microseconds>(original_elapse)
          .count();

  const auto lsh_start = std::chrono::high_resolution_clock::now();
  nn4ir::ComputeRankingScoreFromLSH(lsh_fingerprints, lsh_matrix,
                              query, doc, vW1, vW2, vW3);
  const auto lsh_elapse = std::chrono::high_resolution_clock::now()
                                  - lsh_start;
  const int64_t lsh_time =
      std::chrono::duration_cast<std::chrono::microseconds>(lsh_elapse)
          .count();

  std::cout << "Original time cost in ms:" << std::endl;
  std::cout << original_time / 1000.0 << std::endl;
  std::cout << "LSH time cost in ms:" << std::endl;
  std::cout << lsh_time / 1000.0 << std::endl;

  return 0;
}
