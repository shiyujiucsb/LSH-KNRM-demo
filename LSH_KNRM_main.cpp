#include "LSH_KNRM.h"

int main() {

  /*
   * Input query terms, document terms, and word embedding.
   */

  // We randomly initialize one query and one document here,
  // to only demonstrate time cost reduction using LSH.
  // Query len and document length can be set in KNRM.h.
  std::vector<int> query = nn4ir::InitRandomQuery();
  std::vector<int> doc = nn4ir::InitRandomDocument();

  // Embedding is randomly generated also.
  std::vector<VectorXd> embeddings = nn4ir::InitRandomVectors(VOCAB_SIZE);

  /*
   * Transform word embedding to LSH vectors.
   */
  
  const std::vector<VectorXd> lsh_base_vectors = nn4ir::InitLSHBaseVectors();

  std::vector<LSHFingerprint> lsh_fingerprints;
  lsh_fingerprints.reserve(VOCAB_SIZE);
  for (int i = 0; i < VOCAB_SIZE; i++) {
    lsh_fingerprints.emplace_back(
        nn4ir::ComputeLSHFingerprint(embeddings[i], lsh_base_vectors));
  }

  /*
   * Evaluate ranking model with query and doc, and print time cost.
   */

  nn4ir::ComputeRankingScore(embeddings, query, doc);

  /*
   * Evaluate LSH ranking model with query and doc, and print time cost.
   */

  nn4ir::ComputeRankingScoreFromLSH(lsh_fingerprints, query, doc);

  return 0;
}
