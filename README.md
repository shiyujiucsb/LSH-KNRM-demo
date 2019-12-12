
# Introduction
This directory contains an implementation of Kernel-pooling Neural Ranking Model (KNRM) using
Locality Sensitive Hashing (LSH) in C++.  The code assumes the raw data needed  are all available in memory and how to
access such data from a disk storage is discussed in the paper.  The LSH code for CONV-KNRM is similar.

The code  contains a test to report the rank scoring time  of the LSH-based KNRM for a single document given one query
using a randomzed data sample (query and document content, vocabulary embeddings, model parameters).
The total time of a query processing depends on the number of documents to rank and the other parameters
(e.g. query/document length).  The file  ``LSH_KNRM.h`` includes the default parameter setting, which can be adjusted.
The  test also reports the single-document scoring time  of an implementation of the original KNRM model
extended from a DRMM implementation.

# How to build
```
$ make
```

The build options are in ``Makefile``.

# How to run and test
```
$ ./LSH_KNRM_main
```

# Output sample
```
$ ./LSH_KNRM_main
Original time cost per document in ms: 2.539
LSH time cost per document in ms: 0.195
```

# Notes

* Reference for LSH speed-up:

Ji S, Shao J, Yang T. Efficient Interaction-based Neural Ranking with Locality Sensitive Hashing. In Proceedings of International World Wide Web Conference 2019 (WWW 2019) (pp. 2858-2864).

* KNRM Reference:

Xiong C, Dai Z, Callan J, Liu Z, Power R. End-to-end neural ad-hoc ranking with kernel pooling. In Proceedings of the 40th International ACM SIGIR conference on research and development in information retrieval 2017 Aug 7 (pp. 55-64). ACM.

* CONV-KNRM Reference:

Dai Z, Xiong C, Callan J, Liu Z. Convolutional neural networks for soft-matching n-grams in ad-hoc search. In Proceedings of the eleventh ACM international conference on web search and data mining 2018 Feb 2 (pp. 126-134). ACM.

* DRMM Reference:

Guo J, Fan Y, Ai Q, Croft WB. A deep relevance matching model for ad-hoc retrieval. In Proceedings of the 25th ACM International on Conference on Information and Knowledge Management 2016 Oct 24 (pp. 55-64). ACM.

* DRMM Code Reference: https://github.com/faneshion/DRMM
