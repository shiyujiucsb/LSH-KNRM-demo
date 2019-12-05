
# Introduction
This demo shows the speed-up of Kernel-pooling Neural Ranking Model (KNRM) using Locality Sensitive Hashing (LSH). 

It contains two baselines: 1. The original KNRM model. 2. The LSH-based KNRM model for speed up.

The time cost is computing the ranking score for one document and one query.
All the data (query and document content, vocabulary embeddings) and model (neural network parameters) are randomly generated on the fly.

By default, the number of LSH buckets is 256. The dimension of word embedding is 300. The KNRM model has 30 kernels. The query length is 3-word. The document length is 10000-word. The vocabulary size is 100K. Please check ``LSH_KNRM.h`` for these parameters.

# How to build
``
$ make
``

The build options are in ``Makefile``.

# How to run and test
``
$ ./LSH_KNRM_main
``

# Output sample
``
$ ./LSH_KNRM_main
Original time cost in ms: 51.826
LSH time cost in ms: 2.43
``

# Notes
* Need C++ 11.

* Reference:

@inproceedings{Ji:2019:EIN:3308558.3313576,

 author = {Ji, Shiyu and Shao, Jinjin and Yang, Tao},

 title = 
  {Efficient Interaction-based Neural Ranking with Locality Sensitive Hashing},

 booktitle = {The World Wide Web Conference},

 series = {WWW '19},

 year = {2019},

 isbn = {978-1-4503-6674-8},

 location = {San Francisco, CA, USA},

 pages = {2858--2864},

 numpages = {7},

 url = {http://doi.acm.org/10.1145/3308558.3313576},

 doi = {10.1145/3308558.3313576},

 acmid = {3313576},

 publisher = {ACM},

 address = {New York, NY, USA},
} 

* Code Reference: https://github.com/faneshion/DRMM
