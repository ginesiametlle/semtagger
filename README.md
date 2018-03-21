# semtagger

### About this repository

A universal semantic tagger.

### Training a neural model

```$ ./run.sh --train```

### Using a trained model

```$ ./run.sh --predict --input [INPUT_CONLL_FILE] --output [OUTPUT_CONLL_FILE]```

### Configuration

Please edit [config.sh](./config.sh) for fine-grained control.

### References

1. L. Abzianidze and J. Bos. [_Towards Universal Semantic Tagging_](http://www.aclweb.org/anthology/W17-6901). In Proceedings of the 12th International Conference on Computational Semantics (IWCS) - Short papers. Association for Computational Linguistics, 2017.

2. J. Bjerva, B. Plank and J. Bos. [_Semantic Tagging with Deep Residual Networks_](http://aclweb.org/anthology/C16-1333). In Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers, pages 3531–3541. Association for Computational Linguistics, 2016.
