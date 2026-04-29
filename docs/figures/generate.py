import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# The raw string was truncated, so it was completed syntactically.
raw_json = """{
  "experiment": "exp2_scale_vs_weight",
  "runs": [
    {
      "model": "qwen_fp16",
      "quant": "FP16",
      "condition": "weights",
      "flip_count": 1,
      "seed": 0,
      "ppl": 2719119.8138,
      "arc_easy": 0.25715488215488214,
      "hellaswag": 0.2812,
      "error": null
    },
    {
      "model": "qwen_fp16",
      "quant": "FP16",
      "condition": "weights",
      "flip_count": 5,
      "seed": 0,
      "ppl": Infinity,
      "arc_easy": 0.24242424242424243,
      "hellaswag": 0.2916,
      "error": null
    },
    {
      "model": "qwen_fp16",
      "quant": "FP16",
      "condition": "weights",
      "flip_count": 10,
      "seed": 0,
      "ppl": 2.0138564133336666e+163,
      "arc_easy": 0.2542087542087542,
      "hellaswag": 0.2624,
      "error": null
    },
    {
      "model": "qwen_fp16",
      "quant": "FP16",
      "condition": "weights",
      "flip_count": 1,
      "seed": 1,
      "ppl": 2729647.599,
      "arc_easy": 0.26052188552188554,
      "hellaswag": 0.2792,
      "error": null
    },
    {
      "model": "qwen_fp16",
      "quant": "FP16",
      "condition": "weights",
      "flip_count": 5,
      "seed": 1,
      "ppl": 2728664.3306,
      "arc_easy": 0.2596801346801347,
      "hellaswag": 0.28,
      "error": null
    },
    {
      "model": "qwen_fp16",
      "quant": "FP16",
      "condition": "weights",
      "flip_count": 10,
      "seed": 1,
      "ppl": 71502.5427,
      "arc_easy": 0.2916666666666667,
      "hellaswag": 0.2884,
      "error": null
    },
    {
      "model": "qwen_fp16",
      "quant": "FP16",
      "condition": "weights",
      "flip_count": 1,
      "seed": 2,
      "ppl": 2729935.54,
      "arc_easy": 0.26136363636363635,
      "hellaswag": 0.2796,
      "error": null
    },
    {
      "model": "qwen_fp16",
      "quant": "FP16",
      "condition": "weights",
      "flip_count": 5,
      "seed": 2,
      "ppl": 2760271.2183,
      "arc_easy": 0.2622053872053872,
      "hellaswag": 0.2804,
      "error": null
    },
    {
      "model": "qwen_fp16",
      "quant": "FP16",
      "condition": "weights",
      "flip_count": 10,
      "seed": 2,
      "ppl": 2762990.2772,
      "arc_easy": 0.25841750841750843,
      "hellaswag": 0.2808,
      "error": null
    },
    {
      "model": "qwen_q8",
      "quant": "Q8_0",
      "condition": "weights",
      "flip_count": 1,
      "seed": 0,
      "ppl": 15.9516,
      "arc_easy": 0.6620370370370371,
      "hellaswag": 0.3784,
      "error": null
    },
    {
      "model": "qwen_q8",
      "quant": "Q8_0",
      "condition": "weights",
      "flip_count": 5,
      "seed": 0,
      "ppl": 15.9527,
      "arc_easy": 0.6628787878787878,
      "hellaswag": 0.378,
      "error": null
    },
    {
      "model": "qwen_q8",
      "quant": "Q8_0",
      "condition": "weights",
      "flip_count": 10,
      "seed": 0,
      "ppl": 15.9529,
      "arc_easy": 0.6582491582491582,
      "hellaswag": 0.3784,
      "error": null
    },
    {
      "model": "qwen_q8",
      "quant": "Q8_0",
      "condition": "weights",
      "flip_count": 1,
      "seed": 1,
      "ppl": 15.9558,
      "arc_easy": 0.6645622895622896,
      "hellaswag": 0.3784,
      "error": null
    },
    {
      "model": "qwen_q8",
      "quant": "Q8_0",
      "condition": "weights",
      "flip_count": 5,
      "seed": 1,
      "ppl": 15.9547,
      "arc_easy": 0.6645622895622896,
      "hellaswag": 0.3788,
      "error": null
    },
    {
      "model": "qwen_q8",
      "quant": "Q8_0",
      "condition": "weights",
      "flip_count": 10,
      "seed": 1,
      "ppl": 15.9528,
      "arc_easy": 0.6611952861952862,
      "hellaswag": 0.3792,
      "error": null
    },
    {
      "model": "qwen_q8",
      "quant": "Q8_0",
      "condition": "weights",
      "flip_count": 1,
      "seed": 2,
      "ppl": 15.9555,
      "arc_easy": 0.6641414141414141,
      "hellaswag": 0.3788,
      "error": null
    },
    {
      "model": "qwen_q8",
      "quant": "Q8_0",
      "condition": "weights",
      "flip_count": 5,
      "seed": 2,
      "ppl": 15.9515,
      "arc_easy": 0.6645622895622896,
      "hellaswag": 0.3792,
      "error": null
    },
    {
      "model": "qwen_q8",
      "quant": "Q8_0",
      "condition": "weights",
      "flip_count": 10,
      "seed": 2,
      "ppl": 15.9555,
      "arc_easy": 0.6582491582491582,
      "hellaswag": 0.3788,
      "error": null
    },
    {
      "model": "qwen_q8",
      "quant": "Q8_0",
      "condition": "block_scale",
      "flip_count": 1,
      "seed": 0,
      "ppl": 15.9548,
      "arc_easy": 0.6637205387205387,
      "hellaswag": 0.3784,
      "error": null
    },
    {
      "model": "qwen_q8",
      "quant": "Q8_0",
      "condition": "block_scale",
      "flip_count": 5,
      "seed": 0,
      "ppl": Infinity,
      "arc_easy": 0.255050505050505,
      "hellaswag": 0.2668,
      "error": null
    },
    {
      "model": "qwen_q8",
      "quant": "Q8_0",
      "condition": "block_scale",
      "flip_count": 10,
      "seed": 0,
      "ppl": Infinity,
      "arc_easy": 0.26346801346801346,
      "hellaswag": 0.2552,
      "error": null
    },
    {
      "model": "qwen_q8",
      "quant": "Q8_0",
      "condition": "block_scale",
      "flip_count": 1,
      "seed": 1,
      "ppl": 15.9558,
      "arc_easy": 0.6645622895622896,
      "hellaswag": 0.3784,
      "error": null
    },
    {
      "model": "qwen_q8",
      "quant": "Q8_0",
      "condition": "block_scale",
      "flip_count": 5,
      "seed": 1,
      "ppl": 15.9541,
      "arc_easy": 0.6637205387205387,
      "hellaswag": 0.3796,
      "error": null
    },
    {
      "model": "qwen_q8",
      "quant": "Q8_0",
      "condition": "block_scale",
      "flip_count": 10,
      "seed": 1,
      "ppl": 7.867783797685577e+39,
      "arc_easy": 0.24705387205387205,
      "hellaswag": 0.2652,
      "error": null
    },
    {
      "model": "qwen_q8",
      "quant": "Q8_0",
      "condition": "block_scale",
      "flip_count": 1,
      "seed": 2,
      "ppl": 15.9555,
      "arc_easy": 0.6616161616161617,
      "hellaswag": 0.3784,
      "error": null
    },
    {
      "model": "qwen_q8",
      "quant": "Q8_0",
      "condition": "block_scale",
      "flip_count": 5,
      "seed": 2,
      "ppl": 16.1432,
      "arc_easy": 0.6515151515151515,
      "hellaswag": 0.376,
      "error": null
    },
    {
      "model": "qwen_q8",
      "quant": "Q8_0",
      "condition": "block_scale",
      "flip_count": 10,
      "seed": 2,
      "ppl": 16.2839,
      "arc_easy": 0.6536195286195287,
      "hellaswag": 0.3748,
      "error": null
    },
    {
      "model": "qwen_q4km",
      "quant": "Q4_K_M",
      "condition": "weights",
      "flip_count": 1,
      "seed": 0,
      "ppl": 16.4287,
      "arc_easy": 0.6574074074074074,
      "hellaswag": 0.3728,
      "error": null
    },
    {
      "model": "qwen_q4km",
      "quant": "Q4_K_M",
      "condition": "weights",
      "flip_count": 5,
      "seed": 0,
      "ppl": 16.4277,
      "arc_easy": 0.6599326599326599,
      "hellaswag": 0.3732,
      "error": null
    },
    {
      "model": "qwen_q4km",
      "quant": "Q4_K_M",
      "condition": "weights",
      "flip_count": 10,
      "seed": 0,
      "ppl": 16.4251,
      "arc_easy": 0.6578282828282829,
      "hellaswag": 0.3732,
      "error": null
    },
    {
      "model": "qwen_q4km",
      "quant": "Q4_K_M",
      "condition": "weights",
      "flip_count": 1,
      "seed": 1,
      "ppl": 16.4267,
      "arc_easy": 0.6595117845117845,
      "hellaswag": 0.3732,
      "error": null
    },
    {
      "model": "qwen_q4km",
      "quant": "Q4_K_M",
      "condition": "weights",
      "flip_count": 5,
      "seed": 1,
      "ppl": 16.4267,
      "arc_easy": 0.6595117845117845,
      "hellaswag": 0.3732,
      "error": null
    },
    {
      "model": "qwen_q4km",
      "quant": "Q4_K_M",
      "condition": "weights",
      "flip_count": 10,
      "seed": 1,
      "ppl": 16.4251,
      "arc_easy": 0.6578282828282829,
      "hellaswag": 0.3728,
      "error": null
    },
    {
      "model": "qwen_q4km",
      "quant": "Q4_K_M",
      "condition": "weights",
      "flip_count": 1,
      "seed": 2,
      "ppl": 16.4257,
      "arc_easy": 0.6586700336700336,
      "hellaswag": 0.3732,
      "error": null
    },
    {
      "model": "qwen_q4km",
      "quant": "Q4_K_M",
      "condition": "weights",
      "flip_count": 5,
      "seed": 2,
      "ppl": 16.4257,
      "arc_easy": 0.6586700336700336,
      "hellaswag": 0.3732,
      "error": null
    },
    {
      "model": "qwen_q4km",
      "quant": "Q4_K_M",
      "condition": "weights",
      "flip_count": 10,
      "seed": 2,
      "ppl": 16.4257,
      "arc_easy": 0.6586700336700336,
      "hellaswag": 0.3732,
      "error": null
    },
    {
      "model": "qwen_q4km",
      "quant": "Q4_K_M",
      "condition": "block_scale",
      "flip_count": 1,
      "seed": 0,
      "ppl": 16.4269,
      "arc_easy": 0.6565656565656566,
      "hellaswag": 0.3728,
      "error": null
    },
    {
      "model": "qwen_q4km",
      "quant": "Q4_K_M",
      "condition": "block_scale",
      "flip_count": 5,
      "seed": 0,
      "ppl": 16.4269,
      "arc_easy": 0.6586700336700336,
      "hellaswag": 0.372,
      "error": null
    },
    {
      "model": "qwen_q4km",
      "quant": "Q4_K_M",
      "condition": "block_scale",
      "flip_count": 10,
      "seed": 0,
      "ppl": 16.4315,
      "arc_easy": 0.6582491582491582,
      "hellaswag": 0.3736,
      "error": null
    },
    {
      "model": "qwen_q4km",
      "quant": "Q4_K_M",
      "condition": "block_scale",
      "flip_count": 1,
      "seed": 1,
      "ppl": 16.4272,
      "arc_easy": 0.6582491582491582,
      "hellaswag": 0.3736,
      "error": null
    },
    {
      "model": "qwen_q4km",
      "quant": "Q4_K_M",
      "condition": "block_scale",
      "flip_count": 5,
      "seed": 1,
      "ppl": 16.4308,
      "arc_easy": 0.6557239057239057,
      "hellaswag": 0.3728,
      "error": null
    },
    {
      "model": "qwen_q4km",
      "quant": "Q4_K_M",
      "condition": "block_scale",
      "flip_count": 10,
      "seed": 1,
      "ppl": 16.426,
      "arc_easy": 0.6565656565656566,
      "hellaswag": 0.374,
      "error": null
    },
    {
      "model": "qwen_q4km",
      "quant": "Q4_K_M",
      "condition": "block_scale",
      "flip_count": 1,
      "seed": 2,
      "ppl": 16.4273,
      "arc_easy": 0.6599326599326599,
      "hellaswag": 0.3744,
      "error": null
    },
    {
      "model": "qwen_q4km",
      "quant": "Q4_K_M",
      "condition": "block_scale",
      "flip_count": 5,
      "seed": 2,
      "ppl": 16.4252,
      "arc_easy": 0.6586700336700336,
      "hellaswag": 0.3732,
      "error": null
    },
    {
      "model": "qwen_q4km",
      "quant": "Q4_K_M",
      "condition": "block_scale",
      "flip_count": 10,
      "seed": 2,
      "ppl": 16.4266,
      "arc_easy": 0.6578282828282829,
      "hellaswag": 0.3732,
      "error": null
    },
    {
      "model": "qwen_q4km",
      "quant": "Q4_K_M",
      "condition": "super_scale",
      "flip_count": 1,
      "seed": 0,
      "ppl": 17.4101,
      "arc_easy": 0.6342592592592593,
      "hellaswag": 0.3644,
      "error": null
    },
    {
      "model": "qwen_q4km",
      "quant": "Q4_K_M",
      "condition": "super_scale",
      "flip_count": 5,
      "seed": 0,
      "ppl": 233269177094.6306,
      "arc_easy": 0.24031986531986532,
      "hellaswag": 0.2756,
      "error": null
    },
    {
      "model": "qwen_q4km",
      "quant": "Q4_K_M",
      "condition": "super_scale",
      "flip_count": 10,
      "seed": 0,
      "ppl": 216265599.8843,
      "arc_easy": 0.2601010101010101,
      "hellaswag": 0.2724,
      "error": null
    },
    {
      "model": "qwen_q4km",
      "quant": "Q4_K_M",
      "condition": "super_scale",
      "flip_count": 1,
      "seed": 1,
      "ppl": 16.431,
      "arc_easy": 0.6603535353535354,
      "hellaswag": 0.3728,
      "error": null
    },
    {
      "model": "qwen_q4km",
      "quant": "Q4_K_M",
      "condition": "super_scale",
      "flip_count": 5,
      "seed": 1,
      "ppl": 16.4281,
      "arc_easy": 0.656986531986532,
      "hellaswag": 0.3732,
      "error": null
    },
    {
      "model": "qwen_q4km",
      "quant": "Q4_K_M",
      "condition": "super_scale",
      "flip_count": 10,
      "seed": 1,
      "ppl": 18187538.4635,
      "arc_easy": 0.25925925925925924,
      "hellaswag": 0.2784,
      "error": null
    },
    {
      "model": "qwen_q4km",
      "quant": "Q4_K_M",
      "condition": "super_scale",
      "flip_count": 1,
      "seed": 2,
      "ppl": 16.4396,
      "arc_easy": 0.6590909090909091,
      "hellaswag": 0.3732,
      "error": null
    },
    {
      "model": "qwen_q4km",
      "quant": "Q4_K_M",
      "condition": "super_scale",
      "flip_count": 5,
      "seed": 2,
      "ppl": 407.3135,
      "arc_easy": 0.38215488215488214,
      "hellaswag": 0.324,
      "error": null
    },
    {
      "model": "qwen_q4km",
      "quant": "Q4_K_M",
      "condition": "super_scale",
      "flip_count": 10,
      "seed": 2,
      "ppl": 20945387.8823,
      "arc_easy": 0.24031986531986532,
      "hellaswag": 0.2784,
      "error": null
    },
    {
      "model": "qwen_q4",
      "quant": "Q4_0",
      "condition": "weights",
      "flip_count": 1,
      "seed": 0,
      "ppl": 17.5369,
      "arc_easy": 0.6207912457912458,
      "hellaswag": 0.3768,
      "error": null
    },
    {
      "model": "qwen_q4",
      "quant": "Q4_0",
      "condition": "weights",
      "flip_count": 5,
      "seed": 0,
      "ppl": 17.5378,
      "arc_easy": 0.617003367003367,
      "hellaswag": 0.3772,
      "error": null
    },
    {
      "model": "qwen_q4",
      "quant": "Q4_0",
      "condition": "weights",
      "flip_count": 10,
      "seed": 0,
      "ppl": 17.5401,
      "arc_easy": 0.6178451178451179,
      "hellaswag": 0.3752,
      "error": null
    },
    {
      "model": "qwen_q4",
      "quant": "Q4_0",
      "condition": "weights",
      "flip_count": 1,
      "seed": 1,
      "ppl": 17.542,
      "arc_easy": 0.6157407407407407,
      "hellaswag": 0.376,
      "error": null
    },
    {
      "model": "qwen_q4",
      "quant": "Q4_0",
      "condition": "weights",
      "flip_count": 5,
      "seed": 1,
      "ppl": 17.5416,
      "arc_easy": 0.6178451178451179,
      "hellaswag": 0.376,
      "error": null
    },
    {
      "model": "qwen_q4",
      "quant": "Q4_0",
      "condition": "weights",
      "flip_count": 10,
      "seed": 1,
      "ppl": 17.5411,
      "arc_easy": 0.617003367003367,
      "hellaswag": 0.3756,
      "error": null
    },
    {
      "model": "qwen_q4",
      "quant": "Q4_0",
      "condition": "weights",
      "flip_count": 1,
      "seed": 2,
      "ppl": 17.5412,
      "arc_easy": 0.6195286195286195,
      "hellaswag": 0.376,
      "error": null
    },
    {
      "model": "qwen_q4",
      "quant": "Q4_0",
      "condition": "weights",
      "flip_count": 5,
      "seed": 2,
      "ppl": 17.541,
      "arc_easy": 0.617003367003367,
      "hellaswag": 0.3756,
      "error": null
    },
    {
      "model": "qwen_q4",
      "quant": "Q4_0",
      "condition": "weights",
      "flip_count": 10,
      "seed": 2,
      "ppl": 17.5375,
      "arc_easy": 0.6161616161616161,
      "hellaswag": 0.3764,
      "error": null
    },
    {
      "model": "qwen_q4",
      "quant": "Q4_0",
      "condition": "block_scale",
      "flip_count": 1,
      "seed": 0,
      "ppl": 17.4951,
      "arc_easy": 0.6161616161616161,
      "hellaswag": 0.3756,
      "error": null
    },
    {
      "model": "qwen_q4",
      "quant": "Q4_0",
      "condition": "block_scale",
      "flip_count": 5,
      "seed": 0,
      "ppl": Infinity,
      "arc_easy": 0.25126262626262624,
      "hellaswag": 0.2652,
      "error": null
    },
    {
      "model": "qwen_q4",
      "quant": "Q4_0",
      "condition": "block_scale",
      "flip_count": 10,
      "seed": 0,
      "ppl": Infinity,
      "arc_easy": 0.2676767676767677,
      "hellaswag": 0.2676,
      "error": null
    },
    {
      "model": "qwen_q4",
      "quant": "Q4_0",
      "condition": "block_scale",
      "flip_count": 1,
      "seed": 1,
      "ppl": 17.542,
      "arc_easy": 0.6157407407407407,
      "hellaswag": 0.376,
      "error": null
    },
    {
      "model": "qwen_q4",
      "quant": "Q4_0",
      "condition": "block_scale",
      "flip_count": 5,
      "seed": 1,
      "ppl": 17.5402,
      "arc_easy": 0.6153198653198653,
      "hellaswag": 0.3752,
      "error": null
    },
    {
      "model": "qwen_q4",
      "quant": "Q4_0",
      "condition": "block_scale",
      "flip_count": 10,
      "seed": 1,
      "ppl": 9.330051562389421e+48,
      "arc_easy": 0.24957912457912457,
      "hellaswag": 0.2624,
      "error": null
    },
    {
      "model": "qwen_q4",
      "quant": "Q4_0",
      "condition": "block_scale",
      "flip_count": 1,
      "seed": 2,
      "ppl": 17.5389,
      "arc_easy": 0.6153198653198653,
      "hellaswag": 0.376,
      "error": null
    },
    {
      "model": "qwen_q4",
      "quant": "Q4_0",
      "condition": "block_scale",
      "flip_count": 5,
      "seed": 2,
      "ppl": 17.7944,
      "arc_easy": 0.6115319865319865,
      "hellaswag": 0.3732,
      "error": null
    },
    {
      "model": "qwen_q4",
      "quant": "Q4_0",
      "condition": "block_scale",
      "flip_count": 10,
      "seed": 2,
      "ppl": null,
      "arc_easy": 0.6254208754208754,
      "hellaswag": 0.374,
      "error": null
    },
    {
      "model": "llama_fp16",
      "quant": "FP16",
      "condition": "weights",
      "flip_count": 1,
      "seed": 0,
      "ppl": 13.8843,
      "arc_easy": 0.6910774410774411,
      "hellaswag": 0.412,
      "error": null
    },
    {
      "model": "llama_fp16",
      "quant": "FP16",
      "condition": "weights",
      "flip_count": 5,
      "seed": 0,
      "ppl": 2092.3283,
      "arc_easy": 0.30134680134680136,
      "hellaswag": 0.334,
      "error": null
    },
    {
      "model": "llama_fp16",
      "quant": "FP16",
      "condition": "weights",
      "flip_count": 10,
      "seed": 0,
      "ppl": 403607.932,
      "arc_easy": 0.24873737373737373,
      "hellaswag": 0.274,
      "error": null
    },
    {
      "model": "llama_fp16",
      "quant": "FP16",
      "condition": "weights",
      "flip_count": 1,
      "seed": 1,
      "ppl": 13.8845,
      "arc_easy": 0.6885521885521886,
      "hellaswag": 0.4128,
      "error": null
    },
    {
      "model": "llama_fp16",
      "quant": "FP16",
      "condition": "weights",
      "flip_count": 5,
      "seed": 1,
      "ppl": 13.8845,
      "arc_easy": 0.6893939393939394,
      "hellaswag": 0.4112,
      "error": null
    },
    {
      "model": "llama_fp16",
      "quant": "FP16",
      "condition": "weights",
      "flip_count": 10,
      "seed": 1,
      "ppl": 297273.1408,
      "arc_easy": 0.2609427609427609,
      "hellaswag": 0.276,
      "error": null
    },
    {
      "model": "llama_fp16",
      "quant": "FP16",
      "condition": "weights",
      "flip_count": 1,
      "seed": 2,
      "ppl": 13.8849,
      "arc_easy": 0.6902356902356902,
      "hellaswag": 0.4128,
      "error": null
    },
    {
      "model": "llama_fp16",
      "quant": "FP16",
      "condition": "weights",
      "flip_count": 5,
      "seed": 2,
      "ppl": 13.8854,
      "arc_easy": 0.6919191919191919,
      "hellaswag": 0.4124,
      "error": null
    },
    {
      "model": "llama_fp16",
      "quant": "FP16",
      "condition": "weights",
      "flip_count": 10,
      "seed": 2,
      "ppl": 13.8847,
      "arc_easy": 0.6910774410774411,
      "hellaswag": 0.4116,
      "error": null
    },
    {
      "model": "llama_q8",
      "quant": "Q8_0",
      "condition": "weights",
      "flip_count": 1,
      "seed": 0,
      "ppl": 13.891,
      "arc_easy": 0.6906565656565656,
      "hellaswag": 0.412,
      "error": null
    },
    {
      "model": "llama_q8",
      "quant": "Q8_0",
      "condition": "weights",
      "flip_count": 5,
      "seed": 0,
      "ppl": 13.8914,
      "arc_easy": 0.6927609427609428,
      "hellaswag": 0.4124,
      "error": null
    },
    {
      "model": "llama_q8",
      "quant": "Q8_0",
      "condition": "weights",
      "flip_count": 10,
      "seed": 0,
      "ppl": 13.8905,
      "arc_easy": 0.6914983164983165,
      "hellaswag": 0.4124,
      "error": null
    },
    {
      "model": "llama_q8",
      "quant": "Q8_0",
      "condition": "weights",
      "flip_count": 1,
      "seed": 1,
      "ppl": 13.8899,
      "arc_easy": 0.6902356902356902,
      "hellaswag": 0.4116,
      "error": null
    },
    {
      "model": "llama_q8",
      "quant": "Q8_0",
      "condition": "weights",
      "flip_count": 5,
      "seed": 1,
      "ppl": 13.8904,
      "arc_easy": 0.688973063973064,
      "hellaswag": 0.4112,
      "error": null
    },
    {
      "model": "llama_q8",
      "quant": "Q8_0",
      "condition": "weights",
      "flip_count": 10,
      "seed": 1,
      "ppl": 13.8917,
      "arc_easy": 0.6898148148148148,
      "hellaswag": 0.4124,
      "error": null
    },
    {
      "model": "llama_q8",
      "quant": "Q8_0",
      "condition": "weights",
      "flip_count": 1,
      "seed": 2,
      "ppl": 13.8908,
      "arc_easy": 0.6927609427609428,
      "hellaswag": 0.4112,
      "error": null
    },
    {
      "model": "llama_q8",
      "quant": "Q8_0",
      "condition": "weights",
      "flip_count": 5,
      "seed": 2,
      "ppl": 13.8909,
      "arc_easy": 0.6893939393939394,
      "hellaswag": 0.4108,
      "error": null
    },
    {
      "model": "llama_q8",
      "quant": "Q8_0",
      "condition": "weights",
      "flip_count": 10,
      "seed": 2,
      "ppl": 13.8915,
      "arc_easy": 0.6927609427609428,
      "hellaswag": 0.4128,
      "error": null
    },
    {
      "model": "llama_q8",
      "quant": "Q8_0",
      "condition": "block_scale",
      "flip_count": 1,
      "seed": 0,
      "ppl": 13.8891,
      "arc_easy": 0.6902356902356902,
      "hellaswag": 0.4108,
      "error": null
    },
    {
      "model": "llama_q8",
      "quant": "Q8_0",
      "condition": "block_scale",
      "flip_count": 5,
      "seed": 0,
      "ppl": 1327794.4171,
      "arc_easy": 0.2415824915824916,
      "hellaswag": 0.2844,
      "error": null
    },
    {
      "model": "llama_q8",
      "quant": "Q8_0",
      "condition": "block_scale",
      "flip_count": 10,
      "seed": 0,
      "ppl": 502670.3562,
      "arc_easy": 0.25252525252525254,
      "hellaswag": 0.2876,
      "error": null
    },
    {
      "model": "llama_q8",
      "quant": "Q8_0",
      "condition": "block_scale",
      "flip_count": 1,
      "seed": 1,
      "ppl": 13.8906,
      "arc_easy": 0.6898148148148148,
      "hellaswag": 0.4112,
      "error": null
    },
    {
      "model": "llama_q8",
      "quant": "Q8_0",
      "condition": "block_scale",
      "flip_count": 5,
      "seed": 1,
      "ppl": 13.8902,
      "arc_easy": 0.6919191919191919,
      "hellaswag": 0.412,
      "error": null
    },
    {
      "model": "llama_q8",
      "quant": "Q8_0",
      "condition": "block_scale",
      "flip_count": 10,
      "seed": 1,
      "ppl": 1373555.3815,
      "arc_easy": 0.25126262626262624,
      "hellaswag": 0.282,
      "error": null
    },
    {
      "model": "llama_q8",
      "quant": "Q8_0",
      "condition": "block_scale",
      "flip_count": 1,
      "seed": 2,
      "ppl": 13.8903,
      "arc_easy": 0.6906565656565656,
      "hellaswag": 0.412,
      "error": null
    },
    {
      "model": "llama_q8",
      "quant": "Q8_0",
      "condition": "block_scale",
      "flip_count": 5,
      "seed": 2,
      "ppl": 13.9645,
      "arc_easy": 0.6906565656565656,
      "hellaswag": 0.4116,
      "error": null
    },
    {
      "model": "llama_q8",
      "quant": "Q8_0",
      "condition": "block_scale",
      "flip_count": 10,
      "seed": 2,
      "ppl": 14.2462,
      "arc_easy": 0.6864478114478114,
      "hellaswag": 0.4088,
      "error": null
    },
    {
      "model": "llama_q4km",
      "quant": "Q4_K_M",
      "condition": "weights",
      "flip_count": 1,
      "seed": 0,
      "ppl": 14.3551,
      "arc_easy": 0.6923400673400674,
      "hellaswag": 0.4016,
      "error": null
    },
    {
      "model": "llama_q4km",
      "quant": "Q4_K_M",
      "condition": "weights",
      "flip_count": 5,
      "seed": 0,
      "ppl": 14.3544,
      "arc_easy": 0.6910774410774411,
      "hellaswag": 0.4056,
      "error": null
    },
    {
      "model": "llama_q4km",
      "quant": "Q4_K_M",
      "condition": "weights",
      "flip_count": 10,
      "seed": 0,
      "ppl": 14.3574,
      "arc_easy": 0.6847643097643098,
      "hellaswag": 0.4064,
      "error": null
    },
    {
      "model": "llama_q4km",
      "quant": "Q4_K_M",
      "condition": "weights",
      "flip_count": 1,
      "seed": 1,
      "ppl": 14.3569,
      "arc_easy": 0.6893939393939394,
      "hellaswag": 0.4044,
      "error": null
    },
    {
      "model": "llama_q4km",
      "quant": "Q4_K_M",
      "condition": "weights",
      "flip_count": 5,
      "seed": 1,
      "ppl": 14.3588,
      "arc_easy": 0.6885521885521886,
      "hellaswag": 0.4052,
      "error": null
    },
    {
      "model": "llama_q4km",
      "quant": "Q4_K_M",
      "condition": "weights",
      "flip_count": 10,
      "seed": 1,
      "ppl": 14.3546,
      "arc_easy": 0.6868686868686869,
      "hellaswag": 0.4048,
      "error": null
    },
    {
      "model": "llama_q4km",
      "quant": "Q4_K_M",
      "condition": "weights",
      "flip_count": 1,
      "seed": 2,
      "ppl": 14.3574,
      "arc_easy": 0.6881313131313131,
      "hellaswag": 0.4024,
      "error": null
    },
    {
      "model": "llama_q4km",
      "quant": "Q4_K_M",
      "condition": "weights",
      "flip_count": 5,
      "seed": 2,
      "ppl": 14.3554,
      "arc_easy": 0.6872895622895623,
      "hellaswag": 0.4056,
      "error": null
    },
    {
      "model": "llama_q4km",
      "quant": "Q4_K_M",
      "condition": "weights",
      "flip_count": 10,
      "seed": 2,
      "ppl": 14.3575,
      "arc_easy": 0.6856060606060606,
      "hellaswag": 0.4056,
      "error": null
    },
    {
      "model": "llama_q4km",
      "quant": "Q4_K_M",
      "condition": "block_scale",
      "flip_count": 1,
      "seed": 0,
      "ppl": 14.3561,
      "arc_easy": 0.6885521885521886,
      "hellaswag": 0.4024,
      "error": null
    },
    {
      "model": "llama_q4km",
      "quant": "Q4_K_M",
      "condition": "block_scale",
      "flip_count": 5,
      "seed": 0,
      "ppl": 14.3548,
      "arc_easy": 0.6877104377104377,
      "hellaswag": 0.4036,
      "error": null
    },
    {
      "model": "llama_q4km",
      "quant": "Q4_K_M",
      "condition": "block_scale",
      "flip_count": 10,
      "seed": 0,
      "ppl": 14.355,
      "arc_easy": 0.6877104377104377,
      "hellaswag": 0.4052,
      "error": null
    },
    {
      "model": "llama_q4km",
      "quant": "Q4_K_M",
      "condition": "block_scale",
      "flip_count": 1,
      "seed": 1,
      "ppl": 14.3578,
      "arc_easy": 0.6885521885521886,
      "hellaswag": 0.404,
      "error": null
    },
    {
      "model": "llama_q4km",
      "quant": "Q4_K_M",
      "condition": "block_scale",
      "flip_count": 5,
      "seed": 1,
      "ppl": 14.3568,
      "arc_easy": 0.6906565656565656,
      "hellaswag": 0.4052,
      "error": null
    },
    {
      "model": "llama_q4km",
      "quant": "Q4_K_M",
      "condition": "block_scale",
      "flip_count": 10,
      "seed": 1,
      "ppl": 14.3652,
      "arc_easy": 0.6910774410774411,
      "hellaswag": 0.404,
      "error": null
    },
    {
      "model": "llama_q4km",
      "quant": "Q4_K_M",
      "condition": "block_scale",
      "flip_count": 1,
      "seed": 2,
      "ppl": 14.3599,
      "arc_easy": 0.686026936026936,
      "hellaswag": 0.4044,
      "error": null
    },
    {
      "model": "llama_q4km",
      "quant": "Q4_K_M",
      "condition": "block_scale",
      "flip_count": 5,
      "seed": 2,
      "ppl": 14.3545,
      "arc_easy": 0.6872895622895623,
      "hellaswag": 0.4044,
      "error": null
    },
    {
      "model": "llama_q4km",
      "quant": "Q4_K_M",
      "condition": "block_scale",
      "flip_count": 10,
      "seed": 2,
      "ppl": 14.3582,
      "arc_easy": 0.6877104377104377,
      "hellaswag": 0.404,
      "error": null
    },
    {
      "model": "llama_q4km",
      "quant": "Q4_K_M",
      "condition": "super_scale",
      "flip_count": 1,
      "seed": 0,
      "ppl": 14.361,
      "arc_easy": 0.6902356902356902,
      "hellaswag": 0.4044,
      "error": null
    },
    {
      "model": "llama_q4km",
      "quant": "Q4_K_M",
      "condition": "super_scale",
      "flip_count": 5,
      "seed": 0,
      "ppl": 1411950.6712,
      "arc_easy": 0.2361111111111111,
      "hellaswag": 0.274,
      "error": null
    },
    {
      "model": "llama_q4km",
      "quant": "Q4_K_M",
      "condition": "super_scale",
      "flip_count": 10,
      "seed": 0,
      "ppl": 750866.4883,
      "arc_easy": 0.22516835016835016,
      "hellaswag": 0.2764,
      "error": null
    },
    {
      "model": "llama_q4km",
      "quant": "Q4_K_M",
      "condition": "super_scale",
      "flip_count": 1,
      "seed": 1,
      "ppl": 14.3591,
      "arc_easy": 0.6906565656565656,
      "hellaswag": 0.4032,
      "error": null
    },
    {
      "model": "llama_q4km",
      "quant": "Q4_K_M",
      "condition": "super_scale",
      "flip_count": 5,
      "seed": 1,
      "ppl": 14.3572,
      "arc_easy": 0.6877104377104377,
      "hellaswag": 0.4024,
      "error": null
    },
    {
      "model": "llama_q4km",
      "quant": "Q4_K_M",
      "condition": "super_scale",
      "flip_count": 10,
      "seed": 1,
      "ppl": null,
      "arc_easy": 0.24579124579124578,
      "hellaswag": 0.2756,
      "error": null
    },
    {
      "model": "llama_q4km",
      "quant": "Q4_K_M",
      "condition": "super_scale",
      "flip_count": 1,
      "seed": 2,
      "ppl": 14.3574,
      "arc_easy": 0.6877104377104377,
      "hellaswag": 0.4032,
      "error": null
    },
    {
      "model": "llama_q4km",
      "quant": "Q4_K_M",
      "condition": "super_scale",
      "flip_count": 5,
      "seed": 2,
      "ppl": 20.8168,
      "arc_easy": 0.5879629629629629,
      "hellaswag": 0.3832,
      "error": null
    },
    {
      "model": "llama_q4km",
      "quant": "Q4_K_M",
      "condition": "super_scale",
      "flip_count": 10,
      "seed": 2,
      "ppl": 543.9442,
      "arc_easy": 0.39015151515151514,
      "hellaswag": 0.322,
      "error": null
    },
    {
      "model": "llama_q4",
      "quant": "Q4_0",
      "condition": "weights",
      "flip_count": 1,
      "seed": 0,
      "ppl": 15.0621,
      "arc_easy": 0.6759259259259259,
      "hellaswag": 0.4104,
      "error": null
    },
    {
      "model": "llama_q4",
      "quant": "Q4_0",
      "condition": "weights",
      "flip_count": 5,
      "seed": 0,
      "ppl": 15.0602,
      "arc_easy": 0.672979797979798,
      "hellaswag": 0.4112,
      "error": null
    },
    {
      "model": "llama_q4",
      "quant": "Q4_0",
      "condition": "weights",
      "flip_count": 10,
      "seed": 0,
      "ppl": 15.0605,
      "arc_easy": 0.6759259259259259,
      "hellaswag": 0.4104,
      "error": null
    },
    {
      "model": "llama_q4",
      "quant": "Q4_0",
      "condition": "weights",
      "flip_count": 1,
      "seed": 1,
      "ppl": 15.0603,
      "arc_easy": 0.6755050505050505,
      "hellaswag": 0.4112,
      "error": null
    },
    {
      "model": "llama_q4",
      "quant": "Q4_0",
      "condition": "weights",
      "flip_count": 5,
      "seed": 1,
      "ppl": 15.0578,
      "arc_easy": 0.6742424242424242,
      "hellaswag": 0.4088,
      "error": null
    },
    {
      "model": "llama_q4",
      "quant": "Q4_0",
      "condition": "weights",
      "flip_count": 10,
      "seed": 1,
      "ppl": 15.0592,
      "arc_easy": 0.672979797979798,
      "hellaswag": 0.4088,
      "error": null
    },
    {
      "model": "llama_q4",
      "quant": "Q4_0",
      "condition": "weights",
      "flip_count": 1,
      "seed": 2,
      "ppl": 15.0604,
      "arc_easy": 0.6721380471380471,
      "hellaswag": 0.4108,
      "error": null
    },
    {
      "model": "llama_q4",
      "quant": "Q4_0",
      "condition": "weights",
      "flip_count": 5,
      "seed": 2,
      "ppl": 15.0605,
      "arc_easy": 0.6742424242424242,
      "hellaswag": 0.4104,
      "error": null
    },
    {
      "model": "llama_q4",
      "quant": "Q4_0",
      "condition": "weights",
      "flip_count": 10,
      "seed": 2,
      "ppl": 15.0598,
      "arc_easy": 0.6797138047138047,
      "hellaswag": 0.4096,
      "error": null
    },
    {
      "model": "llama_q4",
      "quant": "Q4_0",
      "condition": "block_scale",
      "flip_count": 1,
      "seed": 0,
      "ppl": 15.0597,
      "arc_easy": 0.6738215488215489,
      "hellaswag": 0.4116,
      "error": null
    },
    {
      "model": "llama_q4",
      "quant": "Q4_0",
      "condition": "block_scale",
      "flip_count": 5,
      "seed": 0,
      "ppl": 1004948.1253,
      "arc_easy": 0.24452861952861954,
      "hellaswag": 0.2816,
      "error": null
    },
    {
      "model": "llama_q4",
      "quant": "Q4_0",
      "condition": "block_scale",
      "flip_count": 10,
      "seed": 0,
      "ppl": 1016870.6413,
      "arc_easy": 0.242003367003367,
      "hellaswag": 0.2764,
      "error": null
    },
    {
      "model": "llama_q4",
      "quant": "Q4_0",
      "condition": "block_scale",
      "flip_count": 1,
      "seed": 1,
      "ppl": 15.0593,
      "arc_easy": 0.6738215488215489,
      "hellaswag": 0.41,
      "error": null
    },
    {
      "model": "llama_q4",
      "quant": "Q4_0",
      "condition": "block_scale",
      "flip_count": 5,
      "seed": 1,
      "ppl": 15.0597,
      "arc_easy": 0.6742424242424242,
      "hellaswag": 0.4108,
      "error": null
    },
    {
      "model": "llama_q4",
      "quant": "Q4_0",
      "condition": "block_scale",
      "flip_count": 10,
      "seed": 1,
      "ppl": 636504.0043,
      "arc_easy": 0.24621212121212122,
      "hellaswag": 0.2652,
      "error": null
    },
    {
      "model": "llama_q4",
      "quant": "Q4_0",
      "condition": "block_scale",
      "flip_count": 1,
      "seed": 2,
      "ppl": 15.0598,
      "arc_easy": 0.6750841750841751,
      "hellaswag": 0.412,
      "error": null
    },
    {
      "model": "llama_q4",
      "quant": "Q4_0",
      "condition": "block_scale",
      "flip_count": 5,
      "seed": 2,
      "ppl": 15.0609,
      "arc_easy": 0.6746632996632996,
      "hellaswag": 0.4104,
      "error": null
    },
    {
      "model": "llama_q4",
      "quant": "Q4_0",
      "condition": "block_scale",
      "flip_count": 10,
      "seed": 2,
      "ppl": 15.1613,
      "arc_easy": 0.6708754208754208,
      "hellaswag": 0.4088,
      "error": null
    }
  ]
}"""

# Replaced the unquoted Infinity string with a large float limit to safely parse it into standard JSON
json_str = raw_json.replace("Infinity", "1e308")
data = json.loads(json_str)

# Convert into a Pandas DataFrame
df = pd.DataFrame(data['runs'])

# Convert ppl to numeric values, coercing any errors
df['ppl'] = pd.to_numeric(df['ppl'], errors='coerce')

# Plot the graph with Seaborn
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='flip_count', y='ppl', hue='model', style='condition', markers=True, dashes=False)

plt.yscale('log')
plt.title('Perplexity vs Number of Bit Flips')
plt.xlabel('Number of Bit Flips')
plt.ylabel('Perplexity (Log Scale)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()

# Save the figure
plt.savefig('ppl_vs_flips.png', dpi=300)