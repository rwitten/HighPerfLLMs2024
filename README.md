# High Performance LLMs 2024
Build a full scale, high-performance LLM from scratch in Jax! We’ll cover training and inference, roofline analysis, compilation, sharding, profiling and more. You’ll leave the class comfortable in Jax and confident in your ability to design high-performance computing systems that reach near their physical limit.

Link to the Discord: [https://discord.gg/2AWcVatVAw](https://discord.gg/2AWcVatVAw)

# Syllabus. We will:
* Build a Jax LLM Implementation From Scratch
* Analyze Single Chip Rooflines And Compilation
* Analyze Distributed Computing via Sharding
* Optimize LLM Training – what happens under the hood, rooflines, sharding
* Optimize LLM Inference – what happens under the hood, rooflines, sharding
* Deep Dive into flash, vLLM, continuous batching, etc.
* Some deep dives along the way:
  - Attention, Flash Attention, vLLM, continuous batching
  - ML: Quantization, Checkpointing, Data Loading, Numerics
  - Practical Tips: Debugging, Overlapping Jax Kernels
  - Larger scale: Goodput
  - Fancy stuff: Ahead of Time Compilation
  - Going deeper: shard map, pallas.

# Approximate Timing
3:30PM Pacific on Wednesdays, starting 2/21/2024. See below for links

# Session Timing, Slides, Videos and Take-Home Exercises

| Session    |              Time                | Link to join (or recording)                                     | Slides                           | Take-Home Exercises                      |  Summary                             |
| --------   | -------                          |  ----                                                           |         -----                    |        -----                             |  -----                               |
| 1          | 3:30PM US Pacific, 2/21/2024     | [Youtube recording](https://www.youtube.com/watch?v=W0Cix2KNyXc)| [slides](s01/Session1Slides.pdf) |  [link](s01/AfterSessionExercises.txt)   |  end-to-end Jax LLM                  |
| 2          | 3:30PM US Pacific, 2/28/2024     | [Youtube recording](https://www.youtube.com/watch?v=RciT5fcuN1E)| [slides](s02/Session2Slides.pdf) |  [link](s02/AfterSessionExercises.txt)   |  single chip perf and rooflines      |
| 3          | 3:30PM US Pacific, 3/13/2024     | [Youtube recording](https://www.youtube.com/watch?v=9jC-YiZ2fkA)| [slides](s03/Session3Slides.pdf) |  [link](s03/AfterSessionExercises.txt)   |  multi chip perf and rooflines, 1    |
| 4          | 3:30PM US Pacific, 3/20/2024     | [Youtube recording](https://youtu.be/V5SPOR4Wilk)               | [slides](s04/Session4Slides.pdf) |  [link](s04/AfterSessionExercises.txt)   |  multi chip perf and rooflines, 1    |
| 5          | 3:30PM US Pacific, 3/27/2024     | [Youtube recording](https://youtu.be/h2khnnFqJMA)               | [slides](s05/Session5Slides.pdf) |  [link](s05/AfterSessionExercises.txt)   |  attention                           |
| 6          | 3:30PM US Pacific, 4/10/2024     | [Youtube recording](https://youtu.be/3dQBwysPgTk)               | [slides](s06/Session6Slides.pdf) |  [link](s06/AfterSessionExercises.txt)   |  optimized training                  |
| postponed  | 3:30PM US Pacific, 4/17/2024     | postponed                                                       |                                  |                                          |
| 7          | 3:30PM US Pacific, 4/24/2024     | [Youtube recording](https://youtu.be/enDiaGBWkV0)               | [slides](s07/Session7Slides.pdf) |  [link](s07/AfterSessionExercises.txt)   |  training e2e, inference analysis    |
| postponed  | 3:30PM US Pacific, 5/01/2024     | postponed                                                       |                                  |                                          |
| 8          | 3:30PM US Pacific, 5/08/2024     | [Youtube recording](https://youtu.be/drb7kXQ0_js)               | [slides](s08/Session8Slides.pdf) |  [link](s08/AfterSessionExercises.txt)   |  training xprof, mfu, naive inference| 
| 9          | 3:30PM US Pacific, 5/22/2024     | [Google Meet link](https://meet.google.com/tdd-brrt-gtp)        |                                  |                                          |
| 10         | 3:30PM US Pacific, 5/29/2024     | [Google Meet link](https://meet.google.com/tdd-brrt-gtp)        |                                  |                                          |


About me:
I’m Rafi Witten, a tech lead on Cloud TPU/GPU Multipod. We develop MaxText and aim to push the frontier on Perf/TCO. In 2023, we executed the ["Largest ML Job"](https://cloud.google.com/blog/products/compute/the-worlds-largest-distributed-llm-training-job-on-tpu-v5e?e=13802955#:~:text=We%20demonstrated%20the%20benefits%20of,JAX%20ML%20framework%2C%20utilizing%20both) ever demonstrated in public and pioneered [“Accurate Quantized Training”](https://cloud.google.com/blog/products/compute/accurate-quantized-training-aqt-for-tpu-v5e?e=13802955), a technique for training with 8-bit integers.

Contact me via Discord [https://discord.gg/2AWcVatVAw](https://discord.gg/2AWcVatVAw)
