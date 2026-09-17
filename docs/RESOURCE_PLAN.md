# Free-resource plan

The first bottleneck is satellite transfer/decompression, not the small neural network. Anonymous AWS access, the supplied CSV and CPU execution require no paid service. Free Colab GPU availability is variable. The notebook uses bundled crops and does not download new satellite files.

The 2018-01-01/03/05 B13 inventory for 04:00–06:20 UTC contains 44 objects and one missing timestamp (January 5 05:10). Listed payload is exactly **1,074,348,509 bytes =1.000565 GiB**. This payload was transferred again after the session reset removed the first transient copy; total session recovery traffic is consequently approximately twice the single-acquisition figure. The final archive stores only small crops and provenance, not the compressed full disks.

At 2018-01-01 00:00 UTC, observed compressed sizes were B08 9,697,179 bytes, B13 24,150,174, B15 23,488,232. Sizes vary by scene. The combined 57,335,585-byte example implies:

| Scope | Approximate raw transfer | Status |
|---|---:|---|
| 44-frame B13 example | 1.000565 GiB | Exact listed payload for one acquisition |
| Whole day, B13 | 3.5 GB decimal | Estimate, 144 frames |
| Whole day, three bands | 8.3 GB decimal | Estimate from one observed timestamp |
| Full 2018, three bands | 3.0 TB decimal /2.74 TiB | Estimate, not an acquisition request |
| Retained 64×64 three-band arrays, full year | 2.6 GB float32 +0.65 GB masks | Uncompressed retained data, not bandwidth |

A compressed full-disk HSD cannot provide an arbitrary regional crop by a simple byte-range request. Cropping reduces retained storage, not initial transfer. A 2.5-hour B13 block consumes about 0.35 GiB for only ten six-frame windows before gaps.

Run the bundled smoke and full-year tabular benchmark first. For expansion, select predeclared contiguous blocks covering seasons; inspect metadata-only `satellite-plan` before fetching. Start with B13 and add B08/B15 on identical timestamps. Keep crops/metadata and discard raw files unless further preprocessing experiments need them. More GPU compute alone does not solve the transfer problem. A provider with genuine regional subsets/shared preprocessing would need separate verification and is not assumed here.

Caps are 1.5 GiB for smoke and 2 GiB for a new study. They conservatively bound listed raw objects needed for uncropped frames per invocation, not network protocol overhead/retries or cumulative monthly usage. Completed crops are checked and reused; interrupted individual transfers restart. New study windows are empty by default. Missing frames are logged, not silently filled.

Whole-disk decoding needs much more RAM than a crop. One-band decoding has been exercised; three-band peak RAM/full-study GPU duration remains unmeasured. Decoder Dask execution is single-threaded, network prefetch is bounded, and training caches at most 128 patches per dataset. The prototype uses zero DataLoader workers for Windows/Colab portability.

Use measured wall time, epochs/second and parameter counts from a sufficiently sized pilot to budget the three-seed campaign. Tiny smoke throughput is not a reliable full-study estimate. Each completed epoch saves last/best checkpoints with optimizer/RNG state. Download the run and its exact data/config before Colab disconnects; no automatic Drive mounting or paid runtime is required.
