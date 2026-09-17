# Sources and provenance

Primary documentation checked on 2026-09-17. This is an implementation bibliography, not a systematic novelty review.

1. [NSRDB Himawari API](https://developer.nlr.gov/docs/solar/nsrdb/himawari-download/): 2016–2020 point product and attributes. The user confirmed the Asia/Australia/Pacific 2 km/10-minute product. Future downloads should request clear-sky GHI and fill/quality attributes. No key is needed for the existing CSV.
2. [NOAA Himawari Open Data on AWS](https://registry.opendata.aws/noaa-himawari/): anonymous archive, buckets and cadence. Actual 2018 object listings are preserved in `outputs/engineering_smoke_inventory.json`; crop JSON files retain source hashes and scan times.
3. [JMA Advanced Himawari Imager](https://www.data.jma.go.jp/mscweb/en/himawari89/space_segment/spsg_ahi.html): spectral bands and nominal sampling. The nominal 2 km IR resolution applies at the sub-satellite point.
4. [Satpy AHI HSD reader](https://satpy.readthedocs.io/en/stable/api/satpy.readers.ahi_hsd.html) and [Scene](https://satpy.readthedocs.io/en/stable/api/satpy.scene.html): HSD calibration, time metadata and resampling.
5. [pvlib Haurwitz model](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.clearsky.haurwitz.html): deterministic solar-zenith-based reference, distinct from NSRDB REST2.
6. [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html): seed/determinism controls. Identical results are not guaranteed across software versions/hardware.

Source CSV SHA-256: `cfc2df5dce83835f3ef1ae21e302d82f3ac091351e7aa10aa74136f9bc6cab8e`.

Workflow assistance: the experiment-design skill structured the controls and progression; scientific-visualization informed the diagnostic's physical scale and provenance. For the latter, attribution is Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026), [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065). Current arXiv author/year metadata was verified during this session. This workflow attribution is not atmospheric evidence.

There is no supplied 2018 ground pyranometer reference. The 2026 VOCI upload is outside the experiment. Preserve provider terms and cite NOAA/JMA and the NSRDB provider when distributing derived data/results.
