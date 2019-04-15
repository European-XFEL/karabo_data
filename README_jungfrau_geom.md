# Adding a geometry description for the Jungfrau detector

This starts with a single module with its 2 x 4 tiles as geometry fragments, in analogy to AGIPD and LPD.

Each Jungfrau ASIC has a pixel margin of double physical size. This fact is ignored at the moment, like for the other detectors, meaning that the border pixel rows/columns are treated like normal pixels concerning size and photon counts. The only implication is that tile origins in the composed image must account for the resulting 2 px ASIC gap.

