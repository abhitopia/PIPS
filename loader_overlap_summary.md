# ARC/BARC Loader Overlap Analysis

## Overview

This document summarizes the program ID overlap between different ARC and BARC dataset loaders. The analysis identifies which datasets share the same program IDs, indicating potential duplication or intentional sharing between datasets.

## Key Findings

1. **BARC Datasets**: The BARC datasets (BARC_GP4OM_OM, BARC_GP4O_OM, BARC_GP4O_OM_SUG, BARC_GP4_OM) have significant overlap with each other:
   - BARC_GP4OM_OM and BARC_GP4O_OM_SUG: 97.16% overlap (BARC_GP4OM_OM contains 97.16% of BARC_GP4O_OM_SUG's program IDs)
   - BARC_GP4O_OM and BARC_GP4_OM: 99.34% overlap (BARC_GP4O_OM contains 99.34% of BARC_GP4_OM's program IDs)
   - All BARC datasets share at least 80% of their program IDs with each other

2. **ARC AGI Datasets**: The ARC AGI datasets have specific overlap patterns:
   - ARCAGI1_TRAIN and ARCAGI2_TRAIN: 100% overlap (ARCAGI2_TRAIN contains all ARCAGI1_TRAIN program IDs)
   - ARCAGI1_EVAL and ARCAGI2_TRAIN: 98.43% overlap (ARCAGI2_TRAIN contains 98.43% of ARCAGI1_EVAL program IDs)
   - ARCAGI2_EVAL and ARCAGI1_EVAL: Small but notable 1.57% overlap (6 program IDs)

3. **No Overlap Between ARC and BARC**: There is zero overlap between ARC and BARC datasets, confirming they are completely separate datasets.

## Detailed Overlap Matrix

### Absolute Counts

| Loader | ARCAGI1_EVAL | ARCAGI1_TRAIN | ARCAGI2_EVAL | ARCAGI2_TRAIN | BARC_GP4OM_OM | BARC_GP4O_OM | BARC_GP4O_OM_SUG | BARC_GP4_OM |
|--------|--------------|---------------|--------------|---------------|---------------|--------------|------------------|-------------|
| ARCAGI1_EVAL | 382 | 0 | 6 | 376 | 0 | 0 | 0 | 0 |
| ARCAGI1_TRAIN | 0 | 391 | 0 | 391 | 0 | 0 | 0 | 0 |
| ARCAGI2_EVAL | 6 | 0 | 6 | 0 | 0 | 0 | 0 | 0 |
| ARCAGI2_TRAIN | 376 | 391 | 0 | 767 | 0 | 0 | 0 | 0 |
| BARC_GP4OM_OM | 0 | 0 | 0 | 0 | 101317 | 89278 | 98438 | 81613 |
| BARC_GP4O_OM | 0 | 0 | 0 | 0 | 89278 | 94547 | 87216 | 86342 |
| BARC_GP4O_OM_SUG | 0 | 0 | 0 | 0 | 98438 | 87216 | 99310 | 79678 |
| BARC_GP4_OM | 0 | 0 | 0 | 0 | 81613 | 86342 | 79678 | 86920 |

### Percentage Overlap (% of row loader)

| Loader | ARCAGI1_EVAL | ARCAGI1_TRAIN | ARCAGI2_EVAL | ARCAGI2_TRAIN | BARC_GP4OM_OM | BARC_GP4O_OM | BARC_GP4O_OM_SUG | BARC_GP4_OM |
|--------|--------------|---------------|--------------|---------------|---------------|--------------|------------------|-------------|
| ARCAGI1_EVAL | 100.00 | 0.00 | 1.57 | 98.43 | 0.00 | 0.00 | 0.00 | 0.00 |
| ARCAGI1_TRAIN | 0.00 | 100.00 | 0.00 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| ARCAGI2_EVAL | 100.00 | 0.00 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| ARCAGI2_TRAIN | 49.02 | 50.98 | 0.00 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| BARC_GP4OM_OM | 0.00 | 0.00 | 0.00 | 0.00 | 100.00 | 88.12 | 97.16 | 80.55 |
| BARC_GP4O_OM | 0.00 | 0.00 | 0.00 | 0.00 | 94.43 | 100.00 | 92.25 | 91.32 |
| BARC_GP4O_OM_SUG | 0.00 | 0.00 | 0.00 | 0.00 | 99.12 | 87.82 | 100.00 | 80.23 |
| BARC_GP4_OM | 0.00 | 0.00 | 0.00 | 0.00 | 93.89 | 99.34 | 91.67 | 100.00 |

## Implications

1. **BARC Dataset Redundancy**: The high overlap among BARC datasets suggests significant redundancy. When using these datasets for training, it's important to consider that they contain many of the same program IDs.

2. **ARC AGI Dataset Relationships**: The ARCAGI2_TRAIN dataset appears to be a superset of both ARCAGI1_TRAIN and most of ARCAGI1_EVAL. This suggests an intentional progression from one version to the next.

3. **Dataset Isolation**: There is complete isolation between ARC and BARC datasets, which means they can be used as truly independent datasets for training and evaluation.

## Recommendations

1. When training on BARC datasets, select one primary dataset rather than using multiple BARC datasets to avoid redundancy.

2. For ARC AGI evaluation, ARCAGI2_EVAL contains unique samples not found in other datasets, making it valuable for independent testing.

3. ARCAGI2_TRAIN can serve as a comprehensive dataset that covers both ARCAGI1_TRAIN and most of ARCAGI1_EVAL. 