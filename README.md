# Computational Neuroscience and Neuroimaging: Foundational Projects

Covers exploratory implementations in EEG sleep-stage classification, fMRI decoding with MVPA, FastSurfer MRI segmentation, learning Hodgkin-Huxley neuron dynamics with MLP/TCN/GRU networks, and a comparison of spiking (STDP) and standard neural networks on MNIST.

## Contents

| # | Project | Data | Main methods |
|---|---------|------|--------------|
| 1 | [Replicating Hodgkin-Huxley Spiking Dynamics with Rudimentary Universal Function Approximators: A Comparative Study](Replicating_Hodgkin_Huxley_Spiking_Dynamics_with_Rudimentary_Universal_Function_Approximators_A_Comparative_Study.ipynb) | Simulated HH neuron | MLP, TCN, GRU |
| 2 | [Spiking vs Standard Neural Networks on Handwritten Digits: Unsupervised STDP Learning, Sparsity and Biological Constraints](Spiking_vs_Standard_Neural_Networks_on_Handwritten_Digits_Unsupervised_STDP_Learning,_Sparsity_and_Biological_Constraints.ipynb) | MNIST | STDP-based SNN (BindsNET), MLP |
| 3 | [Sleep Stage Classification from Single-Channel EEG Band Power Spectral Analysis](Sleep_Stage_Classification_from_Single_Channel_EEG_Band_Power_Spectral_Analysis.ipynb) | Sleep-EDF (PhysioNet) | Spectral analysis, Random Forest |
| 4 | [Decoding Visual Category Representations from fMRI Activity Using Multi-Voxel Pattern Analysis](Decoding_Visual_Category_Representations_from_fMRI_Activity_Using_Multi_Voxel_Pattern_Analysis.ipynb) | OpenNeuro ds001497 | MVPA, linear SVM |
| 5 | [Brain Segmentation and Volumetry of T1-Weighted MRI Using FastSurfer](Brain_Segmentation_and_Volumetry_of_T1_Weighted_MRI_Using_FastSurfer.ipynb) | OpenNeuro ds000001 (configurable) | FastSurfer, FreeSurfer |

---

## 1. Replicating Hodgkin-Huxley Spiking Dynamics with Rudimentary Universal Function Approximators: A Comparative Study

**Notebook:** `Replicating_Hodgkin_Huxley_Spiking_Dynamics_with_Rudimentary_Universal_Function_Approximators_A_Comparative_Study.ipynb`

**Background.** The Hodgkin-Huxley (HH) model is the classic description of how a neuron's membrane voltage responds to injected current through ion channels. Its dynamics depend on three hidden gating variables (m, h, n) that cannot be observed from current and voltage alone. This raises a system-identification question: can a model learn the neuron's behaviour from input-output data without being given those variables? The notebook builds on classical methods (Volterra-Wiener kernels, ARX models) and on recent neural-network and neural-ODE work.

**Objectives.** Using simulated HH data with known ground truth, the study asks (a) whether a model can predict dV/dt from current and voltage without the gating variables, and (b) whether it can then run on its own to simulate a realistic voltage trajectory. It compares three architectures that differ only in access to history:

| Model | Access to history |
|-------|-------------------|
| MLP | None (negative control) |
| TCN | Fixed window, dilated causal convolutions |
| GRU | Learned recurrent summary |

Each model is trained three times and evaluated on one-step prediction (R²), free-running trajectory error (RMSE) on stimulus types never seen in training (a sine wave and a stronger step), and spike counts. The notebook also confirms the hidden-state ambiguity directly in the simulated data.

**Results.** On one-step prediction, the memoryless MLP failed (R² ≈ 0.002) while the TCN and GRU both reached R² ≈ 0.998 and 0.997. In free-running simulation, the GRU was the most stable (RMSE ≈ 26.7 mV), whereas the TCN's error was larger and more variable. This is a simulation study, so it does not make claims about real neural data.

**Note.** A `QUICK_TEST` flag in the configuration cell runs a much faster reduced version. A GPU is recommended for the full run.

---

## 2. Spiking vs Standard Neural Networks on Handwritten Digits: Unsupervised STDP Learning, Sparsity and Biological Constraints

**Notebook:** `Spiking_vs_Standard_Neural_Networks_on_Handwritten_Digits_Unsupervised_STDP_Learning,_Sparsity_and_Biological_Constraints.ipynb`

**Background.** Neurons in the brain communicate through brief electrical pulses called spikes, and at any moment most neurons are silent. This sparse, event-driven activity is thought to be one reason the brain runs on only about 20 watts. Standard artificial neural networks (ANNs) compute continuous values in every unit and learn through backpropagation, which is biologically questionable. Spiking neural networks (SNNs) use leaky integrate-and-fire neurons and can learn with local, timing-based rules that need no answers:
- **STDP:** synapses strengthen when input spikes just before the neuron fires and weaken when it spikes just after.
- **Lateral inhibition:** firing neurons suppress the others, so each learns a different pattern.
- **Adaptive thresholds:** frequently firing neurons become harder to fire, a simple form of homeostasis.

**Objectives.** The notebook builds a small SNN (100 neurons, BindsNET) and compares it with a 784-100-10 ANN on the same 1,500 training and 500 test MNIST images. It asks:
1. Can the SNN learn to recognise digits without being told the answers?
2. How do the two compare on accuracy, activity and a rough energy estimate?
3. How do both cope with brain-like constraints: coarsely rounded synaptic weights and short decision times (10-100 ms)?

**Results.** The ANN reached 85% and the SNN 45%. The SNN was very sparse (about 0.2% of time slots active, about 2.2 spikes/s per neuron) with a lower rough energy estimate (116 nJ versus 365 nJ per image). Its accuracy was unaffected by rounding weights to as few as 3 levels, and it improved as more decision time was allowed. Energy values are textbook estimates, not hardware measurements.

---

## 3. Sleep Stage Classification from Single-Channel EEG Band Power Spectral Analysis

**Notebook:** `Sleep_Stage_Classification_from_Single_Channel_EEG_Band_Power_Spectral_Analysis.ipynb`

**Background.** Electroencephalography (EEG) records the brain's electrical activity through scalp electrodes. Over a night, the brain cycles through Wake, light sleep (N1, N2), deep sleep (N3) and REM sleep, and each stage has a characteristic rhythm: deep sleep is dominated by slow delta waves, and wakefulness by faster beta activity. Sleep specialists traditionally label these stages by hand in 30-second windows (epochs).

**Objectives.** The notebook automates sleep staging on a real overnight recording from the Sleep-EDF dataset. It:
- filters a single EEG channel (Fpz-Cz) to 0.5-30 Hz and segments the night into 30-second epochs;
- extracts delta, theta, alpha and beta band power plus a theta/alpha ratio for each epoch;
- trains a Random Forest to predict the sleep stage, using a block-wise train/test split to avoid leakage between neighbouring epochs;
- evaluates with a confusion matrix and per-stage metrics, and plots true versus predicted hypnograms.

**Results.** Test accuracy was 90.5% (macro F1 0.70) on one subject and one night. Accuracy is inflated by class imbalance, since Wake makes up about 75% of epochs. Wake (F1 0.96) and N3 (F1 0.82) were classified most reliably, while N1 was weakest (F1 0.33).

---

## 4. Decoding Visual Category Representations from fMRI Activity Using Multi-Voxel Pattern Analysis

**Notebook:** `Decoding_Visual_Category_Representations_from_fMRI_Activity_Using_Multi_Voxel_Pattern_Analysis.ipynb`

**Background.** Functional MRI measures brain activity indirectly through the BOLD signal, a blood-oxygenation change that peaks a few seconds after neural activity. Multi-voxel pattern analysis (MVPA) treats the pattern across many voxels as a signature that a classifier can learn to decode. This approach builds on work showing that ventral temporal cortex represents categories such as faces and places in distributed patterns.

**Objectives.** The notebook tests whether the category a person viewed (face, place or object) can be predicted from fMRI data alone. Using one subject (sub-01) from OpenNeuro ds001497, it:
- averages the BOLD signal 4-10 seconds after each trial onset;
- applies an EPI brain mask to produce a trials-by-voxels matrix;
- trains a linear SVM with leave-one-run-out cross-validation across six runs;
- compares accuracy against the 33.3% chance level.

**Results.** Mean accuracy was 67.78% ± 11.81% (90 trials, 21,600 voxels), well above chance. Faces were decoded best (F1 0.83), with objects (0.57) and places (0.61) less well separated. This is a single-subject analysis with no permutation test, so results are descriptive.

---

## 5. Brain Segmentation and Volumetry of T1-Weighted MRI Using FastSurfer

**Notebook:** `Brain_Segmentation_and_Volumetry_of_T1_Weighted_MRI_Using_FastSurfer.ipynb`

**Background.** Structural T1-weighted MRI shows brain anatomy in fine detail, and measures such as regional volumes and cortical thickness are widely used in research and clinical neuroscience. FreeSurfer is the standard tool for extracting them but takes hours per scan. FastSurfer is a deep-learning alternative that segments the brain into anatomical regions in minutes on a GPU.

**Objectives.** The notebook is a reproducible Colab pipeline from raw data to measurements. It:
- downloads a T1 scan directly from OpenNeuro over unsigned S3, with no manual upload (dataset, subject and session are configurable);
- runs FastSurfer in segmentation-only mode (about 1-2 minutes on a GPU, gives ROI volumes) or full mode with surface reconstruction (about 45-90 minutes, adds cortical thickness, surface area and gray-matter volume per region);
- parses the resulting stats files into pandas DataFrames and saves them as CSVs.

**Outputs.** `<subject>_roi_volumes.csv`, plus `<subject>_lh_aparc.csv` and `<subject>_rh_aparc.csv` in full mode.

---

## Getting Started

### Running the notebooks

The fMRI and MRI notebooks use `/content/...` paths, so they are written for [Google Colab](https://colab.research.google.com/). The others also run in Colab or locally. To open a notebook in Colab, use:

```
https://colab.research.google.com/github/<your-username>/<your-repo>/blob/main/<notebook>.ipynb
```

### Requirements

Each notebook installs its own dependencies in its first cells. The main packages are:

| Notebook | Key packages |
|----------|--------------|
| `Replicating_Hodgkin_Huxley_...ipynb` | `torch`, `numpy`, `pandas`, `matplotlib` |
| `Spiking_vs_Standard_...ipynb` | `bindsnet`, `torch`, `torchvision`, `pandas==2.2.3`, `matplotlib` |
| `Sleep_Stage_Classification_...ipynb` | `mne`, `numpy`, `scipy`, `scikit-learn`, `matplotlib` |
| `Decoding_Visual_Category_...ipynb` | `nilearn`, `pandas`, `scikit-learn`, `awscli` |
| `Brain_Segmentation_and_Volumetry_...ipynb` | `awscli`, FastSurfer (cloned from GitHub), `torch`, `pandas`; FreeSurfer 7.4.1 for full mode |

### Hardware and data notes

- **GPU:** recommended for the Hodgkin-Huxley and FastSurfer (MRI) notebooks (Colab: Runtime > Change runtime type > GPU).
- **FreeSurfer license:** only needed for full surface reconstruction in the FastSurfer (MRI) notebook. A free license can be obtained at <https://surfer.nmr.mgh.harvard.edu/registration.html>.
- **Data:** all datasets are downloaded automatically by the notebooks (PhysioNet via MNE, OpenNeuro via public S3, MNIST via torchvision). No data files are stored in this repository.

## Repository Structure

```
.
├── Replicating_Hodgkin_Huxley_Spiking_Dynamics_with_Rudimentary_Universal_Function_Approximators_A_Comparative_Study.ipynb
├── Spiking_vs_Standard_Neural_Networks_on_Handwritten_Digits_Unsupervised_STDP_Learning,_Sparsity_and_Biological_Constraints.ipynb
├── Sleep_Stage_Classification_from_Single_Channel_EEG_Band_Power_Spectral_Analysis.ipynb
├── Decoding_Visual_Category_Representations_from_fMRI_Activity_Using_Multi_Voxel_Pattern_Analysis.ipynb
├── Brain_Segmentation_and_Volumetry_of_T1_Weighted_MRI_Using_FastSurfer.ipynb
└── README.md
```

## Data Sources and Key References

- **Sleep-EDF** database, PhysioNet (accessed through `mne.datasets.sleep_physionet`).
- **OpenNeuro** datasets ds001497 and ds000001. Markiewicz, C. J., et al. (2021). The OpenNeuro resource for sharing of neuroscience data. *eLife*, 10, e71774.
- **MNIST** handwritten digit database.
- **FastSurfer:** <https://github.com/Deep-MI/FastSurfer>
- **Hodgkin, A. L., & Huxley, A. F. (1952).** A quantitative description of membrane current and its application to conduction and excitation in nerve. *The Journal of Physiology*, 117(4), 500-544.
- **Haxby, J. V., et al. (2001).** Distributed and overlapping representations of faces and objects in ventral temporal cortex. *Science*, 293(5539), 2425-2430.
- **Bai, S., Kolter, J. Z., & Koltun, V. (2018).** An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv:1803.01271.
- **Cho, K., et al. (2014).** Learning phrase representations using RNN encoder-decoder for statistical machine translation. *EMNLP 2014*.


## License

Datasets used by the notebooks are subject to their own licenses and terms of use.
