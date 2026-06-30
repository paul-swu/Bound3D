# Bound3D (ECCV 2026)

**Don't Starve the Boundaries: Boundary-Constrained Label Propagation for Weakly Supervised 3D Segmentation**

This is the official repository for the paper **Bound3D** (ECCV 2026).

## 📢 News
* **[June 2026]** The paper has been accepted by ECCV 2026! 

## 📝 Abstract
Although fully supervised methods have substantially advanced segmentation in boundary areas of point clouds, effective weakly supervised approaches remain scarce. This is primarily because limited supervision rarely reaches boundary regions, leaving them lacking reliable supervision. We propose a novel 2D-assisted pseudo-label propagation paradigm that does not rely on the model’s own predictions or any external foundation models, yet is able to generate high-purity pseudo-labels. Compared with SAM-based 2D-3D projection, our pseudo-labels are purer and more uniformly distributed. Even under 1 pt/obj setting on S3DIS, our initial offline propagation achieves >94.4 accuracy (≈ 93k pts per scene). We decomposed the pseudo-labels generation process from the main network, and applied a divide-and-conquer strategy: supervision from interior pseudo-labels serves to stabilize the representation of core class regions, while boundary pseudo-labels are leveraged to enhance boundary robustness. This design reduces the confirmation bias inherent in classic online labeling and alleviates the lack of boundary supervision in existing weakly supervised models. Experiments show that our method outperforms existing state-of-the-art methods.

## 🖼️ Method / Teaser
<img width="1943" height="956" alt="visbd" src="https://github.com/user-attachments/assets/984134ed-be3d-4740-894c-cefc99ff2d6c" />
