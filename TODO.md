### Temporal proposal
- [BMN](http://openaccess.thecvf.com/content_ICCV_2019/papers/Lin_BMN_Boundary-Matching_Network_for_Temporal_Action_Proposal_Generation_ICCV_2019_paper.pdf)
- [Gaussian Network](http://openaccess.thecvf.com/content_CVPR_2019/papers/Long_Gaussian_Temporal_Awareness_Networks_for_Action_Localization_CVPR_2019_paper.pdf)
- [CleanNet](http://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Weakly_Supervised_Temporal_Action_Localization_Through_Contrast_Based_Evaluation_Networks_ICCV_2019_paper.pdf)

## TODO:
1. BMN
   1. ~~Implement Soft-NMS~~
   2. BMN loss function: 
      - OIC loss (*low-high-low*)
      - Think about what to do with overlap? 
   3. BMN length in test (*can it be kept into length 100 without `CUDA_OUT_OF_MEMORY`?*)   
