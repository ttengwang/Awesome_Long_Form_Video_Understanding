# Awesome Long-Term Video Understanding
Real-world videos are usually long, untrimmed, and contain several actions (events). Traditionally, video understanding has focused on short-term analysis, such as action recognition, video object detection/segmentation, or scene understanding in individual video frames or short video clips. However, with the advent of more advanced technologies and the increasing availability of large-scale video datasets, researchers have started exploring long-term video understanding. It involves developing methods that can recognize and understand complex activities, events, or interactions that unfold over longer durations of time. This includes tasks like temporal action detection, video summarization, dense video captioning, video grounding, and future event prediction, among others.

## Representation Learning
* Learning from Untrimmed Videos: Self-Supervised Video Representation Learning with Hierarchical Consistency
* Boundary-sensitive Pre-training for Temporal Localization in Videos
* TSP: Temporally-Sensitive Pretraining of Video Encoders for Localization Tasks

## Dense Video Captioning
* Dense-Captioning Events in Videos, ICCV 2017
* Bidirectional Attentive Fusion with Context Gating for Dense Video Captioning, CVPR 2018
* End-to-End Dense Video Captioning with Masked Transformer, CVPR 2018
* Weakly Supervised Dense Event Captioning in Videos, NIPS 2018
* Streamlined Dense Video Captioning, CVPR 2019
* A Better Use of Audio-Visual Cues: Dense Video Captioning with Bi-modal Transformer, BMVC 2020 
* Sketch, Ground, and Refine: Top-Down Dense Video Captioning, CVPR 2021.
* End-to-End Dense Video Captioning with Parallel Decoding, ICCV 2021

#### Video Paragraph Captioning
* Adversarial Inference for Multi-Sentence Video Description, CVPR 2019
* COOT: Cooperative Hierarchical Transformer for Video-Text Representation Learning, NIPS 2020.
* Multimodal Pretraining for Dense Video Captioning, AACL 2020.
* MART: Memory-Augmented Recurrent Transformer for Coherent Video Paragraph Captioning, ACL 2020.
* Towards Diverse Paragraph Captioning for Untrimmed Videos, CVPR 2021.
* Video Paragraph Captioning as a Text Summarization Task, ACL 2021.


## Action Localization
#### Temporal Action Proposals
* Relaxed Transformer Decoders for Direct Action Proposal Generation, ICCV 2021
* Fast learning of temporal action proposal via dense boundary generator, AAAI 2020
* BMN: boundary-matching network for temporal action proposal generation, ICCV 2019
* BSN: boundary sensitive network for temporal action proposal generation, ECCV 2018

#### Temporal Action Localization
* Deep Learning-based Action Detection in Untrimmed Videos: A Survey, Arxiv 2021
* MS-TCT: Multi-Scale Temporal ConvTransformer for Action Detection, CVPR 2022
* Learning To Refactor Action and Co-Occurrence Features for Temporal Action Localization, CVPR 2022
* Learning to Refactor Action and Co-occurrence Features for Temporal Action Localization, CVPR 2022
* An Empirical Study of End-to-End Temporal Action Detection, CVPR 2022
* Modeling multi-label action dependencies for temporal action localization, CVPR 2021
* Coarse-Fine Networks for Temporal Activity Detection in Videos, CVPR 2021
* Learning salient boundary feature for anchor-free temporal action localization, CVPR 2021
* Temporal action localization in untrimmed videos via multi-stage CNNs, CVPR 2016.

#### Audio-Visual Event Localization
Audio-Visual Event Localization in Unconstrained Videos, ECCV 2018.

## Temporal Video Grounding
* The Elements of Temporal Sentence Grounding in Videos: A Survey and Future Directions, Arxiv 2022.

## Untrimmed Video Datasets
| Dataset  | Annotation | Source | Number | Total hours | Tasks | link | Date Released | 
| ----  | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| ActivityNet 1.3 | timestamps + action | Youtube | 20k | - | Action Localization | | |
| ActivityNet Captions | timestamps + captions | Youtube | 20k | - | Dense captioning, video grounding |
| THUMOS | timestamps + action | - | - | - | Action Localization | | |
| YouCook2 | timestamps + captions | Cooking Videos | - | - | Dense captioning | | |
| MovieNet | timestamps + captions + place/action/style tags | Movies | 1.1k | >2h | movie understanding | [MovieNet](https://movienet.site/) | 2020 |
| Charades | timestamps + action labels | Daily Activities | 9.8k | 30s | action recognition, action localization | [Charades](https://prior.allenai.org/projects/charades) | 2017 | 
| Charades-STA | timestamps + captions | Daily Activities| 9.8k | 30s | video grounding | [Charades-STA](https://github.com/jiyanggao/TALL) | 2017
| TACoS | timestamps + captions | Cooking | - | - | video grounding | [TACoS]
| DiDeMo |
| TACoS Multi-Level | timestamps + captions | Cooking | - | - | Dense captioning | [TACoS Multi-Level](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/vision-and-language/tacos-multi-level-corpus)
| VIOLIN |  | YouTube and TV shows| 6.7k / 15.8k clips | | Video-and-Language Inference | [VIOLIN](https://github.com/jimmy646/violin)
| HowTo100M | boundaries + captions |  | 1.22M/136M clips | 582h | pretraining | - |
| YT-temporal180M |  boundaries + captions | - | - | - |  pretraining | - | 

## Untrimmed Video Feature Extraction Tools

* https://github.com/zjr2000/Untrimmed-Video-Feature-Extractor

# TODO
- [ ] add more details of datasets
- [ ] add pretraining, detection and grounding papers
- [ ] add a plot to illustrate differences between untrimmed videos and trimmed videos, and also between different localization tasks
