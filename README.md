#  Awesome Long-Term Video Understanding
  
  
Real-world videos are usually long, untrimmed, and contain several actions (events). Traditionally, video understanding has focused on short-term analysis, such as action recognition, video object detection/segmentation, or scene understanding in individual video frames or short video clips. However, with the advent of more advanced technologies and the increasing availability of large-scale video datasets, researchers have started exploring long-term video understanding. It involves developing methods that can recognize and understand complex activities, events, or interactions that unfold over longer durations of time. This includes tasks like temporal action detection, dense video captioning, video grounding, future video prediction, and video summarization, among others.
  
This repository curates a collection of research works specifically focused on long-term videos. This is an active repository and your contributions are always welcome!
  
  
- [Awesome Long-Term Video Understanding](#awesome-long-term-video-understanding)
  - [Representation Learning](#representation-learning)
  - [Efficient Modeling in Long-Term Videos](#efficient-modeling-in-long-term-videos)
  - [Action Localization](#action-localization)
    - [Temporal Action Localization](#temporal-action-localization)
    - [Audio-Visual Event Localization](#audio-visual-event-localization)
  - [Dense Video Captioning](#dense-video-captioning)
  - [Temporal Video Grounding](#temporal-video-grounding)
  - [Long-Term Video Prediction](#long-term-video-prediction)
  - [Other Tasks](#other-tasks)
  - [Datasets \& Tools](#datasets--tools)
    - [Long-Term (Untrimmed) Video Datasets](#long-term-untrimmed-video-datasets)
    - [Video Feature Extractor](#video-feature-extractor)
  - [TODO](#todo)
  
##  Representation Learning
  
* HierVL: Learning Hierarchical Video-Language Embeddings, CVPR 2023.
* Learning Grounded Vision-Language Representation for Versatile Understanding in Untrimmed Videos, arXiv 2023.
* Temporal Alignment Networks for Long-Term Video, CVPR 2022.
* Learning from Untrimmed Videos: Self-Supervised Video Representation Learning with Hierarchical Consistency, CVPR 2022.
* Boundary-sensitive Pre-training for Temporal Localization in Videos, ICCV 2021.
* TSP: Temporally-Sensitive Pretraining of Video Encoders for Localization Tasks, ICCVW 2021.
* COOT: Cooperative Hierarchical Transformer for Video-Text Representation Learning, NIPS 2020.
  
##  Efficient Modeling in Long-Term Videos
  
  
* MeMViT: Memory-Augmented Multiscale Vision Transformer for Efficient Long-Term Video Recognition, CVPR 2022.
* Long-term feature banks for detailed video understanding, CVPR 2019.
* (TODO)
  
##  Action Localization
  
###  Temporal Action Localization
  
***Resources:***
* [Github repo: Awesome Weakly Supervised Temporal Action Localization](https://github.com/Pilhyeon/Awesome-Weakly-Supervised-Temporal-Action-Localization )
* [Github repo: Awesome Temporal Action Localization](https://github.com/Alvin-Zeng/Awesome-Temporal-Action-Localization )
* [Github repo: Awesome Temporal Action Detection Temporal Action Proposal Generation](https://github.com/zhenyingfang/Awesome-Temporal-Action-Detection-Temporal-Action-Proposal-Generation )
* [survey] A survey on temporal action localization, IEEE Access 2020.
  
***Latest Papers (after 2023)***
* ETAD: Training Action Detection End to End on a Laptop, CVPR 2023.
* TriDet: Temporal Action Detection With Relative Boundary Modeling, CVPR 2023.
* Distilling Vision-Language Pre-training to Collaborate with Weakly-Supervised Temporal Action Localization, CVPR 2023.
* Re2TAL: Rewiring Pretrained Video Backbones for Reversible Temporal Action Localization, CVPR 2023.
* Boosting Weakly-Supervised Temporal Action Localization With Text Information, CVPR 2023.
* Proposal-based Multiple Instance Learning for Weakly-supervised Temporal Action Localization, CVPR 2023.
  
***Representitive Papers (before 2023):***
* ActionFormer: Localizing Moments of Actions with Transformers, ECCV 2022.
* G-tad: Sub-graph localization for temporal action detection, CVPR 2020.
* Relaxed Transformer Decoders for Direct Action Proposal Generation, ICCV 2021.
* Fast learning of temporal action proposal via dense boundary generator, AAAI 2020.
* BMN: boundary-matching network for temporal action proposal generation, ICCV 2019.
* Graph convolutional networks for temporal action localization, ICCV 2019.
* Rethinking the faster r-cnn architecture for temporal action localization, CVPR 2018.
* BSN: boundary sensitive network for temporal action proposal generation, ECCV 2018.
* SST: Single-Stream Temporal Action Proposals, CVPR 2017.
* Temporal action detection with structured segment networks, ICCV 2017.
* R-c3d: Region convolutional 3d network for temporal activity detection, ICCV 2017.
* CDC: Convolutional-De-Convolutional Networks for Precise Temporal Action Localization in Untrimmed Videos, CVPR 2017.
* TURN TAP: Temporal Unit Regression Network for Temporal Action Proposals, ICCV 2017.
* Single Shot Temporal Action Detection, ACM MM 2017.
* Temporal action localization in untrimmed videos via multi-stage CNNs, CVPR 2016.
* Temporal Localization of Actions with Actoms, TPAMI 2014.
  
  
###  Audio-Visual Event Localization
  
* Dense-Localizing Audio-Visual Events in Untrimmed Videos: A Large-Scale Benchmark and Baseline, CVPR 2023. [[code]](https://github.com/ttgeng233/UnAV )
* Towards Long Form Audio-visual Video Understanding, Arxiv 2023.
* Audio-Visual Event Localization in Unconstrained Videos, ECCV 2018.
  
  
##  Dense Video Captioning
  
* Learning Grounded Vision-Language Representation for Versatile Understanding in Untrimmed Videos, arXiv 2023.
* Vid2Seq: Large-Scale Pretraining of a Visual Language Model for Dense Video Captioning, CVPR 2023. [[code]](https://github.com/google-research/scenic/tree/main/scenic/projects/vid2seq )
* Unifying event detection and captioning as sequence generation via pre-training, ECCV 2022.
* End-to-end Dense Video Captioning as Sequence Generation, Coling 2022.
* End-to-End Dense Video Captioning with Parallel Decoding, ICCV 2021. [[code]](https://github.com/ttengwang/PDVC )
* Sketch, Ground, and Refine: Top-Down Dense Video Captioning, CVPR 2021.
* Streamlined Dense Video Captioning, CVPR 2019.
* Bidirectional Attentive Fusion with Context Gating for Dense Video Captioning, CVPR 2018.
* End-to-End Dense Video Captioning with Masked Transformer, CVPR 2018.
* Weakly Supervised Dense Event Captioning in Videos, NIPS 2018.
* Dense-Captioning Events in Videos, ICCV 2017.
  
***Video Paragraph Captioning*:**
* Adversarial Inference for Multi-Sentence Video Description, CVPR 2019
* COOT: Cooperative Hierarchical Transformer for Video-Text Representation Learning, NIPS 2020.
* Multimodal Pretraining for Dense Video Captioning, AACL 2020.
* MART: Memory-Augmented Recurrent Transformer for Coherent Video Paragraph Captioning, ACL 2020.
* Towards Diverse Paragraph Captioning for Untrimmed Videos, CVPR 2021.
* Video Paragraph Captioning as a Text Summarization Task, ACL 2021.
  
  
##  Temporal Video Grounding
  
***Resources*:**
* [survey] The Elements of Temporal Sentence Grounding in Videos: A Survey and Future Directions, Arxiv 2022.
* [survey] A survey on temporal sentence grounding in videos, ACM TOMM 2023.
* [survey] A survey on video moment localization, ACM Computing Surveys, 2023.
* [Github repo: Awesome Video Grounding](https://github.com/NeverMoreLCH/Awesome-Video-Grounding )
  
***Representative Papers (before 2023)*:**
* Negative sample matters: A renaissance of metric learning for temporal grounding, AAAI 2022.
* Local-global video-text interactions for temporal grounding, CVPR 2020.
* Dense regression network for video grounding, CVPR 2020.
* Learning 2d temporal adjacent networks for moment localization with natural language, AAAI 2020.
* To Find Where You Talk: Temporal Sentence Localization in Video with Attention Based Location Regression, AAAI 2020.
* Man: Moment alignment network for natural language moment retrieval via iterative graph adjustment, CVPR 2019.
* Semantic conditioned dynamic modulation for temporal sentence grounding in videos, NIPS 2019.
* Temporally grounding natural sentence in video, EMNLP 2018.
* TALL: Temporal activity localization via language query, ICCV 2017.
  
  
***Latest Papers (after 2023)*:**
* Learning Grounded Vision-Language Representation for Versatile Understanding in Untrimmed Videos, Arxiv 2023.
* ProTeG´e: Untrimmed Pretraining for Video Temporal Grounding by Video ´ Temporal Grounding, CVPR 2023.
* (TODO)
  
##  Long-Term Video Prediction
  
* Revisiting Hierarchical Approach for Persistent Long-Term Video Prediction, ICLR 2021.
* Hierarchical Long-term Video Prediction Without Supervision, ICML 2018. [[paper]](https://arxiv.org/abs/1806.04768 )
* Learning to Generate Long-term Future via Hierarchical Prediction, ICML 2017.
  
  
  
##  Other Tasks
  
* **[Spatiotemporal Grounding]** Relational Space-Time Query in Long-Form Videos, CVPR 2023.
* **[Tracking]** XMem: Long-Term Video Object Segmentation with An Atkinson-Shiffrin Memory Model, ECCV 2022.
* **[VQA]** Long-Term Video Question Answering Via Multimodal Hierarchical Memory Attentive Networks, TCSVT 2021.
* **[Video summarization]** Video Summarization with Long Short-term Memory, ECCV 2016.
  
  
##  Datasets & Tools
  
  
###  Long-Term (Untrimmed) Video Datasets
  
| Dataset  | Annotation | Source | Number | Duration | Tasks | link | Date Released | 
| ----  | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| ActivityNet 1.3 | timestamps + action | Youtube | 20k | - | Action Localization | | |
| ActivityNet Captions | timestamps + captions | Youtube | 20k | - | Dense captioning, video grounding |
| THUMOS | timestamps + action | - | - | - | Action Localization | | |
| YouCook2 | timestamps + captions | Cooking Videos | - | - | Dense captioning | | |
| MovieNet | timestamps + captions + place/action/style tags | Movies | 1.1k | >2h | movie understanding | [MovieNet](https://movienet.site/ ) | 2020 |
| Charades | timestamps + action labels | Daily Activities | 9.8k | 30s | action recognition, action localization | [Charades](https://prior.allenai.org/projects/charades ) | 2017 | 
| Charades-STA | timestamps + captions | Daily Activities| 9.8k | 30s | video grounding | [Charades-STA](https://github.com/jiyanggao/TALL ) | 2017
| TACoS | timestamps + captions | Cooking | - | - | video grounding | [TACoS]
| DiDeMo |
| TACoS Multi-Level | timestamps + captions | Cooking | - | - | Dense captioning | [TACoS Multi-Level](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/vision-and-language/tacos-multi-level-corpus )
| VIOLIN |  | YouTube and TV shows| 6.7k / 15.8k clips | | Video-and-Language Inference | [VIOLIN](https://github.com/jimmy646/violin )
| HowTo100M | boundaries + captions |  | 1.22M/136M clips | 582h | pretraining | - |
| YT-temporal180M |  boundaries + captions | - | - | - |  pretraining | - | 
|EPIC-KITCHENS-100 |
|HD_VILA_100M|
  
  
  
###  Video Feature Extractor
  
  
* https://github.com/zjr2000/Untrimmed-Video-Feature-Extractor
  
##  TODO
- [ ] add efficient modeling papers and grounding papers
- [ ] add more details of datasets
- [ ] add a plot to illustrate differences between untrimmed videos and trimmed videos, and also between different localization tasks
  
