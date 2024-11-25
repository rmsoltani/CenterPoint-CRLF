# Radar Fusion Into Lidar-Based Detection

> [**Radar Fusion Into Lidar Based Detection**](),            
> Reza Soltani, John Lee, Nathaniel Sigafoos,        


<!--
    @article{soltani2024lirar,
      title={Radar Fusion Into Lidar-Based Detection},
      author={Soltani, Reza and Lee, John, Sigafoos, Nathaniel},
      journal={CVPR},
      year={2024},
    }
-->

## Abstract
Perception is one of the core concepts in autonomous driving. This involves the detection and tracking of an autonomous vehicle's (AV) surrounding objects using multiple sensors mounted to the AV. Fusing these sensors and creating multi-modality data increases metric accuracy and improves the system's robustness. This results in a more sophisticated and reliable perception system. While the topis of radar-camera fusion and lidar-camera fusion have been widely studied, the topic of radar-lidar fusion is often avoided. Those that do study it often present radar-lidar fusion as being complicated and difficult to implement properly. While this can be the case, it does not always have to be. In t his paper, we propose LiRAR, an early-fusion approach that allows for the near seamless addition of radar into models designed around lidar. We evaluate the effectiveness of this approach on the nuScenes dataset, where we show up to a 0.71% increase in mAP across all classes, achieving a score of 59.94. In addition, we analysis the strengths and weaknesses of this approach, and discuss methods that could be used to expand upon it.


## Main results


#### 3D detection on nuScenes val set 

|         |  MAP ↑  | NDS ↑ |
|---------|---------|-------|
|  LiRAR  |  59.94  | 67.14 |   
   

All results are tested on a Nvidia A100 GPU with batch size 4.


## Usage

### Installation

Please refer to [INSTALL](docs/INSTALL.md) to set up libraries needed for distributed training and sparse convolution.

### Benchmark Evaluation and Training 

Please refer to [NUSC](docs/NUSC.md) to prepare the data. Then follow the instruction there to reproduce our detection and tracking results. All detection configurations are included in [configs](configs).


## License

LiRAR is release under MIT license (see [LICENSE](LICENSE)). It is developed based on a forked version of [CenterPoint](https://github.com/tianweiy/CenterPoint). See the [NOTICE](docs/NOTICE) for details. Note that the nuScenes dataset is under non-commercial licenses.