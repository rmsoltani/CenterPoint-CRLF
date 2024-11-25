# Radar Fusion Into Lidar-Based Detection

> [**Radar Fusion Into Lidar Based Detection**](https://arxiv.org/abs/2006.11275),            
> Reza Soltani, John Lee, Nathaniel Sigafoos,        
<!-- > *arXiv technical report ([arXiv 2006.11275](https://arxiv.org/abs/2006.11275))*   -->


<!--
    @article{soltani2024lirar,
      title={Radar Fusion Into Lidar-Based Detection},
      author={Soltani, Reza and Lee, John, Sigafoos, Nathaniel},
      journal={CVPR},
      year={2024},
    }
-->

## Abstract
TODO


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