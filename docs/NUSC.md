## Getting Started with CenterPoint on nuScenes
Modified from [CenterPoint](https://github.com/tianweiy/CenterPoint/blob/3cf7d870537e287c99b43b68636ea392a5e6f519/docs/NUSC.md)'s original document.

### Prepare data

#### Download data and organise as follows

```
# For nuScenes Dataset         
└── NUSCENES_DATASET_ROOT
       ├── samples       <-- key frames
       ├── sweeps        <-- frames without annotation
       ├── maps          <-- unused
       ├── v1.0-trainval <-- metadata
```

Create a symlink to the dataset root 
```bash
mkdir data && cd data
ln -s DATA_ROOT 
mv DATA_ROOT nuScenes # rename to nuScenes
```
Remember to change the DATA_ROOT to the actual path in your system. 


#### Create data

Data creation should be under the gpu environment.

```
# nuScenes
python -m tools.create_data nuscenes_data_prep --root_path=NUSCENES_TRAINVAL_DATASET_ROOT --version=v1.0-trainval --nsweeps=10 --modalities=[lidar,radar]
```

In the end, the data and info files should be organized as follows

```
# For nuScenes Dataset 
└── LiRAR
       └── data    
              └── nuScenes 
                     ├── samples       <-- key frames
                     ├── sweeps        <-- frames without annotation
                     ├── maps          <-- unused
                     |── v1.0-trainval <-- metadata and annotations
                     |── infos_train_10sweeps_withvelo_radar_filter_True.pkl <-- train annotations
                     |── infos_val_10sweeps_withvelo_radar_filter_True.pkl <-- val annotations
                     |── dbinfos_train_10sweeps_withvelo_radar.pkl <-- GT database info files
                     |── gt_database_10sweeps_withvelo_radar <-- GT database 
```

### Train & Evaluate in Command Line

**Now we only support training and evaluation with gpu. Cpu only mode is not supported.**

Use the following command to start training using 1 GPU. The models and logs will be saved to ```work_dirs/CONFIG_NAME``` 

```bash
python -m tools.train CONFIG_PATH
```


For testing with one gpu and see the inference time,

```bash
python -m tools.dist_test CONFIG_PATH --work_dir work_dirs/CONFIG_NAME --checkpoint work_dirs/CONFIG_NAME/latest.pth --speed_test 
```
