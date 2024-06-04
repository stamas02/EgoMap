# Egomap

# How to use

## Detect Movement
```
    python detect_motion.py -s PATH/TO/VIDEO/FILE.MP4 -d PATH/TO/OUTPUT/FOLDER
```
The script generates a movement.pickle file containing a boolean NumPy array, where each element corresponds to a frame in the source video, indicating whether forward motion was detected. Additionally, it produces a video file that is identical to the original, except that each frame is annotated with the corresponding label displayed as text.

You can manually observe the content of movement.pickle file:
```
    movement_file = "movement.pickle" # replace with the location
    with open(movement_file, 'rb') as file:
        movement = pickle.load(file)
    print(movement)
```

## Extract Features
```
    python extract_bovw_features.py -s PATH/TO/VIDEO/FILE.MP4 -v PATH/TO/FEATURE/EXTRACTOR
```

The script extracts features from each frame to be used for map building. The resulting data is saved in a file located next to the video file, with the same name as the video file but suffixed by "_features.pickle".

"*_features.pickle" conttains a numpy array of size (frame_count, feature_dim).

Pre-trained feature extractor is available in models/bowv.pickle

```
    python extract_bovw_features.py -s PATH/TO/VIDEO/FILE.MP4 -v models/bowv.pickle
```

## Mapping
```
    python map_it.py -s PATH/TO/FEATURE/FILE -m PATH/TO/MOVEMENT/FILE
```

Input files for the parameters -s and -m can be created using the scripts extract_bovw_features.py and detect_motion.py, respectively, as described above.

The map_it.py script will create two files next to the feature file location. The first is result.csv, which contains frame-level predictions. The second is the saved map object. The resulting map object can be examined as follows:

```
    import pickle
    
    # Replace 'map.pkl' with the path to your map file
    map_file = "map.pkl"
    with open(map_file, 'rb') as file:
        map = pickle.load(file)
    
    # List of map stations
    print(map.visit_locations)
    
    # List of views in the first map station
    print(map.visit_locations[0].graph.nodes)
    
    # Transition matrix between learned stations
    print(map.location_transition_mapping)
```


## Training custom Feature Extractor (BOVW)
```
    python learn_vocabulary.py -s PATH/TO/IMAGE/FOLDER -dim 10 -d PATH/TO/OUTPUT/MODEL/FILE
```
To learn more run 
```
    python learn_vocabulary.py --help
```

## Sample Videos

Sample videos of map creation available under folder "sample_video". The two sample videos are from the HSP dataset (seee reference below)

Guzov, Vladimir, et al. "Human poseitioning system (hps): 3d human pose estimation and self-localization in large scenes from body-mounted sensors." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.

### Interpreting sample videos
On the left side of the display, the original video footage plays, capturing the dynamic environment from a first-person perspective. Simultaneously, on the right side, the map learning process unfolds visually through a bipartite graph representation.

In this graph, each node positioned on the left corresponds to a Station node within the constructed map, representing areas visited during the recording. Conversely, nodes positioned on the right side denote Transition nodes, capturing the movement between these locations.
During both transitions between locations and visits to specific areas, nodes within the graph are colour-coded either red or green. The green hue signifies a high likelihood that the corresponding node accurately represents the current state observed in the video. Conversely, nodes coloured red indicate a lower likelihood of a match, suggesting a divergence between the observed state and the existing map representation.

Notably, if all corresponding nodes are shaded red, it suggests that a new node, representing a previously unseen station, is the most probable addition to the map. However, a new node is only appended once the current visit or transition segment concludes.