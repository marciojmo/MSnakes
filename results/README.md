## baseline_50pts

    MIN_DISTANCE_BETWEEN_POINTS = 5  Minimum distance to consider two points overlaped
    MAX_DISTANCE_BETWEEN_POINTS = 50  Maximum distance to insert another point into spline
    SEARCH_KERNEL_SIZE = 7  # The size of the search kernel
    
    n_starting_points = 50  # The number of starting points of the snake
    closed = True  # Indicates if the snake is closed or open
    alpha = 0.5  # The weight of the uniformity energy
    beta = 0.5  # The weight of the curvature energy
    w_line = 0.5  # The weight to the line energy
    w_edge = 0.5  # The weight to the edge energy
    w_term = 0.5  # The weight to the term energy


## experiment_1.1

Added Another energy term describing how far each vertex is away from its nearest line segment.

```
n_starting_points = 50  # The number of starting points of the snake
MAX_DISTANCE_POINT_LINESEG_TO_SNAP = 30  #  To penalize vertex whose distance to nearst line segment is below this threshold 
delta = 0.1  # The weight of the user configured energy
```

## experiment_1.2

```
MAX_DISTANCE_POINT_LINESEG_TO_SNAP = 10  #  To penalize vertex whose distance to nearst line segment is below this threshold 
```

```
# FIXME: would make it tend to get far away from line segments in order to have 0
return np.min(vdists) if np.min(vdists) < self.MAX_DISTANCE_POINT_LINESEG_TO_SNAP else 0
```

## experiment_1.3

```
return np.min(vdists) if np.min(vdists) < self.MAX_DISTANCE_POINT_LINESEG_TO_SNAP else 100
```

## experiment_1.4

```
return np.min(vdists)
```

