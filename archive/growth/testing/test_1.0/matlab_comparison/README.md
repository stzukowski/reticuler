Trying to obtain the same trees as in MATLAB. Problems:
1. passing geometry to FreeFEM with .msh file
2. constructing script based on graph.P (different order of adding points in buildmesh...)
3. constructing script based on the branches - precision of the floating points; trying to match them:
freefem2_v3 - .12e
freefem2_v4 - .12f
4. Even with above the difference can come from calculating the tip angle... (freefem2_v4); trying to decrease the precision in constructing the freefem script:
freefem2_v5 - .6e in building the freefem script