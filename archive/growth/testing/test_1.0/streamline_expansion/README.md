Important!!!
Two operations: (i) normalization and (ii) streamline formula are not commutating!
First, dr needs to be normalized, and only then we can use streamline formula (with already desired travel length) to obtain dR.

v2:
Checking the geometry (and computing time) of the tree with slightly changed FreeFEM script (commented adaptmesh befor first solution for the field).
v3:
reticuler_0.1 from pip (with replaced [Ref1] in PDE_solver).
v4:
reticuler_1.0, but extender.__streamline_extension without beta -> inf limit and without np.round
v5:
reticuler_0.3 from pip (good geometry!)
v6:
reticuler 1.0 with dR_1 normalization
v7:
reticuler_0.3 with prints
v8:
reticuler 1.0 with dR_1 normalization and extender.__streamline_extension without beta -> inf limit and without np.round
prints
v9, v10:
reticuler_0.3 and reticuler_1.0 with printing (on PC)
v11:
reticuler_1.0 with normalization inside extenders.find_test_dR (before rotation)
v12:
as above, but with beta -> inf limit and with np.round
v13:
as above, but beta<10000 and np.round(...,12)

Important!!!
Two operations: (i) normalization and (ii) streamline formula are not commutating!
First, dr needs to be normalized, and only then we can use streamline formula (with already desired travel length) to obtain dR.