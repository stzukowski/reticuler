import os
import subprocess
import numpy as np
from datetime import datetime
from tempfile import NamedTemporaryFile

class FreeFEM:
    """A parent class for PDE solvers based on the finite element method implemented in FreeFEM [Ref2]_.

    Attributes
    ----------
    equation : int
        - 0: Laplace
        - 1: Poisson
        - 2: elasticity (ThickFingers)
    eta : float
        The growth exponent (v = a1**``eta'').
        High values increase competition between the branches.
        Low values stabilize the growth.
    ds : float
        A distance over which the fastest branch in the network
        will move in each timestep.
    is_script_saved : bool
        If True, the FreeFEM script is saved to a file.

    References
    ----------
    .. [Ref2] https://freefem.org/

    """

    def __init__(self, equation, eta, ds, is_script_saved):
        """Initialize FreeFEM.

        Parameters
        ----------
        equation : int
        eta : float
        ds : float
        is_script_saved : bool

        Returns
        -------
        None.

        """
        self.equation = equation
        self.eta = eta
        self.ds = ds
        self.is_script_saved = is_script_saved

    def inNd(self, a, b, assume_unique=False):
        """ Based on https://stackoverflow.com/questions/16216078/test-for-membership-in-a-2d-numpy-array
        and http://stackoverflow.com/a/16973510/190597 (Jaime, 2013-06)
        View the array as dtype np.void (bytes). The items along the last axis are
        viewed as one value. This allows comparisons to be performed on the entire row.
        """
        def asvoid(arr):
            arr = np.ascontiguousarray(arr)
            if np.issubdtype(arr.dtype, np.floating):
                """ np.array([-0.]).view(np.void) != np.array([0.]).view(np.void)
                Adding 0. converts -0. to 0.
                """
                arr += 0.
            return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[-1])))
        a = asvoid(a)
        b = asvoid(b)
        return np.isin(a, b, assume_unique)

    def arr2str(self, arr):
        return np.array2string(arr, separator=",", formatter={'float_kind': lambda x: f"{x:.6e}"},max_line_width=np.inf,threshold=np.inf).replace("\n", "")

    def array_from_string(self, out_string, key):
        ind_0 = out_string.find(key.encode("ascii"))+len(key)
        ind_1 = out_string.find(key.encode("ascii")+b"end")
        return np.fromstring(out_string[ind_0:ind_1], sep="\t")[1:]

    def prepare_contour(self, border_contour, inside_buildmesh, i, points, label, border_name="contour"):
        for j, pair in enumerate(zip(points, points[1:])):
            x0 = pair[0][0]
            y0 = pair[0][1]
            x1 = pair[1][0]
            y1 = pair[1][1]

            border_contour = (
                border_contour
                + "border {b_n}{i}connection{j}(t=0, 1){{x={x0:.6e}+t*({ax:.6e});y={y0:.6e}+t*({ay:.6e}); label={label};}}\n".format(
                    b_n=border_name, i=i, j=j, x0=x0, ax=x1 - x0, y0=y0, ay=y1 - y0, label=label[j] if not np.isscalar(label) else label
                )
            )

            inside_buildmesh = (
                inside_buildmesh + " {b_n}{i}connection{j}(1) +".format(b_n=border_name, i=i, j=j)
            )    
        
        return border_contour, inside_buildmesh
        
    def prepare_contour_list(self, border_contour, inside_buildmesh, i, points, label, ns_border=1, border_name="contour", i_tsh=1023):
        n_points = len(points)
        if np.isscalar(ns_border):
            ns_border = np.ones(n_points-1, dtype=int) * ns_border

        border_contour = (
            border_contour
            + f"real[int] {border_name}{i}X({n_points}); real[int] {border_name}{i}Y({n_points}); int[int] {border_name}{i}N({n_points-1});"
            )
        if not np.isscalar(label):
            border_contour = (
                border_contour
                + f" int[int] {border_name}{i}BC({n_points-1});"
            )
        for j in range((n_points-1)//i_tsh+1):
            ind0 = j*i_tsh
            ind1 = (j+1)*i_tsh

            border_contour = (
                border_contour
                + f"\n{border_name}{i}X({ind0}:{ind1})={self.arr2str(points[ind0:ind1,0])};\n{border_name}{i}Y({ind0}:{ind1})={self.arr2str(points[ind0:ind1,1])};\n"
                )
            
            if ind0 < (n_points-1):
                if not np.isscalar(ns_border):
                    border_contour = (
                    border_contour
                    + f"{border_name}{i}N({ind0}:{ind1})={self.arr2str(ns_border[ind0:ind1])};\n"
                    )            
                if not np.isscalar(label):
                    border_contour = (
                    border_contour
                    + f"{border_name}{i}BC({ind0}:{ind1})={self.arr2str(label[ind0:ind1])};\n"
                    )   
        
        if not np.isscalar(label):
            label = f"{border_name}{i}BC(i)"
        
        border_contour = (
            border_contour
            + f"border {border_name}{i}(t=0, 1; i){{ x = {border_name}{i}X(i)*(1-t) + {border_name}{i}X(i+1)*t; y = {border_name}{i}Y(i)*(1-t) + {border_name}{i}Y(i+1)*t; label={label};}}\n\n"
        )

        inside_buildmesh = (
            inside_buildmesh + f" {border_name}{i}({border_name}{i}N) +"
        )
        return border_contour, inside_buildmesh

    def prepare_script_box(self, connections_bc, points, points_per_unit_len=0.5):
        """Return part of the FreeFEM script with the geometry of the box."""
        
        ns_border = np.max(( np.ones(connections_bc.shape[0]), \
                points_per_unit_len*np.linalg.norm(np.diff(points[connections_bc[:,:2]], axis=1)[:,0], axis=1) ),
               axis=0 ).astype(int)
            
        border_box = "\nreal buildTime0=clock();\n\n"
        inside_buildmesh_box = ""
        border_box, inside_buildmesh_box = self.prepare_contour_list(border_box, inside_buildmesh="", i="", points=np.vstack((points, points[0])), label=connections_bc[:,2], ns_border=ns_border, border_name="box")
        # border_box, inside_buildmesh_box = prepare_contour(border_box, inside_buildmesh="", i="", points=np.vstack((points, points[0])), label=connections_bc[:,2], border_name="box")
        return border_box, inside_buildmesh_box

    def run_freefem(self, script):
        """Run FreeFEM script."""
        
        if self.is_script_saved:
            # script_name = f"script_{id(self)}_{datetime.now().strftime("%Y_%m_%d-%p%I_%M_%S")}.edp"
            script_name = self.system.exp_name + f"_{datetime.now().strftime('%Y_%m_%d-%p%I_%M_%S')}.edp"
            with open(script_name, "w") as edp_file:
                edp_file.write(script)        
        else:
            temporary_files = []  # to close at the end
            with NamedTemporaryFile(suffix=".edp", mode="w", delete=False) as edp_file:
                edp_file.write(script)
                temporary_files.append(edp_file)

        cmd = [
            "FreeFem++",
            "-nw",
            "-nc",
            "-v",
            "0",
            "-f",
            "{file_name}".format(file_name=edp_file.name),
        ]
        out_freefem = subprocess.run(
            args=cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # close temporary files
        if not self.is_script_saved:
            for tmp_file in temporary_files:
                tmp_file.close()
                os.unlink(tmp_file.name)
                
        if out_freefem.returncode or "nan" in out_freefem.stdout.decode():
            script_name = self.system.exp_name + f"_{datetime.now().strftime('%Y_%m_%d-%p%I_%M_%S')}_failed.edp"
            print(f"\nFreeFem++ failed... Saving the script: {script_name}\n")
            print("stdout:", out_freefem.stdout.decode())
            print("stderr:", out_freefem.stderr.decode())
            with open(script_name, "w") as edp_temp_file:
                edp_temp_file.write(script)
            
        return out_freefem