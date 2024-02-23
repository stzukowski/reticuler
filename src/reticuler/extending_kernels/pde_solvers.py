"""Solvers for PDEs describing the field around the network.

Classes:
    FreeFEM
    
"""

import numpy as np
import scipy
import subprocess
import os.path
import os
from platform import system
from tempfile import NamedTemporaryFile
import textwrap

import shapely
from shapely.ops import linemerge 
from shapely.geometry import MultiLineString, LinearRing

from reticuler.system import LEFT_WALL_PBC, RIGHT_WALL_PBC, DIRICHLET, CONSTANT_FLUX

def rotation_matrix(angle):
    """Construct a matrix to rotate a vector by an ``angle``.

    Parameters
    ----------
    angle : float

    Returns
    -------
    array
        An 2-2 array.

    Examples
    --------
    >>> rot = rotation_matrix(angle)
    >>> rotated_vector = np.dot(rot, vector)

    """
    return np.array(
        [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
    )

def arr2str(arr):
    return np.array2string(arr, separator=",", precision=6,max_line_width=np.inf,threshold=np.inf).replace("\n", "")

def prepare_contour(border_contour, inside_buildmesh, i, points, label, border_name="contour"):
    for j, pair in enumerate(zip(points, points[1:])):
        x0 = pair[0][0]
        y0 = pair[0][1]
        x1 = pair[1][0]
        y1 = pair[1][1]

        border_contour = (
            border_contour
            + "border {b_n}{i}connection{j}(t=0, 1){{x={x0:.6e}+t*({ax:.6e});y={y0:.6e}+t*({ay:.6e}); label={label};}}\n".format(
                b_n=border_name, i=i, j=j, x0=x0, ax=x1 - x0, y0=y0, ay=y1 - y0, label=label
            )
        )

        inside_buildmesh = (
            inside_buildmesh + " {b_n}{i}connection{j}(1) +".format(b_n=border_name, i=i, j=j)
        )    
    
    return border_contour, inside_buildmesh

def prepare_contour1(border_contour, inside_buildmesh, i, points, label, i_tsh=1023):
    border_contour = (
        border_contour
        + "real[int] contour{i}X({n}); real[int] contour{i}Y({n}); int[int] contour{i}N({n}-1); contour{i}N=1;".format(i=i, n=len(points))
        )

    for j in range((len(points)-1)//i_tsh+1):
        border_contour = (
            border_contour
            + "\ncontour{i}X({ind0}:{ind1})={pointsX};\ncontour{i}Y({ind0}:{ind1})={pointsY};\n".format( \
                        i=i, ind0=j*i_tsh, ind1=(j+1)*i_tsh, \
                        pointsX=arr2str(points[j*i_tsh:(j+1)*i_tsh,0]),
                        pointsY=arr2str(points[j*i_tsh:(j+1)*i_tsh,1])
                            ) 
            )
    border_contour = (
        border_contour
        + "border contour{i}(t=0, 1; i){{ x = contour{i}X(i)*(1-t) + contour{i}X(i+1)*t; y = contour{i}Y(i)*(1-t) + contour{i}Y(i+1)*t; label={label};}}\n\n".format(
            i=i, label=label
        )
    )

    inside_buildmesh = (
        inside_buildmesh + " contour{i}(contour{i}N) +".format(i=i)
    )
    return border_contour, inside_buildmesh


class FreeFEM:
    """PDE solver based on the finite element method implemented in FreeFEM [Ref2]_.
    Forges flux info into tip trajectory with the streamline algorithm [Ref1]_.

    Attributes
    ----------
    equation : int, default 0
        - 0: Laplace
        - 1: Poisson
    eta : float, default 1.0
        The growth exponent (v=a1**eta).
        High values increase competition between the branches.
        Low values stabilize the growth.
    ds : float, default 0.01
        A distance over which the fastest branch in the network
        will move in each timestep.   
    flux_info : array
        An array of a1a2a3 coefficients for each tip in the network.
    bifurcation_type : int, default 0
        - 0: no bifurcations
        - 1: a1 bifurcations (velocity criterion)
        - 2: a3/a1 bifurcations (bimodality criterion)
        - 3: random bifurcations
    bifurcation_thresh : float, default depends on bifurcation_type
        Threshold for the bifurcation criterion.
    bifurcation_angle : float, default 2pi/5
        Angle between the daughter branches after bifurcation.
        Default angle (72 degrees) corresponds to the analytical solution
        for fingers in a diffusive field.
    inflow_thresh : float, default 0.05
        Threshold to put asleep the tips with less than ``inflow_thresh``
        of max flux/velocity.
    distance_from_bif_thresh : float, default 2.1*``ds``
        A minimal distance the tip has to move from the previous bifurcations
        to split again. 


    References
    ----------
    .. [Ref1] "Through history to growth dynamics: backward evolution of spatial networks",
        S. Żukowski, P. Morawiecki, H. Seybold, P. Szymczak, Sci Rep 12, 20407 (2022). 
        https://doi.org/10.1038/s41598-022-24656-x
        
    .. [Ref2] https://freefem.org/

    """

    def __init__(
            self, 
            network,
            equation=0,
            eta=1.0,
            ds=0.01,
            bifurcation_type=0,
            bifurcation_thresh=None,
            bifurcation_angle=2 * np.pi / 5,
            inflow_thresh=0.05,
            distance_from_bif_thresh=None,
        ):
        """Initialize FreeFEM.

        Parameters
        ----------
        network : Network
        equation : int, default 0
        eta : float, default 1.0
        ds : float, default 0.01
        bifurcation_type : int, default 0
        bifurcation_thresh : float, default 0
        bifurcation_angle : float, default 2pi/5
        inflow_thresh : float, default 0.05

        Returns
        -------
        None.

        """
        self.equation = equation
        self.eta = eta
        self.ds = ds
        self.flux_info = []
        
        self.bifurcation_type = bifurcation_type  # no bifurcations, a1, a3/a1, random
        self.bifurcation_thresh = bifurcation_thresh
        if bifurcation_thresh is None:
            self.bifurcation_thresh = 0
            if self.bifurcation_type == 1:
                self.bifurcation_thresh = 0.8  # a1 bifurcations
            if self.bifurcation_type == 2:
                self.bifurcation_thresh = -0.1  # a3/a1 bifurcations
            if self.bifurcation_type == 3:
                self.bifurcation_thresh = 3 * ds
                # random bifurcations: bif_probability
        self.bifurcation_angle = bifurcation_angle  # 2*np.pi/5

        # less than `inflow_thresh` of max flux/velocity puts branches asleep
        self.inflow_thresh = inflow_thresh
        self.distance_from_bif_thresh = 2.1 * ds if distance_from_bif_thresh is None else distance_from_bif_thresh
        
        # parts of the script
        self.__script_init = textwrap.dedent(
            """
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // INITIALISATION
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            verbosity = 0;
            
            real time0=clock();
            
            // Defining the base vectors of the analytical field around a tip
            func real BaseVector(int nf, complex zf)
            {
              real result=0;
              
              if (nf%2==0) 
                result = -imag(zf^(nf/2.));
              else 
                result = real(zf^(nf/2.));
              return result;
            }
            
            // Adaptation around the tip
            func real tipfield( real[int] X, real[int] Y, real sigma, int nTips)
            {
            real err=0;
            for(int i=0;i<nTips;i++)
            	{
            		real rsq=(x-X(i))^2 + (y-Y(i))^2;
            		
            		if (rsq==0)
            			err+=1-erf(1);
            		else //if (rsq<2.*square(sigma))
            			err+=1 - 0.3*erf(1) + 0.3*erf(sqrt(rsq/(2*sigma^2)));
            		// else
            		//	err+=1;
            	}
            return err;
            }
            
            // Projection of a mesh around the tip
            func int inCircle (real x, real y, real R)
            {
                if (x^2+y^2<R^2) return 1;
                else return 0;
            }
            
            // Counting vertices in the circle around the tip
            real x0=0., y0=0.;
            func int[int] countNvAroundTips (real R, mesh Th, int nbVertices, int nbTips, real[int] X, real[int] Y)
            {
            	int[int] nvAroundTips(nbTips);
            	for(int i=0;i<nbTips;++i)
            	{
            		x0=X(i);
            		y0=Y(i);
            		int nvAroundTip = 0;
            		for (int i = 0; i < nbVertices; i++)
            			if ((x0-Th(i).x)^2 + (y0-Th(i).y)^2 < R^2) 
            				nvAroundTip += 1;		
            		nvAroundTips(i) = nvAroundTip;
            	};
            	
            	return nvAroundTips;
            }
            """
        )
        
        self.__script_border_box, self.__script_inside_buildmesh_box = \
            self.prepare_script_box(network.box.connections_bc(), \
                                    network.box.points)
        
        self.__script_problem_Laplace = textwrap.dedent(
            """
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // DEFINING PROBLEM AND equation TO SOLVE
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            fespace Vh(Th,P2);
            Vh u,v;
            
            real dirichletOut = 2; // also under a_i integrals
            problem potential(u,v,solver=sparsesolver)=
                     int2d(Th)(dx(u)*dx(v) + dy(u)*dy(v))
                         -int1d(Th,3)(v)  // constant flux
            			 +on(4,u=dirichletOut) // constant field
            			 // -int2d(Th)(v) // rain in domain
                         +on(1,u=0);
            """
        )

        self.__script_problem_Poisson = textwrap.dedent(
            """ 
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // DEFINING PROBLEM AND equation TO SOLVE
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            fespace Vh(Th,P2);
            Vh u,v;
            
            real dirichletOut = 2; // also under a_i integrals
            problem potential(u,v,solver=sparsesolver)=
                     int2d(Th)(dx(u)*dx(v) + dy(u)*dy(v))
                         -int2d(Th)(v) 
            			 -int1d(Th,3)(v)  // constant flux
                         +on(4,u=dirichletOut) // constant field
                         +on(1,u=0);
            """
        )

        self.__script_adaptmesh = textwrap.dedent(
            """
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // ADAPTING THE MESH AND SOLVING FOR THE FIELD
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////        

            // counting cells around the tips
            real R=0.01; // circle around the tip over which field is integrated
            int[int] nvAroundTips = countNvAroundTips (3.*R, Th, Th.nv, nbTips, X, Y);
            
            // First adaptation
            real firstAdaptTime0=clock();
            // Th = adaptmesh(Th,5.*tipfield(X,Y,3.*R,nbTips),nbvx=500000,nbsmooth=100,iso=true);
            Th = adaptmesh(Th,1,nbvx=500000,hmax=0.1,nbsmooth=100,iso=true,ratio=1.8);
            real firstAdaptTime=clock() - firstAdaptTime0;
            // plot(Th, wait=true);
            
            // Solving the problem for the first time
            real firstRunTime0=clock();
            potential;
            // cout<<"First solve completed."<<endl;
            real firstRunTime=clock() - firstRunTime0;
            
            // Adaptation loop
            real adaptTime0=clock();
            // cout << endl << endl << "Adaptation..." << endl;
            fespace Vh0(Th,P0);
            Vh0 h=1;
            real error=0.02;
            int adaptCounter=1;
            while(nvAroundTips.min < 250 || adaptCounter<=3)
            {
            	// cout << "Adaptation step: " << adaptCounter << ", h[].min = " << h[].min;
            	// cout << ", nvAroundTip.min = " << nvAroundTips.min << endl;
            	potential;
            	Th=adaptmesh(Th,[u, 20.*tipfield(X,Y,3.*R,nbTips)],err=error,nbvx=1000000,iso=true,ratio=2,hmin=1e-5);
            	error = 0.5*error;
            	u=u;
            	h=hTriangle; // the triangle size
            	nvAroundTips = countNvAroundTips (3.*R, Th, Th.nv, nbTips, X, Y);
            	adaptCounter++;
            }
            
            // cout << endl << "Adaptation finished." << " h[].min = " << h[].min;
            // cout << ", nvAroundTip.min = " << nvAroundTips.min << endl;
            
            // solving with adapted mesh
            potential;
            // cout << "Problem solved." << endl;
            // plot(u, wait=true, fill=true);
            
            real adaptTime=clock() - adaptTime0;
            """
        )

        self.__script_integrate_a1a2a3 = textwrap.dedent(
            """
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // INTEGRATING THE FIELD TO GET a_i COEFFICIENTS
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////        
            real coeffTime0=clock();
            // cout << endl << endl << "Finding the Tip coefficients..." << endl;
            
            mesh Ph;
            real[int] a(3); // list of coefficients of the expansion
            int exponant=2; // precision of the exponential
            // ofstream freefemOutput("{file_name}");
            
            cout.precision(12);
            cout << "kopytko ";
            for(int i=0;i<nbTips;++i)
            {
                // cout << "Processing Tip " << i << " ";   
                x0=X(i);y0=Y(i);
                // cout << "(x0, y0) = (" << x0 << ", " <<y0<< "), angle = " << angle(i) << endl;
                
            	// cout << "Projecting... Th.nv = " << Th.nv;
                Ph=trunc(Th,(sqrt((x-x0)^2+(y-y0)^2) < 3*R)); 
            	// cout << ", Ph.nv = " << Ph.nv << endl;
            	
                for(int order=1; order<=a.n; ++order){ 
                    a[order-1]=
                    int2d(Ph)( u*exp(-(sqrt((x-x0)^2 + (y-y0)^2)/R)^exponant)
            		*BaseVector(order,exp(-angle(i)*1i)*( (x-x0) + (y-y0)*1i) ) );
                    
                    if (BC(i)==4) 
                    {
                        a[order-1]-=int2d(Ph)( dirichletOut*exp(-(sqrt((x-x0)^2 + (y-y0)^2)/R)^exponant)
                		*BaseVector(order,exp(-angle(i)*1i)*( (x-x0) + (y-y0)*1i) ) );
                        a[order-1]*=-1;
                    }
                        
                    a[order-1]/=(int2d(Ph)(exp(-(sqrt((x-x0)^2 + (y-y0)^2)/R)^exponant)
            		*square(BaseVector(order,exp(-angle(i)*1i)*( (x-x0) + (y-y0)*1i) ) )));
            		
            		cout << a[order-1] << ",";
                    // cout << "a(" << order << ") = " << a[order-1] << endl;
                }
            	// freefemOutput << Th.nv << " ";
            	// freefemOutput << Ph.nv << " ";
            	// freefemOutput << adaptCounter << " ";
            	
            	// cout << endl;
            };
            
            // cout << endl << endl << "Building mesh took: " << buildTime; 
            // cout << endl << "First adapt took: " << firstAdaptTime; 
            // cout << endl << "First run took: " << firstRunTime; 
            // cout << endl << "Adaptation took: " << adaptTime; 
            // cout << endl << "Calculating coefficients took: " << clock()- coeffTime0;
            // cout << endl << "Total time: " << clock()-time0 << endl << endl;
            """
        )

    def __streamline_extension(self, beta, dr):
        """Calculate a vector over which the tip is shifted.

        Derived from the fact that the finger proceeds along a unique
        streamling going through the tip.

        Parameters
        ----------
        beta : float
            a1/a2 value
        dr : float
            A distance over which the tip is moving.

        Returns
        -------
        dR : array
            An 1-2 array.

        """
        if np.abs(beta) < 1000:
            y = ((beta**2) / 9) * ((27 * dr / (2 * beta**2) + 1) ** (2 / 3) - 1)
        else:
            y = dr - (9*dr**2)/(4*beta**2) + (27*dr**3) / \
                (2*beta**4) - (1701*dr**4)/(16*beta**6)
        x = np.around(
            np.sign(beta) * 2 * ((y**3 / beta**2) +
                                 (y / beta) ** 4) ** (1 / 2), 9)                                                      
        return np.array([x, y])

    def find_test_dRs(self, network, is_dr_normalized, is_zero_approx_step=False):
        """Find a single test shift over which the tip is moving.

        Parameters
        ----------
        network : object of class Network

        Returns
        -------
        dRs_test : array
            An n-2 array with dx and dy shifts for each tip.
        dt : float

        """
        max_a1 = np.max(self.flux_info[..., 0])
        if is_dr_normalized:
            # normalize dr, so that the fastest tip moves over ds
            dt = self.ds / max_a1 ** self.eta
        else:
            dt = self.ds
        
        # shallow copy of active_branches (creates new list instance, but the elements are still the same)
        branches_to_iterate = network.active_branches.copy()
        dRs_test = np.zeros((len(branches_to_iterate), 2))
        for i, branch in enumerate(branches_to_iterate):
            a1 = self.flux_info[i, 0]
            a2 = self.flux_info[i, 1]
            a3 = self.flux_info[i, 2]
            
            # check bifurcations and moving condition
            is_bifurcating = False
            is_moving = True
            if is_zero_approx_step:
                # bifurcation
                if (
                    self.bifurcation_type
                    and branch.length() > self.distance_from_bif_thresh
                ):
                    # the second condition above is used to avoid many bifurcations
                    # in almost one point which can occur while ds is very small
                    if (self.bifurcation_type == 1 and a1 > self.bifurcation_thresh) or (
                        self.bifurcation_type == 2 and a3 / a1 < self.bifurcation_thresh
                    ):
                        is_bifurcating = True
                    elif self.bifurcation_type == 3:
                        p = self.bifurcation_thresh * (a1 / max_a1) ** self.eta
                        r = np.random.uniform(0, 1)  # uniform distribution [0,1)
                        if p > r:
                            is_bifurcating = True
                # moving condition
                if a1/max_a1 < self.inflow_thresh or \
                    (a1/max_a1)**self.eta < self.inflow_thresh:
                    is_moving = False
                    network.sleeping_branches.append(branch)
                    network.active_branches.remove(branch)
                    print("! Branch {ID} is sleeping !".format(ID=branch.ID))         
            
            if is_moving:
                # __streamline_extension formula is derived in the coordinate
                # system where the tip segment lies on a negative Y axis;
                # hence, we rotate obtained dR vector to that system
                tip_angle = np.pi / 2 - branch.tip_angle()
                dr = dt * a1**self.eta
                beta = a1 / a2
                dRs_test[i] = np.dot(
                    rotation_matrix(
                        tip_angle), self.__streamline_extension(beta, dr)
                )
            else:
                dRs_test[i] = -10
            if is_bifurcating:
                print("! Branch {ID} bifurcated !".format(ID=branch.ID))
                dR = np.dot(
                    rotation_matrix(
                        tip_angle), self.__streamline_extension(beta, dr)
                )
                dRs_test[i] = -10
                dRs_test = np.vstack( (dRs_test, [
                    np.dot(
                        rotation_matrix(-self.bifurcation_angle / 2), dR),
                    np.dot(rotation_matrix(
                        self.bifurcation_angle / 2), dR) ]) )
                branch.dR = dRs_test[-2:]
            else:
                branch.dR = dRs_test[i]
            
        dRs_test = dRs_test[dRs_test[:,0]>-10]
        return dRs_test, dt
    
    def prepare_script_box(self, connections_bc, points, points_per_unit_len=0.5):
        """Return part of the FreeFEM script with the geometry of the box."""
        
        ns_border = np.max(( np.ones(connections_bc.shape[0]), \
                points_per_unit_len*np.linalg.norm(np.diff(points[connections_bc[:,:2]], axis=1)[:,0], axis=1) ),
               axis=0 ).astype(int)
            
        border_box = "\nreal buildTime0=clock();\n\n"
        border_box = (
            border_box
            + "int[int] boxN={ns};\n".format(ns=arr2str(ns_border))
            + "int[int] boxBC={bcs};\n".format(bcs=arr2str(connections_bc[:,2]))
            + "real[int,int] boxXY={points};\n".format(points=arr2str(points))
            + "border box(t=0, 1; i){{ int ii=(i+1)%{n}; x=boxXY(i,0)*(1-t) + boxXY(ii,0)*t; y=boxXY(i,1)*(1-t) + boxXY(ii,1)*t; label=boxBC(i);}}\n\n".format(
                n=len(points)
            )
        )
        # cout<<endl<<i<<"  "<<ii<<"  "<<x<<"  "<<y;
        inside_buildmesh_box = " box(boxN) +"
        return border_box, inside_buildmesh_box

    def prepare_script(self, network):
        """Return a FreeFEM script with `network` geometry."""

        tips = np.empty((len(network.active_branches), 4))
        border_network = ""
        inside_buildmesh = self.__script_inside_buildmesh_box
        for i, branch in enumerate(network.branches):
            border_network, inside_buildmesh = prepare_contour(border_network, inside_buildmesh, i, branch.points, label=branch.BC, border_name="branch")
            if branch in network.active_branches:
                ind = network.active_branches.index(branch)
                tips[ind, 0] = branch.BC # boundary condition
                tips[ind, 1] = branch.tip_angle() # angle with X axis
                tips[ind, 2] = branch.points[-1, 0] 
                tips[ind, 3] = branch.points[-1, 1]
        inside_buildmesh = inside_buildmesh[:-1]

        buildmesh = (
            textwrap.dedent(
                """
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // BUILDING MESH
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            """
            )
            + self.__script_border_box
            + border_network
            + "\nmesh Th = buildmesh({inside_buildmesh});\n".format( # , fixedborder=true
                inside_buildmesh=inside_buildmesh
            )
            + "\nreal buildTime=clock() - buildTime0;\n"
            + "// plot(Th, wait=true, fill=true, bb=[[1.25,0.25],[1.75,0.75]])\n;"
        )
                
        tip_information = textwrap.dedent(
            """
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // TIP INFORMATION
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            
            int nbTips={n_tips};
            int[int] BC(nbTips); BC={bc};
            real[int] angle(nbTips); angle={angle};
            real[int] X(nbTips); X={x};
            real[int] Y(nbTips); Y={y};\n
            """.format(
                n_tips=len(network.active_branches),
                bc=arr2str(tips[:,0]),
                angle=arr2str(tips[:,1]),
                x=arr2str(tips[:,2]),
                y=arr2str(tips[:,3]),
            )
        )

        script = self.__script_init + buildmesh + tip_information
        if self.equation:
            script = script + self.__script_problem_Poisson
        else:
            script = script + self.__script_problem_Laplace
        script = script + self.__script_adaptmesh + self.__script_integrate_a1a2a3

        return script
    
    def run_freefem_temp(self, script):
        """Run FreeFEM from temporary file and import the a1a2a3 coefficients."""
        temporary_files = []  # to close at the end
        with NamedTemporaryFile(suffix=".edp", mode="w", delete=False) as edp_temp_file:
            edp_temp_file.write(script)
            temporary_files.append(edp_temp_file)

        cmd = [
            "FreeFem++",
            "-nw",
            "-nc",
            "-v",
            "0",
            "-f",
            "{file_name}".format(file_name=edp_temp_file.name),
        ]
        result = subprocess.run(
            args=cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # close temporary files
        for tmp_file in temporary_files:
            tmp_file.close()
            os.unlink(tmp_file.name)
        
        return result

    def run_freefem(self, script):
        """Run FreeFEM and import the a1a2a3 coefficients. Useful for debugging."""

        with open("script.edp", "w") as edp_temp_file:
            edp_temp_file.write(script)

        cmd = [
            "FreeFem++",
            "-nw",
            "-nc",
            "-v",
            "0",
            "-f",
            "{file_name}".format(file_name=edp_temp_file.name),
        ]
        result = subprocess.run(
            args=cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        return result

    def solve_PDE(self, network):
        """Solve the PDE for the field around the network.

        Prepare a FreeFEM script, export it to a temporary file and run.
        Then, import the a1a2a3 coefficients to `self.flux_info`.

        Parameters
        ----------
        network : object of class Network
            Network around which the field will be calculated.

        Returns
        -------
        None.

        """
        script = self.prepare_script(network)
        if system() == "Windows":
            out_freefem = self.run_freefem(script) # useful for debugging
        else:
            out_freefem = self.run_freefem_temp(script)
        
        if out_freefem.returncode:
            print("\nFreeFem++ failed in the first try.\n")
            print("stdout:", out_freefem.stdout.decode())
            print("stderr:", out_freefem.stderr.decode())
            out_freefem = self.run_freefem_temp(script.replace(\
                            "nvAroundTips.min < 250", \
                            "nvAroundTips.min < 350")
                                                )
            if out_freefem.returncode:
                print("\nFreeFem++ worked with stronger mesh adaptation.\n")
            
        ai_coeffs_flat = np.fromstring(out_freefem.stdout[out_freefem.stdout.find(b"kopytko")+7:], sep=",")
        self.flux_info = ai_coeffs_flat.reshape(len(ai_coeffs_flat) // 3, 3)
        with np.printoptions(formatter={"float": "{:.6e}".format}):
            print("a1a2a3") # , self.flux_info)
            for i, branch in enumerate(network.active_branches):
                print("Branch {}:".format(branch.ID), self.flux_info[i], ", l={}".format(branch.length()))
            

class FreeFEM_ThickFingers:
    """PDE solver based on the finite element method implemented in FreeFEM [Ref2]_.

    Attributes
    ----------
    finger_width : float, default 0.02
        The width of the fingers.
    mobility_ratio : float, default 1e4
        Mobility ratio between inside and outside of the fingers.
        mobility_outside = 1, mobilty_inside = `mobility_ratio`
    equation : int, default 0
        - 0: Laplace
        - 1: Poisson
    eta : float, default 1.0
        The growth exponent (v=a1**eta).
        High values increase competition between the branches.
        Low values stabilize the growth.
    ds : float, default 0.01
        A distance over which the fastest branch in the network
        will move in each timestep.
    bifurcation_type : int, default 0
        - 0: no bifurcations
        - 1: velocity bifurcations (velocity criterion)
        - 2: random bifurcations
    bifurcation_thresh : float, default 0
        Threshold for the bifurcation criterion.
    bifurcation_angle : float, default 2pi/5
        Angle between the daughter branches after bifurcation.
        Default angle (72 degrees) corresponds to the analytical solution
        for fingers in a diffusive field.
    inflow_thresh : float, default 0.05
        Threshold to put asleep the tips with less than ``inflow_thresh``
        of max flux/velocity.
    distance_from_bif_thresh : float, default 2.1*``ds``
        A minimal distance the tip has to move from the previous bifurcations
        to split again.    
    flux_info : array
        A 2-n array with total flux and angle of highest flux direction 
        for each tip in the network.


    References
    ----------
    .. [Ref2] https://freefem.org/

    """

    def __init__(
            self, 
            network,
            finger_width=0.02, 
            mobility_ratio=1e4,
            equation=0,
            eta=1.0,
            ds=0.01,
            bifurcation_type=0,
            bifurcation_thresh=None,
            bifurcation_angle=2 * np.pi / 5,
            inflow_thresh=0.05,
            distance_from_bif_thresh=None,
        ):
        """Initialize FreeFEM.

        Parameters
        ----------
        network : Network
        finger_width : float, default 0.02
        mobility_ratio : float, default 1e4
        equation : int, default 0
        eta : float, default 1.0
        ds : float, default 0.01
        bifurcation_type : int, default 0
        bifurcation_thresh : float, default depends on bifurcation_type
        bifurcation_angle : float, default 2pi/5
        inflow_thresh : float, default 0.05
        distance_from_bif_thresh : float, default 2.1*``ds``

        Returns
        -------
        None.

        """
        self.finger_width = finger_width
        self.mobility_ratio = mobility_ratio 
        self.equation = equation
        self.eta = eta
        self.ds = ds
        self.flux_info = []

        self.bifurcation_type = bifurcation_type  # no bifurcations, a1, a3/a1, random
        self.bifurcation_thresh = bifurcation_thresh
        if bifurcation_thresh is None:
            self.bifurcation_thresh = 0
            if self.bifurcation_type == 1:
                self.bifurcation_thresh = 0.8  # velocity bifurcations
            if self.bifurcation_type == 2:  # random bifurcations
                self.bifurcation_thresh = 3 * ds
                # random bifurcations: bif_probability
        self.bifurcation_angle = bifurcation_angle  # 2*np.pi/5

        # less than `inflow_thresh` of max flux/velocity puts branches asleep
        self.inflow_thresh = inflow_thresh
        self.distance_from_bif_thresh = 2.1 * ds if distance_from_bif_thresh is None else distance_from_bif_thresh
         
        # parts of the script
        self.pbc = "" if network.box.boundary_conditions[0]==2 else ", periodic=PBC"
        
        self.__script_init = textwrap.dedent(
            """
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // INITIALISATION
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            verbosity = 0;
            
            real time0=clock();
            
            // Adaptation around the tip
            func real tipfield( real[int] X, real[int] Y, int nTips, real R)
            {
                real err=0.;
                for(int i=0;i<nTips;i++)
                {
                    real rr=((x-X(i))^2 + (y-Y(i))^2)^0.5;
                    if (rr>0.999*R & rr<1.001*R){
                        err+=1.;
                        }
                }
            	return err;
            }
            
            // Counting vertices on the tips
            func int[int] countNvOnTips( mesh th, int[int] tLabels, int nTips)
            {
                int[int] nvOnTips(nTips);
                for(int i=0;i<nTips;++i)
                {
                    int ndof=0;
                    int1d(th, tLabels(i), qfe=qf1pE)( (ndof++)*1.);
                    nvOnTips(i) = ndof;
                };
                return nvOnTips;
            }
            """
        )
        
        self.script_border_box, self.script_inside_buildmesh_box = \
            self.prepare_script_box(network)
        
        self.__script_mobility = textwrap.dedent(
            """
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // DEFINING MOBILITY
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            """
            )
        
        self.__script_problem_Laplace = textwrap.dedent(
            """
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // DEFINING PROBLEM AND equation TO SOLVE
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            
            fespace Vh(Th,P2{pbc});
            Vh u,v,dxu,dyu,du;
            
            problem potential(u,v,solver=sparsesolver)=
                     int2d(Th)(mobility*(dx(u)*dx(v) + dy(u)*dy(v)))
                                // -int1d(Th,3)(v)  // constant flux
                                +on(3,u=0) // constant field
                                // -int2d(Th)(v) // rain in domain
                                +on(1,u=1);
            """
        ).format(pbc=self.pbc)

        self.__script_problem_Poisson = textwrap.dedent(
            """ 
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // DEFINING PROBLEM AND equation TO SOLVE
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            fespace Vh(Th,P2{pbc});
            Vh u,v,dxu,dyu,du;

            problem potential(u,v,solver=sparsesolver)=
                     int2d(Th)(mobility*(dx(u)*dx(v) + dy(u)*dy(v)))
                                -int2d(Th)(v) // Poissonian source term
                                // -int1d(Th,3)(v)  // constant flux
                                +on(3,u=0) // constant field
                                // -int2d(Th)(v) // rain in domain
                                +on(1,u=1);            
            """
        ).format(pbc=self.pbc)

        self.__script_adaptmesh = textwrap.dedent(
            """
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // ADAPTING THE MESH AND SOLVING FOR THE FIELD
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////        

            // int iTip=0;
            // real x0Th=X(iTip)-2*tipR, y0Th=Y(iTip)-2*tipR, x1Th=X(iTip)+2*tipR, y1Th=Y(iTip)+2*tipR;
            // plot(Th, wait=true, bb=[[x0Th, y0Th],[x1Th, y1Th]]);
            
            // Solving the problem for the first time
            real firstRunTime0=clock();
            potential;
            // cout<<"First solve completed."<<endl;
            real firstRunTime=clock() - firstRunTime0;
            
            // Adaptation loop
            real adaptTime0=clock();
            // cout << endl << endl << "Adaptation..." << endl;
            real error=0.01;
            int adaptCounter=0;
            int[int] nvOnTips = countNvOnTips(Th, tipLabels, nbTips);
            // plot(Th, wait=true);
            while(nvOnTips.min < 70 || adaptCounter<1)
            {{
                // cout << "Adaptation step: " << adaptCounter;
                // cout << ", nvOnTip.min = " << nvOnTips.min << endl;
                // plot(Th, wait=true, bb=[[x0Th, y0Th],[x1Th, y1Th]]);
                Th=adaptmesh(Th, [u, 0.005*tipfield(X,Y,nbTips,tipR)],err=error,nbvx=500000,verbosity=0,nbsmooth=100,iso=1,ratio=1.8,keepbackvertices=1{pbc}); // Adapting mesh according to the first solution
                u=u;
                mobility=mobility;
                error=error/2;
                potential; // Solving one more time with adapted mesh
                nvOnTips = countNvOnTips(Th, tipLabels, nbTips);
                adaptCounter++;
                // plot(Th, wait=true);
            }}
            cout << "Adaptation step: " << adaptCounter;
            cout << ", nvOnTip.min = " << nvOnTips.min << endl;
            // plot(Th, wait=true, bb=[[x0Th, y0Th],[x1Th, y1Th]]);
            // cout << "Problem solved." << endl;
            // plot(u, wait=true, fill=true);
            
            real adaptTime=clock() - adaptTime0;
            """
        ).format(pbc=self.pbc)

        self.__script_tip_integration = textwrap.dedent(
            """
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // CALCULATE FLUXES AND EXPORT RESULTS
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            
            // Calculating gradient
            dxu=dx(u);
            dyu=dy(u);
            // du=(dxu^2+dyu^2)^0.5;
            // plot(du, wait=true, fill=true);
            
            // Deteremining the flux coming to the tip
            // More on reading the field values in specific points:
            // https://www.ljll.math.upmc.fr/pipermail/freefempp/2013-July/002798.html
            // https://ljll.math.upmc.fr/pipermail/freefempp/2009/000337.html
            int avgWindow = 5;
            cout.precision(12);
            cout << "kopytko ";
            for(int k=0;k<nbTips;k++)
            {
                int ndof=nvOnTips(k), n=0;
                real[int] angles(ndof), fluxes(ndof); // angles with X axis
                int1d(Th, tipLabels(k), qfe=qf1pE)( (angles(n++)=atan2(y-Y(k), x-X(k)))*1.
                                                +(fluxes(n)=abs(dxu*N.x+dyu*N.y))*1.);
                // cout<<"tip"<<tipLabels(k)<<endl;
                cout<<"angles"<<tipLabels(k)<<angles<<"angles"<<tipLabels(k)<<"end"<<endl;
                cout<<"fluxes"<<tipLabels(k)<<fluxes<<"fluxes"<<tipLabels(k)<<"end"<<endl;
                real totGrad =  int1d(Th, tipLabels(k))( abs([dxu,dyu]'*[N.x,N.y]) );
            	 cout<<"tot_flux"<<tipLabels(k)<<"1 "<<totGrad<<"tot_flux"<<tipLabels(k)<<"end"<<endl;
                
                // real maxGrad=0, maxAngle=pi/2;
                // real[int] fluxesMvAvg(ndof-avgWindow+1), anglesMvAvg(ndof-avgWindow+1);
                // for (int i=0; i<=(ndof-avgWindow); i++){
                    // real sumGrad=0, sumAng=0;
                    // for (int j=i; j<i+avgWindow; j++){
                        // sumGrad += fluxes[j];
                        // sumAng += angles[j];
                    // }
                    // fluxesMvAvg(i) = sumGrad / avgWindow;
                    // anglesMvAvg(i) = sumAng / avgWindow;
                    // if (fluxesMvAvg(i)>maxGrad){
                        // maxGrad=fluxesMvAvg(i);
                        // maxAngle=anglesMvAvg(i);
                    // }
                // }
                // real totGrad =  int1d(Th, tipLabels(k))( abs([dxu,dyu]'*[N.x,N.y]) );
                // cout << totGrad << "," << maxAngle << ",";
            }            
            """
        )

    def __check_bifurcation_and_moving_conditions(self, network):
        """Check bifurcation and moving conditions."""

        a1 = self.pde_solver.flux_info[..., 0]
        max_a1 = np.max(a1)
        # checking which branches are_moving
        # (first condition for low eta, second for high)
        are_moving = np.logical_and(a1/max_a1 > self.inflow_thresh,
                                    (a1/max_a1)**self.eta > self.inflow_thresh)

        # shallow copy of active_branches (creates new list instance, but the elements are still the same)
        branches_to_iterate = network.active_branches.copy()
        for i, branch in enumerate(branches_to_iterate):
            a1 = self.pde_solver.flux_info[i, 0]
            if (
                self.bifurcation_type
                and branch.length() > self.distance_from_bif_thresh
            ):
                # the second condition above is used to avoid many bifurcations
                # in almost one point which can occur while ds is very small
                if (self.bifurcation_type == 1 and a1 > self.bifurcation_thresh):
                    branch.is_bifurcating = True
                elif self.bifurcation_type == 3:
                    p = self.bifurcation_thresh * (a1 / max_a1) ** self.eta
                    r = np.random.uniform(0, 1)  # uniform distribution [0,1)
                    if p > r:
                        branch.is_bifurcating = True
                        
            if not are_moving[i]:
                network.sleeping_branches.append(branch)
                network.active_branches.remove(branch)
                print("! Branch {ID} is sleeping !".format(ID=branch.ID))

        return are_moving
                       
    def find_test_dRs(self, network, is_dr_normalized, is_zero_approx_step=False):
        """Find a single test shift over which the tip is moving.

        Parameters
        ----------
        network : object of class Network

        Returns
        -------
        dRs_test : array
            An n-2 array with dx and dy shifts for each tip.
        dt : float
        
        """

        if is_dr_normalized:
            # normalize dr, so that the fastest tip moves over ds
            dt = self.ds / np.max(self.flux_info[..., 0] ** self.eta)
        else:
            dt = self.ds
        dRs_test = np.empty((len(network.active_branches), 2))
        for i, branch in enumerate(network.active_branches):
            a1 = self.flux_info[i, 0]
            angle = self.flux_info[i, 1] # angle between X axis and the highest gradient direction
            dr = dt * a1**self.eta
            dR = [dr, 0]
            # rotation_matrix rotates in counter-clockwise direction, hence the minus
            dRs_test[i] = np.dot(rotation_matrix(-angle), dR)
            branch.dR = dRs_test[i]
            
        return dRs_test, dt             

    def prepare_script_box(self, network):
        """Return parts of the FreeFEM script with the geometry of the ``network.box``."""
        pt_between = lambda pt1, pt2 : pt1 + (pt2-pt1)/np.linalg.norm(pt2-pt1)*self.finger_width/2
        
        points = network.box.points.copy()
        boundary_conditions = network.box.boundary_conditions.copy()
        for i, seed_ind in enumerate(network.box.seeds_connectivity[:,0]):
            seed_pt = network.box.points[seed_ind]
            left_pt = network.box.points[seed_ind-1]
            right_pt = network.box.points[(seed_ind+1) % len(network.box.points)]
            v1 = pt_between(seed_pt, left_pt)
            v2 = pt_between(seed_pt, right_pt)
            
            points[seed_ind+i] = v1
            points = np.insert(points, seed_ind+i+1, v2, axis=0)
            boundary_conditions = np.insert(boundary_conditions, seed_ind+i+1, DIRICHLET)
        
        br_conn = network.branch_connectivity
        ind_bt = np.where(br_conn[:,1]==-1)[0]
        if ind_bt.size:
            breakthrough_pt = network.branches[br_conn[ind_bt[0],0]].points[-1]
            v1 = pt_between(breakthrough_pt, network.box.points[1])
            v2 = pt_between(breakthrough_pt, network.box.points[2])
            points = np.insert(points, 2, [v1, v2], axis=0)
            boundary_conditions = np.insert(boundary_conditions, 1, 2*[CONSTANT_FLUX])
    
        connections = np.vstack(
                        [np.arange(len(points)), np.roll(
                            np.arange(len(points)), -1)]
                    ).T
        connections_bc = np.column_stack((connections, boundary_conditions))
        
        border_box, inside_buildmesh_box = \
            FreeFEM.prepare_script_box(self, connections_bc, points, points_per_unit_len=0.5) # 10
        
        return border_box, inside_buildmesh_box

    def finger_contour(self, branch, network, 
                       script_mobility="", mobility_regions="", 
                       border_contour="", inside_buildmesh=""):
        """Return contour of the ``branch`` and parts of the scripts associated with it."""
        pt_between = lambda pt1, pt2 : pt1 + (pt2-pt1)/np.linalg.norm(pt2-pt1)*self.finger_width/2
        
        if branch.ID in network.box.seeds_connectivity[:,1]:
            ind_seed = network.box.seeds_connectivity[network.box.seeds_connectivity[:,1]==branch.ID, 0]
            # box (boundary)
            box_pt_right = pt_between(network.box.points[ind_seed], \
                                  network.box.points[(ind_seed+1) % len(network.box.points)])
            box_pt_left = pt_between(network.box.points[ind_seed], \
                                  network.box.points[ind_seed-1])
                
            script_mobility = script_mobility + \
                textwrap.dedent("""\nint indRegion{i} = Th({x}, {y}).region;""".format(
                    i=branch.ID, x=branch.points[-1][0], y=branch.points[-1][1] ))
            mobility_regions = mobility_regions + "region==indRegion{i} || ".format(i=branch.ID)
                
        # contour can intersect itself if finger_width>ds (distances between the points in the branch)
        # hence, we select the points on the branch that are finger_width/2 apart from each other
        skeleton = [branch.points[0]]
        segment_lengths = np.linalg.norm(branch.points[1:]-branch.points[:-1], axis=1)
        len_sum = 0
        for i, seg in enumerate(segment_lengths):
            len_sum = len_sum + seg
            if len_sum>self.finger_width/4:
                len_sum = 0
                skeleton.append(branch.points[i+1])
        if len(skeleton)>1:
            skeleton[-1] = branch.points[-1]
        else:
            skeleton.append(branch.points[-1])
        skeleton = np.array(skeleton)
        skeleton_shifts_up = (skeleton[1:]-skeleton[:-1]) # vectors between points: 0 -> 1, 1 -> 2, etc.
        skeleton_shifts_down = (skeleton[:-1]-skeleton[1:]) # vectors between points: 1 -> 0, 2 -> 1, etc.
    
        # angles between X axis and segments going up from points 0, 1, ..., n-1
        angles_up = np.arctan2(skeleton_shifts_up[:,1], skeleton_shifts_up[:,0]) 
        # angles between X axis and segments going down from points 1, 2, ..., n
        angles_down = np.arctan2(skeleton_shifts_down[:,1], skeleton_shifts_down[:,0])
        # note: the lists with angles are shifted (angles at point 1 are: angles_down[0] and angles_up[1]
        full_angles_right = (angles_down[:-1] + angles_up[1:])/2
        full_angles_left = full_angles_right + np.pi
    
        contour_right = skeleton[1:-1] + self.finger_width/2 * \
                        np.stack((np.cos(full_angles_right), np.sin(full_angles_right))).T
        contour_right = np.vstack(( box_pt_right, contour_right ))
        
        contour_left = skeleton[1:-1] + self.finger_width/2 * \
                        np.stack((np.cos(full_angles_left), np.sin(full_angles_left))).T
        # order from the end to the start:
        contour_left = np.vstack(( np.flip(contour_left, axis=0), box_pt_left ))
        
        contour_tip = np.empty((0,2))
        if len(network.branch_connectivity)==0 or \
            branch.ID not in network.branch_connectivity[:,0]:
            # if the branch is a dead end -> tip with a semi-circular cap
            angle_tip_right = angles_down[-1] + np.pi/2
            angle_tip_left = angle_tip_right + np.pi
            tip_right = skeleton[-1] + self.finger_width/2 * \
                            np.array([np.cos(angle_tip_right), np.sin(angle_tip_right)])
            tip_left = skeleton[-1] + self.finger_width/2 * \
                    np.array([np.cos(angle_tip_left), np.sin(angle_tip_left)])
            
            contour_right = np.vstack(( contour_right, tip_right ))
            contour_left = np.vstack(( tip_left, contour_left ))    
           
            # semi-circular cap v2
            tilt_shift = tip_right-skeleton[-1]
            tilt_ang = np.arctan2(tilt_shift[1], tilt_shift[0])
            
            angles = np.linspace(0,np.pi,50)+tilt_ang
            contour_tip = np.stack((skeleton[-1,0]+self.finger_width/2*np.cos(angles),
                                    skeleton[-1,1]+self.finger_width/2*np.sin(angles))).T
            
            border_contour = (
                border_contour
                + "border tip{i}(t=0, 1){{x={x0:.6e}+{f_w_half:.6e}*cos(t*pi+{phi0:.6e});y={y0:.6e}+{f_w_half:.6e}*sin(t*pi+{phi0:.6e}); label={bc};}}\n".format(
                    i=branch.ID, x0=skeleton[-1,0], y0=skeleton[-1,1], 
                    f_w_half=self.finger_width/2, bc=1000+branch.ID, phi0=tilt_ang
                )
            )
            inside_buildmesh = (
                inside_buildmesh + " tip{i}(101) +".format(i=branch.ID)
            )
            
        if (network.branch_connectivity[network.branch_connectivity[:,1]==-1,0]==branch.ID).any():
            breakthrough_pt = branch.points[-1]
            v1 = pt_between(breakthrough_pt, network.box.points[1])
            v2 = pt_between(breakthrough_pt, network.box.points[2])
            contour_right = np.vstack(( contour_right, v1 ))
            contour_left = np.vstack(( v2, contour_left ))  

        # print(contour_right)
        # print(contour_left)
        # import matplotlib.pyplot as plt
        # plt.plot(*contour_right.T, ".-")
        # plt.plot(*contour_left.T, ".-")
        # points_to_plot = network.box.points[network.box.connections]
        # for pts in points_to_plot:
        #     plt.plot(*pts.T, color="0")
        
        return contour_right, contour_tip, contour_left, \
            script_mobility, mobility_regions, border_contour, inside_buildmesh
        
    
    def prepare_script0(self, network):
        """Return a full FreeFEM script with the `network` geometry."""

        tip_information = textwrap.dedent(
            """
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // TIP INFORMATION
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            
            real tipR={f_w_half};
            int nbTips={n_tips};
            int[int] tipLabels(nbTips);
            int[int] indRegions(nbTips);
            real[int] X(nbTips); 
            real[int] Y(nbTips);
            """.format(
                f_w_half = self.finger_width/2,
                n_tips=len(network.active_branches)
            )
        )
        for i, branch in enumerate(network.active_branches):
            tip_information = (
                tip_information
                + "\ntipLabels({j})={bc};".format(j=i, bc=1000+branch.ID)
                + "\nX({j})={x:.6e};".format(j=i, x=branch.points[-1, 0])
                + "\nY({j})={y:.6e};".format(j=i, y=branch.points[-1, 1])
                )
        tip_information = tip_information + "\n"               
        
        mobility_regions = ""
        script_mobility = self.__script_mobility                                                  
        border_contour = ""
        inside_buildmesh = self.script_inside_buildmesh_box
        for i, branch in enumerate(network.branches):
            contour_right, _, contour_left, \
                script_mobility, mobility_regions, \
                    border_contour, inside_buildmesh = \
                        self.finger_contour(branch, network, \
                                            script_mobility, mobility_regions, \
                                                border_contour, inside_buildmesh)
            
            for k, points in enumerate([contour_right, contour_left]):
                # building mesh for contours on the right and left of the finger
                for j, pair in enumerate(zip(points, points[1:])):
                    border_contour = (
                        border_contour
                        + "border branch{i}connection{j}side{k}(t=0, 1){{x={x0:.6e}+t*({ax:.6e});y={y0:.6e}+t*({ay:.6e}); label=888;}}\n".format(
                            i=branch.ID, j=j, k=k, 
                            x0=pair[0][0], ax=pair[1][0] - pair[0][0], 
                            y0=pair[0][1], ay=pair[1][1] - pair[0][1]
                        )
                    )
    
                    inside_buildmesh = (
                        inside_buildmesh + " branch{i}connection{j}side{k}(1) +".format(i=branch.ID, j=j, k=k)
                    )
                    
        inside_buildmesh = inside_buildmesh[:-1]
        script_mobility = script_mobility + textwrap.dedent(
            """
            fespace Vh0(Th, P0{pbc});
            Vh0 mobility = {mobilityOutside}*!({mobility_regions}) + {mobilityInside}*({mobility_regions});
            """.format(mobilityOutside=1, mobilityInside=self.mobility_ratio, 
                        mobility_regions = mobility_regions[:-4], pbc=self.pbc)
            )

        buildmesh = (
            textwrap.dedent(
                """
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // BUILDING MESH
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            """
            )
            + self.script_border_box
            + border_contour
            )
        if network.box.boundary_conditions[0]!=2:
            buildmesh = buildmesh + \
                "func PBC=[[{left},y],[{right},y]];".format( left=LEFT_WALL_PBC, right=RIGHT_WALL_PBC)
        buildmesh = buildmesh + \
            "\nmesh Th = buildmesh({inside_buildmesh}, fixedborder=true);\n".format(
                inside_buildmesh=inside_buildmesh
            ) + \
            "\nreal buildTime=clock() - buildTime0;\n// plot(Th, wait=true);\n"

        script = self.__script_init + buildmesh + tip_information + script_mobility
        if self.equation:
            script = script + self.__script_problem_Poisson
        else:
            script = script + self.__script_problem_Laplace
        script = script + self.__script_adaptmesh + self.__script_tip_integration

        return script
    
    
    def prepare_script(self, network):
        """Return a full FreeFEM script with the `network` geometry."""
        
        tips = np.empty((len(network.active_branches), 3))
        pts = [] # list with regularized points (skeleton)
        points_in = [] # points inside subdomains with higher mobility
        border_contour = ""
        inside_buildmesh = self.script_inside_buildmesh_box
        for i, branch in enumerate(network.branches):
            # contour can intersect itself if finger_width>ds (distances between the points in the branch)
            # hence, we select the points on the branch that are finger_width/2 apart from each other
            skeleton = [branch.points[0], branch.points[0]+[0,branch.points[1,1]/2]]
            # we add the 2nd point in skeleton to ensure that the bottom cap is compatible with box
            segment_lengths = np.linalg.norm(branch.points[2:]-branch.points[1:-1], axis=1)
            len_sum = 0
            for i, seg in enumerate(segment_lengths):
                len_sum = len_sum + seg
                if len_sum>self.finger_width/4:
                    len_sum = 0
                    skeleton.append(branch.points[i+2])
            if len(skeleton)==2:
                skeleton.append(branch.points[-1])
            else:
                tip_vec = branch.points[-1] - branch.points[-2]
                # if branch.ID in network.branch_connectivity[:,0]:
                skeleton.insert(-1, branch.points[-1] - 0.5*tip_vec)
                skeleton[-1] = branch.points[-1]
            skeleton = np.array(skeleton)
            pts.append(skeleton)
            
            # if the branch is a dead end -> tip with a semi-circular cap
            if len(network.branch_connectivity)==0 or \
                    branch.ID not in network.branch_connectivity[:,0]:
                tip_vec =  skeleton[-1]-skeleton[-2]
                points_in.append(skeleton[-1] + tip_vec/np.linalg.norm(tip_vec)*self.finger_width/4)
                tilt_ang = np.arctan2(tip_vec[1], tip_vec[0])-np.pi/2
                border_contour = (
                    border_contour
                    + "border tip{i}(t=0, 1){{x={x0:.6e}+{f_w_half:.6e}*cos(t*pi+{phi0:.6e});y={y0:.6e}+{f_w_half:.6e}*sin(t*pi+{phi0:.6e}); label={bc};}}\n".format(
                        i=branch.ID, x0=branch.points[-1,0], y0=branch.points[-1,1], 
                        f_w_half=self.finger_width/2, bc=1000+branch.ID, phi0=tilt_ang
                    )
                )
                inside_buildmesh = (
                    inside_buildmesh + " tip{i}(101) +".format(i=branch.ID)
                )  
                
            if branch in network.active_branches:
                ind = network.active_branches.index(branch)
                tips[ind, 0] = 1000+branch.ID # tip label
                tips[ind, 1] = branch.points[-1, 0] # x
                tips[ind, 2] = branch.points[-1, 1] # y
        border_contour = border_contour + "\n"
        
        # thicken tree and find intersection with the box
        tree = MultiLineString(pts)
        thick_tree = tree.buffer(distance=self.finger_width/2, cap_style=2, join_style=1, resolution=99)
        box = LinearRing(network.box.points)
        box = linemerge(box.union(box.intersection(thick_tree)))
        
        # polygons to contours
        polygons = [thick_tree] if thick_tree.geom_type=="Polygon" else thick_tree.geoms
        contours = []
        for i, poly in enumerate(polygons):
            points_in.append(poly.representative_point().coords[0])
            poly = shapely.geometry.polygon.orient(poly) # now, exterior is ccw, but interiors are cw
            
            # exteriors
            poly_exterior = linemerge(poly.exterior.difference(box))
            lines = [poly_exterior] if poly_exterior.geom_type=="LineString" else poly_exterior.geoms
            for line in lines:
                contours.append( np.array(line.coords) )
            
            # interiors
            for ring in poly.interiors:
                contours.append( np.asarray(ring.coords) )     
                                            
        # contours to border
        for i, points in enumerate(contours):
            # points = np.flip(points0, axis=0)
            border_contour, inside_buildmesh = prepare_contour(border_contour, inside_buildmesh, i, points, label=888, border_name="contour" )
        inside_buildmesh = inside_buildmesh[:-1]
        buildmesh = (
            textwrap.dedent(
                """
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // BUILDING MESH
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            """
            )
            + self.script_border_box
            + border_contour
            )
        if network.box.boundary_conditions[0]!=2:
            buildmesh = buildmesh + \
                "func PBC=[[{left},y],[{right},y]];".format( left=LEFT_WALL_PBC, right=RIGHT_WALL_PBC)
        buildmesh = buildmesh + \
            "\n// plot({inside_buildmesh},dim=2);\nmesh Th = buildmesh({inside_buildmesh}, fixedborder=true);\n".format(
                inside_buildmesh=inside_buildmesh
            ) + \
            "\nreal buildTime=clock() - buildTime0;\n// plot(Th, wait=true);\n"

        # tips 
        tip_information = textwrap.dedent(
            """
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // TIP INFORMATION
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            
            real tipR={f_w_half};
            int nbTips={n_tips};
            int[int] tipLabels={tip_labels};
            real[int] X={x};
            real[int] Y={y};\n
            """.format(
                f_w_half=self.finger_width/2,
                n_tips=len(network.active_branches),
                tip_labels=arr2str(tips[:,0]),
                x=arr2str(tips[:,1]),
                y=arr2str(tips[:,2]),
            )
        )             
        
        # mobility 
        mobility_regions = ""
        script_mobility = self.__script_mobility
        for i, p in enumerate(points_in):
            script_mobility = script_mobility + \
                textwrap.dedent("""\nint indRegion{i} = Th({x}, {y}).region;""".format(
                    i=i, x=p[0], y=p[1] ))
            mobility_regions = mobility_regions + "region==indRegion{i} || ".format(i=i)
        script_mobility = script_mobility + textwrap.dedent(
            """
            fespace Vh0(Th, P0{pbc});
            Vh0 mobility = {mobilityOutside}*!({mobility_regions}) + {mobilityInside}*({mobility_regions});
            """.format(mobilityOutside=1, mobilityInside=self.mobility_ratio, 
                        mobility_regions = mobility_regions[:-4], pbc=self.pbc)
            )
        
        # whole script
        script = self.__script_init + buildmesh + tip_information + script_mobility
        if self.equation:
            script = script + self.__script_problem_Poisson
        else:
            script = script + self.__script_problem_Laplace
        script = script + self.__script_adaptmesh + self.__script_tip_integration

        return script
    
    
    def solve_PDE(self, network):
        """Solve the PDE for the field around the network.

        Prepare a FreeFEM script, export it to a temporary file and run.
        Then, import the results to `self.flux_info`.

        Parameters
        ----------
        network : object of class Network
            Network around which the field will be calculated.

        Returns
        -------
        None.

        """
        script = self.prepare_script(network)
        if system() == "Windows":
            out_freefem = FreeFEM.run_freefem(self, script) # useful for debugging
            print(out_freefem.stdout)
        else:
            out_freefem = FreeFEM.run_freefem_temp(self, script)
            # print(out_freefem.stdout)
            
        # flux_info calculated in FreeFem:
        # flux_info = np.fromstring(out_freefem.stdout[out_freefem.stdout.find(b"kopytko")+7:], sep=",")
        # self.flux_info = flux_info.reshape(len(flux_info) // 2, 2)
        
        # determining flux_info here:
        import_array = lambda key : np.fromstring(out_freefem.stdout[\
                                            out_freefem.stdout.find(key.encode("ascii"))+len(key):\
                                            out_freefem.stdout.find(key.encode("ascii")+b"end")], \
                                                  sep="\t")[1:]
 
        angles=[]; fluxes=[]; 
        self.flux_info = np.zeros((len(network.active_branches), 2))
        for i, branch in enumerate(network.active_branches):
            tip_label = 1000+branch.ID
            angles.append(import_array("angles{}".format(tip_label)))
            fluxes.append(import_array("fluxes{}".format(tip_label))) 
            ind_cut = np.where(np.diff(angles[-1])<0)[0]
            if ind_cut.size:
                angles[-1][ind_cut[0]+1:] = angles[-1][ind_cut[0]+1:] + 2*np.pi
            order = np.argsort(angles[-1])
            angles[-1] = angles[-1][order]
            fluxes[-1] = fluxes[-1][order]
            
            # gaussian convolution
            # (https://stackoverflow.com/questions/22291567/smooth-data-and-find-maximum)
            f = scipy.interpolate.interp1d(angles[-1], fluxes[i])
            xx = np.linspace(angles[-1][0], angles[-1][-1], 1000)
            yy = f(xx)
            window = scipy.signal.windows.gaussian(100, 1000)
            smoothed = scipy.signal.convolve(yy, window/window.sum(), \
                                             mode="same")
            
            # Total flux:
            self.flux_info[i,0] = import_array("tot_flux{}".format(tip_label))
            # Highest gradiend direction (angle with X axis)
            self.flux_info[i,1] = xx[np.argmax(smoothed)]
            
        with np.printoptions(formatter={"float": "{:.6g}".format}):
            # print("flux_info: \n", self.flux_info)
            print("angles: \n", self.flux_info[...,1]*180/np.pi)
