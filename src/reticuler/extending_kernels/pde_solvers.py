"""Solvers for PDEs describing the field around the network.

Classes:
    FreeFEM
    
"""

import numpy as np
import scipy
import subprocess
from datetime import datetime
import os.path
import os
from tempfile import NamedTemporaryFile
import textwrap

import shapely
from shapely.ops import linemerge 
from shapely.geometry import MultiLineString, LinearRing, Polygon

from reticuler.utilities.misc import LEFT_WALL_PBC, RIGHT_WALL_PBC, DIRICHLET_1, DIRICHLET_0, NEUMANN_1, DIRICHLET_GLOB_FLUX
from reticuler.utilities.misc import rotation_matrix

def asvoid(arr):
    """
    Based on https://stackoverflow.com/questions/16216078/test-for-membership-in-a-2d-numpy-array
    and http://stackoverflow.com/a/16973510/190597 (Jaime, 2013-06)
    View the array as dtype np.void (bytes). The items along the last axis are
    viewed as one value. This allows comparisons to be performed on the entire row.
    """
    arr = np.ascontiguousarray(arr)
    if np.issubdtype(arr.dtype, np.floating):
        """ Care needs to be taken here since
        np.array([-0.]).view(np.void) != np.array([0.]).view(np.void)
        Adding 0. converts -0. to 0.
        """
        arr += 0.
    return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[-1])))
def inNd(a, b, assume_unique=False):
    a = asvoid(a)
    b = asvoid(b)
    return np.in1d(a, b, assume_unique)

def arr2str(arr):
    return np.array2string(arr, separator=",", precision=12,max_line_width=np.inf,threshold=np.inf).replace("\n", "")

def array_from_string(out_string, key):
    ind_0 = out_string.find(key.encode("ascii"))+len(key)
    ind_1 = out_string.find(key.encode("ascii")+b"end")
    return np.fromstring(out_string[ind_0:ind_1], sep="\t")[1:]

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
    
def prepare_contour_list(border_contour, inside_buildmesh, i, points, label, ns_border=1, border_name="contour", i_tsh=1023):
    if not np.isscalar(ns_border):
        ns_border = arr2str(ns_border)
    border_contour = (
        border_contour
        + "real[int] {b_n}{i}X({n}); real[int] {b_n}{i}Y({n}); int[int] {b_n}{i}N({n}-1); {b_n}{i}N={N};".format(b_n=border_name, i=i, n=len(points), N=ns_border)
        )
    if not np.isscalar(label):
        border_contour = (
        border_contour
        + "int[int] {b_n}{i}BC({n});".format(b_n=border_name, i=i, n=len(points))
        )
    for j in range((len(points)-1)//i_tsh+1):
        border_contour = (
            border_contour
            + "\n{b_n}{i}X({ind0}:{ind1})={pointsX};\n{b_n}{i}Y({ind0}:{ind1})={pointsY};\n".format( \
                        b_n=border_name, i=i, ind0=j*i_tsh, ind1=(j+1)*i_tsh, \
                        pointsX=arr2str(points[j*i_tsh:(j+1)*i_tsh,0]),
                        pointsY=arr2str(points[j*i_tsh:(j+1)*i_tsh,1])
                            ) 
            )
        if not np.isscalar(label):
            border_contour = (
            border_contour
            + "{b_n}{i}BC({ind0}:{ind1})={bcs};\n".format(b_n=border_name, i=i, ind0=j*i_tsh, ind1=(j+1)*i_tsh, bcs=arr2str(label))
            )     

    if not np.isscalar(label):
        label = "{b_n}{i}BC(i)".format(b_n=border_name, i=i)
    border_contour = (
        border_contour
        + "border {b_n}{i}(t=0, 1; i){{ x = {b_n}{i}X(i)*(1-t) + {b_n}{i}X(i+1)*t; y = {b_n}{i}Y(i)*(1-t) + {b_n}{i}Y(i+1)*t; label={label};}}\n\n".format(
            b_n=border_name, i=i, label=label
        )
    )

    inside_buildmesh = (
        inside_buildmesh + " {b_n}{i}({b_n}{i}N) +".format(b_n=border_name, i=i)
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
        Default: 0.8 for a1 bifurcation; -0.1 for a3/a1
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
            is_script_saved=False,
            is_backward=False,
            is_leaf=False,
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
        self.is_script_saved = is_script_saved
        
        self.is_backward = is_backward
        self.is_leaf = is_leaf
        
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
            // DEFINING PROBLEM AND EQUATION TO SOLVE
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            
            fespace Vh(Th,P2);
            Vh u,v;
            
            real dirichletOut = 1; // also under a_i integrals
            problem potential(u,v,solver=sparsesolver)=
                     int2d(Th)(dx(u)*dx(v) + dy(u)*dy(v))
                         // -int2d(Th)(v) // rain in domain
                         -int1d(Th,{NEUMANN_1})(v)  // constant flux
                         +on({DIRICHLET_1},u=dirichletOut) // constant field
                         +on({DIRICHLET_0},u=0);
            """
        ).format(NEUMANN_1=NEUMANN_1, DIRICHLET_0=DIRICHLET_0, DIRICHLET_1=DIRICHLET_1)

        self.__script_problem_Poisson = self.__script_problem_Laplace.replace(
                                            "// -int2d(Th)(v) // rain in domain", 
                                            "-int2d(Th)(v) // rain in domain")

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
            Th = adaptmesh(Th,1,nbvx=500000,hmax=0.1,nbsmooth=100,iso=true,ratio=1.8,keepbackvertices=1);
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
            fespace Vh0(Th,P0); Vh0 h=1;
            real error=0.02;
            int adaptCounter=1;
            while(nvAroundTips.min < 250 || adaptCounter<=3)
            {
            	// cout << "Adaptation step: " << adaptCounter << ", h[].min = " << h[].min;
            	// cout << ", nvAroundTip.min = " << nvAroundTips.min << endl;
            	potential;
            	Th=adaptmesh(Th,[u/u[].max, 20.*tipfield(X,Y,3.*R,nbTips)],err=error,nbvx=1000000,iso=true,ratio=2,hmin=1e-5,keepbackvertices=1);
            	error = 0.5*error;
            	u=u;
            	h=hTriangle; // the triangle size
            	nvAroundTips = countNvAroundTips (3.*R, Th, Th.nv, nbTips, X, Y);
            	adaptCounter++;
                // plot(Th, wait=true);
            }
            
            // cout << endl << "Adaptation finished." << " h[].min = " << h[].min;
            // cout << ", nvAroundTip.min = " << nvAroundTips.min << endl;
            
            // solving with adapted mesh
            potential;
            // cout << "Problem solved." << endl;
            // plot(Th, wait=true);
            // plot(u, wait=true, fill=true, value=true);
            
            real adaptTime=clock() - adaptTime0;
            """
        )

        # // ofstream freefemOutput("{file_name}");
        self.__script_integrate = textwrap.dedent(
            """
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // INTEGRATING THE FIELD TO GET a_i COEFFICIENTS
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////        
            
            real coeffTime0=clock();
            // cout << endl << endl << "Finding the Tip coefficients..." << endl;
            
            mesh Ph;
            real[int] a(3); // list of coefficients of the expansion
            int exponant=2; // precision of the exponential
            
            cout.precision(12);
            cout << "kopytko"<<nbTips<<endl;
            for(int i=0;i<nbTips;++i)
            {{
                // cout << "Processing Tip " << i << " ";   
                x0=X(i);y0=Y(i);
                // cout << "(x0, y0) = (" << x0 << ", " <<y0<< "), angle = " << angle(i) << endl;

                // cout << "Projecting... Th.nv = " << Th.nv;
                Ph=trunc(Th,(sqrt((x-x0)^2+(y-y0)^2) < 3*R));
            	   // cout << ", Ph.nv = " << Ph.nv << endl;

                for(int order=1; order<=a.n; ++order){{
                    a[order-1]=
                    int2d(Ph)( u*exp(-(sqrt((x-x0)^2 + (y-y0)^2)/R)^exponant)
            		*BaseVector(order,exp(-angle(i)*1i)*( (x-x0) + (y-y0)*1i) ) );
                    
                    if (BC(i)=={DIRICHLET_1}) 
                    {{
                        a[order-1]-=int2d(Ph)( dirichletOut*exp(-(sqrt((x-x0)^2 + (y-y0)^2)/R)^exponant)
                		*BaseVector(order,exp(-angle(i)*1i)*( (x-x0) + (y-y0)*1i) ) );
                        a[order-1]*=-1;
                    }}
                        
                    a[order-1]/=(int2d(Ph)(exp(-(sqrt((x-x0)^2 + (y-y0)^2)/R)^exponant)
            		*square(BaseVector(order,exp(-angle(i)*1i)*( (x-x0) + (y-y0)*1i) ) )));
            		
            		cout << a[order-1] << " ";
                // cout << "a(" << order << ") = " << a[order-1] << endl;
                }}
            	// freefemOutput << Th.nv << " ";
            	// freefemOutput << Ph.nv << " ";
            	// freefemOutput << adaptCounter << " ";
            	
            	// cout << endl;
            }};
            cout << "kopytko" << "end";
            // cout << endl << endl << "Building mesh took: " << buildTime; 
            // cout << endl << "First adapt took: " << firstAdaptTime; 
            // cout << endl << "First run took: " << firstRunTime; 
            // cout << endl << "Adaptation took: " << adaptTime; 
            // cout << endl << "Calculating coefficients took: " << clock()- coeffTime0;
            // cout << endl << "Total time: " << clock()-time0 << endl << endl;
            """.format(DIRICHLET_1=DIRICHLET_1)
        )

    def update_scripts_leaf(self):
        
        add_after = lambda text, after_what, add_what: text.replace(after_what, after_what+add_what)
        
        self.is_leaf = True
        script_distance = textwrap.dedent("""
                        // Distance from the rim and required edges
                        fespace Vh1(Th,P1);
                        Vh1 u1,v1,dist;
                        varf vb(u1,v1) = on(2,u1=1); // Defines a variational form vb that imposes Dirichlet condition u1=1 on boundary label 2
                        Vh1 ub=vb(0, Vh1, tgv=1); // Solves the var. prob. to create boundary marker function ub; tgv=1 enables strong imposition of BC; ub=1 on boundary label 2, 0 elsewhere
                        ub[]=ub[] ? 0:1; //  inverts the marker values; Now ub marks interior points as 1, boundary 2 as 0
                        distance(Th,ub,dist[],distmax=100);
                        // plot(dist,wait=1,fill=1);
                        Vh1 distExp=exp(-dist/0.05);
                        plot(distExp, wait=true, fill=1);
                        """
                        )
        self.__script_adaptmesh = self.__script_adaptmesh.replace("\n// Solving the problem", \
                                          script_distance+"\n// Solving the problem")
        self.__script_adaptmesh = self.__script_adaptmesh.replace("nvAroundTips.min < 250", "nvAroundTips.min < 100")
        # self.__script_adaptmesh = self.__script_adaptmesh.replace("Th = adaptmesh(Th,1,", "// Th = adaptmesh(Th,1,")
        self.__script_adaptmesh = add_after(self.__script_adaptmesh, "keepbackvertices=1",",requirededges=reqEdgs")
        
        self.__script_init = """load "distance"\n""" + self.__script_init
        
        self.__script_adaptmesh = add_after( self.__script_adaptmesh, \
                                            "// counting cells around the tips",
                                            "\nint[int] reqEdgs=[{DIRICHLET_1}];".format(DIRICHLET_1=DIRICHLET_1) )
        self.__script_adaptmesh = add_after(self.__script_adaptmesh, \
                                            "u=u;", "distExp=distExp;")
        self.__script_adaptmesh = add_after(self.__script_adaptmesh, \
                                            "*R,nbTips)", ", distExp")
        
        self.__script_adaptmesh = self.__script_adaptmesh.replace("// plot(Th", "plot(Th")
    
        script_flux_rim = textwrap.dedent("""
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // CALCULATE FLUXES AT THE RIM
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            // Counting vertices on the rim
            func int countNvOnRim( mesh th, int tLabels)
            {{
                int nvOnRim=0;
                int1d(th, tLabels, qfe=qf1pE)( (nvOnRim++)*1.);
                return nvOnRim;
            }}
            int nvOnRim=countNvOnRim(Th, {DIRICHLET_1});

            // Calculating gradient
            Vh dxu,dyu;
            dxu=dx(u);
            dyu=dy(u);

            // Deteremining the flux coming to the tip
            // More on reading the field values in specific points:
            // https://www.ljll.math.upmc.fr/pipermail/freefempp/2013-July/002798.html
            // https://ljll.math.upmc.fr/pipermail/freefempp/2009/000337.html

            int ndof=countNvOnRim(Th, {DIRICHLET_1}), n=0;
            real[int] xs(ndof), ys(ndof), fluxes(ndof); // angles with X axis
            int1d(Th, {DIRICHLET_1}, qfe=qf1pE)( (xs(n++)=x)*1.
                                      +(ys(n)=y)*1.
                                       +(fluxes(n)=abs(dxu*N.x+dyu*N.y))*1.);
            // cout<<"tip"<<tipLabels(k)<<endl;
            cout<<"xs"<<xs<<"xs"<<"end"<<endl;
            cout<<"ys"<<ys<<"ys"<<"end"<<endl;
            cout<<"fluxes"<<fluxes<<"fluxes"<<"end"<<endl;
            real totGrad =  int1d(Th, {DIRICHLET_1})( abs([dxu,dyu]'*[N.x,N.y]) );
            cout<<"tot_flux"<<"1 "<<totGrad<<"tot_flux"<<"end"<<endl;

            // plot([angles,fluxes], wait=true);
            """.format(DIRICHLET_1=DIRICHLET_1)
        )
        self.__script_integrate = self.__script_integrate + script_flux_rim

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
        inside_buildmesh_box = ""
        border_box, inside_buildmesh_box = prepare_contour_list(border_box, inside_buildmesh="", i="", points=np.vstack((points, points[0])), label=connections_bc[:,2], ns_border=ns_border, border_name="box", i_tsh=1023)
        
        return border_box, inside_buildmesh_box

    def prepare_script(self, network):
        """Return a FreeFEM script with `network` geometry."""

        if self.is_leaf:
            self.__script_border_box, self.__script_inside_buildmesh_box = \
                self.prepare_script_box(network.box.connections_bc(), \
                                        network.box.points)

        tips = np.empty((len(network.active_branches), 4))
        border_network = ""
        inside_buildmesh = self.__script_inside_buildmesh_box
        for i, branch in enumerate(network.branches):
            # border_network, inside_buildmesh = prepare_contour_list(border_network, inside_buildmesh, i, branch.points, label=branch.BC, border_name="branch")
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
            # + "\nplot({inside_buildmesh}, dim=2, wait=true);\n\n".format(
                # inside_buildmesh=inside_buildmesh
            # )
            + "\nmesh Th = buildmesh({inside_buildmesh}, fixedborder=true);\n".format(
                inside_buildmesh=inside_buildmesh
            )
            + "\nreal buildTime=clock() - buildTime0;\n"
            + "// plot(Th, wait=true, fill=true);\n"
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
        script = script + self.__script_adaptmesh + self.__script_integrate

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

        script_name = "script_{ID}_{date}.edp".format(ID=id(self),
                        date=datetime.now().strftime("%Y_%m_%d-%p%I_%M_%S"))
        with open(script_name, "w") as edp_temp_file:
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
        
        if self.is_script_saved:
            out_freefem = self.run_freefem(script) # useful for debugging
            # print(out_freefem.stdout)
        else:
            out_freefem = self.run_freefem_temp(script)
            # print(out_freefem.stdout)
        
        if out_freefem.returncode:
            print("\nFreeFem++ failed in the first try.\n")
            print("stdout:", out_freefem.stdout.decode())
            print("stderr:", out_freefem.stderr.decode())
            out_freefem = self.run_freefem_temp(script.replace(\
                            "nvAroundTips.min < 250", \
                            "nvAroundTips.min < 350")
                                                )
            if out_freefem.returncode:
                print("\nFreeFem++ didn't work with stronger mesh adaptation.\n")
            
        ai_coeffs_flat = array_from_string(out_freefem.stdout, "kopytko")
        self.flux_info = ai_coeffs_flat.reshape(len(ai_coeffs_flat) // 3, 3)
        
        with np.printoptions(formatter={"float": "{:.6e}".format}):
            print("a1a2a3") # , self.flux_info)
            for i, branch in enumerate(network.active_branches):
                print("Branch {}:".format(branch.ID), self.flux_info[i], ", l={}".format(branch.length()))
        
        if self.is_backward:
            return self.flux_info.copy()
            
        if self.is_leaf:
            rim_xs = array_from_string(out_freefem.stdout,"xs")
            rim_ys = array_from_string(out_freefem.stdout,"ys")
            rim_fluxes = array_from_string(out_freefem.stdout,"fluxes")
            rim_xy_flux = np.stack((rim_xs, rim_ys, rim_fluxes)).T        
            return rim_xy_flux

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
            is_script_saved=False,
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
        self.is_script_saved = is_script_saved

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
        DIRICHLET_GLOB_FLUX_script = ""
        if (network.box.boundary_conditions==DIRICHLET_GLOB_FLUX).any(): 
            DIRICHLET_GLOB_FLUX_script = f"""
            // Normalize "u" if DIRICHLET_GLOB_FLUX BC
            real globFlux=int1d(Th, {DIRICHLET_GLOB_FLUX})( abs([dxu,dyu]'*[N.x,N.y])*mobility );
            u=u/globFlux;
            // Recalculate gradients
            dxu=dx(u);
            dyu=dy(u);
            """
        
        self.pbc = "" if network.box.boundary_conditions[0]!=2 else ", periodic=PBC"
        
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
        
        # contours based on the thickened tree
        box_ring, _, _, _, _, _, _ = \
            self.fingers_and_box_contours(network)
        self.script_border_box, self.script_inside_buildmesh_box = \
            self.prepare_script_box(network, box_ring)
        
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
            // DEFINING PROBLEM AND EQUATION TO SOLVE
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            
            fespace Vh(Th,P2{pbc});
            Vh u,v,dxu,dyu,du;
            
            problem potential(u,v,solver=CG)=
                     int2d(Th)(mobility*(dx(u)*dx(v) + dy(u)*dy(v)))
                                // -int2d(Th)(v) // rain in domain
                                -int1d(Th,{NEUMANN_1})(mobility*v)  // constant flux (local)
                                +on({DIRICHLET_GLOB_FLUX},u=0) // constant flux (global)
                                +on({DIRICHLET_0},u=0) // constant field
                                +on({DIRICHLET_1},u=1);
            """
        ).format(pbc=self.pbc, NEUMANN_1=NEUMANN_1, DIRICHLET_GLOB_FLUX=DIRICHLET_GLOB_FLUX, DIRICHLET_1=DIRICHLET_1, DIRICHLET_0=DIRICHLET_0)

        self.__script_problem_Poisson = self.__script_problem_Laplace.replace(
                                            "// -int2d(Th)(v) // rain in domain", 
                                            "-int2d(Th)(v) // rain in domain")

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
            // Th=adaptmesh(Th); // initial adaptation
            potential;
            cout<<"First solve completed."<<endl;
            real firstRunTime=clock() - firstRunTime0;
            
            // Adaptation loop
            real adaptTime0=clock();
            cout << endl << endl << "Adaptation..." << endl;
            real error=0.01;
            int adaptCounter=0;
            int[int] nvOnTips = countNvOnTips(Th, tipLabels, nbTips);
            plot(Th, wait=true);
            while(nvOnTips.min < 70 || adaptCounter<1)
            {{
                cout << "Adaptation step: " << adaptCounter;
                cout << ", nvOnTip.min = " << nvOnTips.min << endl;
                // plot(Th, wait=true, bb=[[x0Th, y0Th],[x1Th, y1Th]]);
                Th=adaptmesh(Th, [u/u[].max, 0.01*tipfield(X,Y,nbTips,tipR)],err=error,nbvx=500000,verbosity=0,nbsmooth=100,iso=1,ratio=1.8,keepbackvertices=1{pbc}); // Adapting mesh according to the first solution
                u=u;
                mobility=mobility;
                error=error/2;
                potential; // Solving one more time with adapted mesh
                nvOnTips = countNvOnTips(Th, tipLabels, nbTips);
                adaptCounter++;
                plot(Th, wait=true);
            }}
            cout << "Adaptation step: " << adaptCounter;
            cout << ", nvOnTip.min = " << nvOnTips.min << endl;
            plot(Th, wait=true);
            cout << "Problem solved." << endl;
            plot(u, wait=true, fill=true, value=true);
            
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
            {DIRICHLET_GLOB_FLUX_script}            
            // Deteremining the flux coming to the tip
            // More on reading the field values in specific points:
            // https://www.ljll.math.upmc.fr/pipermail/freefempp/2013-July/002798.html
            // https://ljll.math.upmc.fr/pipermail/freefempp/2009/000337.html
            // int avgWindow = 5;
            cout.precision(12);
            cout << "kopytko ";
            for(int k=0;k<nbTips;k++)
            {{
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
                // for (int i=0; i<=(ndof-avgWindow); i++){{
                    // real sumGrad=0, sumAng=0;
                    // for (int j=i; j<i+avgWindow; j++){{
                        // sumGrad += fluxes[j];
                        // sumAng += angles[j];
                    // }}
                    // fluxesMvAvg(i) = sumGrad / avgWindow;
                    // anglesMvAvg(i) = sumAng / avgWindow;
                    // if (fluxesMvAvg(i)>maxGrad){{
                        // maxGrad=fluxesMvAvg(i);
                        // maxAngle=anglesMvAvg(i);
                    // }}
                // }}
                // real totGrad =  int1d(Th, tipLabels(k))( abs([dxu,dyu]'*[N.x,N.y]) );
                // cout << totGrad << "," << maxAngle << ",";
            }}
            cout << "kopytko" << "end";
            """.format(DIRICHLET_GLOB_FLUX_script=DIRICHLET_GLOB_FLUX_script)
        )

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

    def prepare_script_box(self, network, box_ring):
        """Return parts of the FreeFEM script with the geometry of the ``network.box``."""
    
        box_ring_pts = np.asarray(box_ring.coords)
        p0 = network.box.points[0]
        p0_ind = np.where(np.logical_and(*(p0==box_ring_pts).T))[0][0]
        box_ring_pts = np.roll(box_ring_pts[:-1], -p0_ind, axis=0)
        
        connections_to_add = np.vstack(
            [np.arange(len(box_ring_pts)), np.roll(
                np.arange(len(box_ring_pts)), -1)]
        ).T
        
        border_nodes_mask = np.diff(network.box.boundary_conditions)!=0
        border_nodes = network.box.points[1:][border_nodes_mask]
        border_nodes_inds2 = np.where(inNd(box_ring_pts,border_nodes))[0] # ?assumes that the points are ordered (if not we need to use a solution like for the seeds)
        boundary_conditions = np.ones(len(connections_to_add), dtype=int)
        boundary_conditions[:border_nodes_inds2[1]] = network.box.boundary_conditions[0]
        bcs0 = network.box.boundary_conditions[1:][border_nodes_mask]
        for i, ind in enumerate(border_nodes_inds2[:-1]):
            boundary_conditions[ind:border_nodes_inds2[i+1]] = bcs0[i]    
        boundary_conditions[border_nodes_inds2[-1]:] = bcs0[-1] 
        # points_to_plot = box_ring_pts[connections_to_add]
        # for i, pts in enumerate(points_to_plot):
        #     plt.plot(*pts.T, '.-', ms=1, lw=5, \
        #     color="{}".format(boundary_conditions[i]/5))
        # for p in network.box.points[1:][border_nodes]:
        #     plt.plot(*p, '.',ms=20, c='r')
        
        border_box, inside_buildmesh_box = \
            FreeFEM.prepare_script_box(self, 
                                        np.column_stack((connections_to_add, boundary_conditions)), \
                                        box_ring_pts, \
                                        points_per_unit_len=0.5)
        
        return border_box, inside_buildmesh_box
    
    def fingers_and_box_contours(self, network):
        """ Prepares contours of the thickened tree using shapely library. """
        border_contour = ""
        inside_buildmesh = ""
        pts = []
        pts_in = [] # points inside subdomains with higher mobility
        tips_all = []; tips_active = [];
        for i, branch in enumerate(network.branches):
            
            # # don't take too much points
            # skeleton = [branch.points[0]]
            # segment_lengths = np.linalg.norm(branch.points[2:]-branch.points[1:-1], axis=1)
            # len_sum = 0
            # for j, seg in enumerate(segment_lengths):
            #     len_sum = len_sum + seg
            #     if len_sum>self.finger_width/2:
            #         len_sum = 0
            #         skeleton.append(branch.points[j+2])
            # skeleton[-1] = branch.points[-1]
            # pts.append(np.array(skeleton))
            
            pts.append(branch.points)
            
            if len(network.branch_connectivity)==0 or \
                    branch.ID not in network.branch_connectivity[:,0]:
                
                tips_all.append([1000+branch.ID, branch.points[-1]])
                pts_in.append(branch.points[-1])
                
                if branch in network.active_branches:
                    tips_active.append([1000+branch.ID, \
                                        branch.points[-1, 0], \
                                        branch.points[-1, 1] ]) # tip label, x, y
        tips_active = np.asarray(tips_active)
        
        # thicken tree and find intersection with the box
        tree = MultiLineString(pts)
        thick_tree = tree.buffer(distance=self.finger_width/2, cap_style=1, join_style=1, quad_segs=25)
        box_ring = LinearRing(network.box.points)
        box_polygon = Polygon(box_ring)
        box_ring = linemerge( [*box_ring.difference(thick_tree).geoms,
                          *box_ring.intersection(thick_tree).geoms])
        thick_tree = box_polygon.intersection(thick_tree)
        
        # polygons to contours_tree
        polygons = [thick_tree] if thick_tree.geom_type=="Polygon" else thick_tree.geoms
        contours_tree = []
        for i, poly in enumerate(polygons):
            pts_in.append(poly.representative_point().coords[0])
            poly = shapely.geometry.polygon.orient(poly) # now, exterior is ccw, but interiors are cw
            
            # exteriors
            poly_exterior = poly.exterior.difference(box_ring)
            lines = [poly_exterior] if poly_exterior.geom_type=="LineString" else poly_exterior.geoms
            for line in lines:
                line1 = line.simplify(tolerance=1e-4)
                contours_tree.append( np.array(line1.coords) )
            
            # interiors
            for ring in poly.interiors:
                ring1 = ring.simplify(tolerance=1e-4)
                contours_tree.append( np.asarray(ring1.coords) )
        
        contours_tree_bc = []
        for cont in contours_tree:
            contours_tree_bc.append(888)
            for b_label, tip_xy in tips_all:
                mask = np.linalg.norm(cont-tip_xy,axis=1)<self.finger_width/2*1.01
                if mask.any():
                    contours_tree_bc[-1] = ~mask*888 + mask*b_label
        
        return box_ring, contours_tree, contours_tree_bc, tips_active, pts_in, border_contour, inside_buildmesh
    
    def prepare_script(self, network):
        """Return a full FreeFEM script with the `network` geometry."""
        # contours based on the thickened tree
        box_ring, contours_tree, contours_tree_bc, tips, points_in, border_contour, inside_buildmesh = \
            self.fingers_and_box_contours(network)
           
        self.script_border_box, self.script_inside_buildmesh_box = \
            self.prepare_script_box(network, box_ring)
                       
        # contours_tree to border
        for i, points in enumerate(contours_tree):
            # points = np.flip(points0, axis=0)
            border_contour, inside_buildmesh = \
                prepare_contour_list(border_contour, inside_buildmesh, i, points, \
                                     label=contours_tree_bc[i], border_name="contour" )
        
        inside_buildmesh = self.script_inside_buildmesh_box + inside_buildmesh[:-1]
        
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
        if network.box.boundary_conditions[0]==2:
            buildmesh = buildmesh + \
                "func PBC=[[{left},y],[{right},y]];".format( left=LEFT_WALL_PBC, right=RIGHT_WALL_PBC)
        buildmesh = buildmesh + \
            "\nplot({inside_buildmesh},dim=2);\nmesh Th = buildmesh({inside_buildmesh}, fixedborder=true);\n".format(
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
            plot(mobility, wait=true, cmm="mobility", fill=true, value=true);
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
        
        if self.is_script_saved:
            out_freefem = FreeFEM.run_freefem(self, script) # useful for debugging
            # print(out_freefem.stdout)
        else:
            out_freefem = FreeFEM.run_freefem_temp(self, script)
            # print(out_freefem.stdout)

        if out_freefem.returncode:
            print("\nFreeFem++ failed in the first try.\n")    
            print(out_freefem.stdout)
        
        # flux_info calculated in FreeFem:
        # flux_info = np.fromstring(out_freefem.stdout[out_freefem.stdout.find(b"kopytko")+7:], sep=",")
        # self.flux_info = flux_info.reshape(len(flux_info) // 2, 2)
        
        # determining flux_info 
        angles=[]; fluxes=[]; 
        self.flux_info = np.zeros((len(network.active_branches), 2))
        for i, branch in enumerate(network.active_branches):
            tip_label = 1000+branch.ID
            angles.append(array_from_string(out_freefem.stdout, f"angles{tip_label}"))
            fluxes.append(array_from_string(out_freefem.stdout, f"fluxes{tip_label}")) 
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
            self.flux_info[i,0] = array_from_string(out_freefem.stdout, f"tot_flux{tip_label}")
            # Highest gradiend direction (angle with X axis)
            ang_max = xx[np.argmax(smoothed)]
            self.flux_info[i,1] = (ang_max + np.pi) % (2*np.pi) - np.pi
            
        # with np.printoptions(formatter={"float": "{:.6g}".format}):
        #     # print("flux_info: \n", self.flux_info)
        #     print("angles: \n", self.flux_info[...,1]*180/np.pi)
