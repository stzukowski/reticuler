import logging
import textwrap
import numpy as np

from reticuler import DIRICHLET_1, DIRICHLET_0, NEUMANN_1

logger = logging.getLogger("reticuler")
from reticuler import rotation_matrix

from reticuler.utilities.misc import sigmoid
from reticuler.utilities.misc import LEFT_WALL_PBC, RIGHT_WALL_PBC, DIRICHLET_1, DIRICHLET_0, NEUMANN_0, NEUMANN_1, DIRICHLET_0_GLOB_FLUX
from reticuler.utilities.geometry import Branch
from reticuler.extending_kernels.pde_solvers.freefem import FreeFEM

class FreeFEM_ThinFingers(FreeFEM):
    """A PDE solver based on the finite element method implemented in FreeFEM [Ref2]_.
    Solves PDE, computes a1a2a3 coefficients and forges it into tip trajectory with the streamline algorithm [Ref1]_.

    Attributes
    ----------
    equation, eta, ds, is_script_saved inherited from FreeFEM

    flux_info : array
        A 3-n array of a1a2a3 coefficients for each tip in the network.

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
    distance_from_bif_thresh : float, default 2.1*``ds``
        A minimal distance the tip has to move from the previous bifurcations
        to split again.
    sleep_frac_thresh : float, default 0.01
        Numerical threshold to put asleep tips that are slower than
        ``sleep_frac_thresh`` * max velocity.
    crit_shields_param : float, default 0
        Threshold in the growth law:
        v = (flux - ``crit_shields_param``)**``eta''        

    is_backward : bool, default False
        If True, solve_PDE returns flux_info.

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
            is_script_saved=False,
            bifurcation_type=0,
            bifurcation_thresh=None,
            bifurcation_angle=2 * np.pi / 5,
            distance_from_bif_thresh=None,
            sleep_frac_thresh=0.01,
            crit_shields_param=0,            
            is_backward=False,
        ):
        """Initialize FreeFEM_ThinFingers.

        Parameters
        ----------
        network : Network
        equation : int, default 0
        eta : float, default 1.0
        ds : float, default 0.01
        bifurcation_type : int, default 0
        bifurcation_thresh : float, default 0
        bifurcation_angle : float, default 2pi/5
        distance_from_bif_thresh : float, default 2.1*``ds``
        sleep_frac_thresh : float, default 0.01
        crit_shields_param : float, default 0        
        is_backward : bool, default False

        Returns
        -------
        None.

        """
        super().__init__(equation, eta, ds, is_script_saved)
        self.crit_shields_param = crit_shields_param # v = (flux - crit_shields_param)**eta
        self.is_backward = is_backward
        
        self.bifurcation_type = bifurcation_type  # no bifurcations, a1, a3/a1, random
        self.bifurcation_thresh = bifurcation_thresh
        if bifurcation_thresh is None:
            if self.bifurcation_type == 1:
                self.bifurcation_thresh = 0.8  # a1 bifurcations
            elif self.bifurcation_type == 2:
                self.bifurcation_thresh = -0.1  # a3/a1 bifurcations
            elif self.bifurcation_type == 3:
                self.bifurcation_thresh = 5 * ds # random bifurcations: bif_probability
            else:
                self.bifurcation_thresh = 0
        self.bifurcation_angle = bifurcation_angle  # 2*np.pi/5
        self.distance_from_bif_thresh = 2.1 * ds if distance_from_bif_thresh is None else distance_from_bif_thresh

        # less than ``sleep_frac_thresh`` of max flux/velocity puts branches asleep
        self.sleep_frac_thresh = sleep_frac_thresh
        
        # parts of the script
        self._script_init = textwrap.dedent(
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
        
        self._script_border_box, self._script_inside_buildmesh_box = \
            super().prepare_script_box(network.box.connections_bc(), \
                                        network.box.points)
        
        self._script_problem = textwrap.dedent(
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

        if self.equation==1:
            self._script_problem = self._script_problem.replace(
                                            "// -int2d(Th)(v) // rain in domain", 
                                            "-int2d(Th)(v) // rain in domain")

        self._script_adaptmesh = textwrap.dedent(
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
            plot(Th, wait=true);
            
            // Solving the problem for the first time
            real firstRunTime0=clock();
            potential;
            // cout<<"First solve completed."<<endl;
            real firstRunTime=clock() - firstRunTime0;
            
            // Adaptation loop
            real adaptTime0=clock();
            // cout << endl << endl << "Adaptation..." << endl;
            fespace Vh0(Th,P0); // Vh0 h=1;
            real error=0.02;
            int adaptCounter=1;
            while(nvAroundTips.min < 250 || adaptCounter<=3)
            {
            	// cout << "Adaptation step: " << adaptCounter << ", h[].min = " << h[].min;
                // h=hTriangle; // the triangle size
            	// cout << ", nvAroundTip.min = " << nvAroundTips.min << endl;
            	Th=adaptmesh(Th,[u/u[].max, 20.*tipfield(X,Y,3.*R,nbTips)],err=error,nbvx=1000000,iso=true,ratio=2,hmin=1e-5,keepbackvertices=1);
            	u=u;
                error=error/2;
                potential;
            	nvAroundTips = countNvAroundTips (3.*R, Th, Th.nv, nbTips, X, Y);
            	adaptCounter++;
                plot(Th, wait=true);
            }
            
            // cout << endl << "Adaptation finished." << " h[].min = " << h[].min;
            // cout << ", nvAroundTip.min = " << nvAroundTips.min << endl;
            // cout << "Problem solved." << endl;
            plot(u, wait=true, fill=true, value=true);
            
            real adaptTime=clock() - adaptTime0;
            """
        )

        # // ofstream freefemOutput("{file_name}");
        self._script_tip_integration = textwrap.dedent(
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
        is_dr_normalized : bool
        is_zero_approx_step : bool, default False

        Returns
        -------
        dRs_test : array
            An n-2 array with dx and dy shifts for each tip.
        dt : float

        """
        max_a1 = np.max(self.flux_info[..., 0])
        if is_dr_normalized:
            # normalize dr, so that the fastest tip moves over ds
            v_max = np.maximum( (max_a1 - self.crit_shields_param)** self.eta, 1e-12)
            dt = self.ds / v_max
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
                # moving condition
                if a1 < self.crit_shields_param and \
                    ( (a1-self.crit_shields_param) / (max_a1-self.crit_shields_param) )**self.eta < self.sleep_frac_thresh:
                    is_moving = False

                # bifurcation
                if (
                    self.bifurcation_type
                    and branch.length() > self.distance_from_bif_thresh
                    and is_moving
                ):
                    # the condition above is used to avoid many bifurcations
                    # in almost one point which can occur while ds is very small
                    if (self.bifurcation_type == 1 and a1 > self.bifurcation_thresh) or (
                        self.bifurcation_type == 2 and a3 / a1 < self.bifurcation_thresh
                    ):
                        is_bifurcating = True
                    elif self.bifurcation_type == 3:
                        p = self.bifurcation_thresh * dt * a1**self.eta
                        r = np.random.uniform(0, 1)  # uniform distribution [0,1)
                        if p > r:
                            is_bifurcating = True
            
            if is_moving:
                # __streamline_extension formula is derived in the coordinate
                # system where the tip segment lies on a negative Y axis;
                # hence, we rotate obtained dR vector to that system
                tip_angle = np.pi / 2 - branch.tip_angle()
                dr = dt * (a1 - self.crit_shields_param)**self.eta
                beta = a1 / a2
                dRs_test[i] = np.dot(
                    rotation_matrix(
                        tip_angle), self.__streamline_extension(beta, dr)
                )
            else:
                dRs_test[i] = -10
                network.sleeping_branches.append(branch)
                network.active_branches.remove(branch)
                logger.info("! Branch %d is sleeping !", branch.ID)

            if is_bifurcating:
                logger.info("! Branch %d bifurcated !", branch.ID)
                dR = dRs_test[i].copy() 
                dRs_test[i] = -10
                dRs_test = np.vstack( (dRs_test, [
                    np.dot(
                        rotation_matrix(-self.bifurcation_angle / 2), dR),
                    np.dot(rotation_matrix(
                        self.bifurcation_angle / 2), dR) ]) )

                # create new branches
                max_branch_id = len(network.branches) - 1
                for j, dR in enumerate(dRs_test[-2:]):
                    branch_new = Branch(
                        ID=max_branch_id + j + 1,
                        BC=branch.BC,
                        points=np.array([branch.points[-1]]),
                        steps=np.array([branch.steps[-1]]),
                    )
                    network.branches.append(branch_new)
                    network.active_branches.append(branch_new)
                    network.add_connection([branch.ID, branch_new.ID])
                network.active_branches.remove(branch)
            else:
                branch.dR = dRs_test[i]
            
        dRs_test = dRs_test[dRs_test[:,0]>-10]
        return dRs_test, dt

    def prepare_script(self, network):
        """Return a FreeFEM script with ``network`` geometry."""

        tips = np.empty((len(network.active_branches), 4))
        border_network = ""
        inside_buildmesh = self._script_inside_buildmesh_box
        for i, branch in enumerate(network.branches):
            border_network, inside_buildmesh = super().prepare_contour_list(border_network, inside_buildmesh, i, branch.points, label=branch.BC, border_name="branch")
            # border_network, inside_buildmesh = super().prepare_contour(border_network, inside_buildmesh, i, branch.points, label=branch.BC, border_name="branch")
            if branch in network.active_branches:
                ind = network.active_branches.index(branch)
                tips[ind, 0] = branch.BC # boundary condition
                tips[ind, 1] = branch.tip_angle() # angle with X axis
                tips[ind, 2] = branch.points[-1, 0] 
                tips[ind, 3] = branch.points[-1, 1]
        inside_buildmesh = inside_buildmesh[:-2]

        buildmesh = (
            textwrap.dedent(
                """
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // BUILDING MESH
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            """
            )
            + self._script_border_box
            + border_network
            + "\nplot({inside_buildmesh}, dim=2, wait=true);\n\n".format(
                inside_buildmesh=inside_buildmesh
            )
            + "\nmesh Th = buildmesh({inside_buildmesh}, fixedborder=true);\n".format(
                inside_buildmesh=inside_buildmesh
            )
            + "\nreal buildTime=clock() - buildTime0;\n"
            + "plot(Th, wait=true);\n"
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
                bc=super().arr2str(tips[:,0]),
                angle=super().arr2str(tips[:,1]),
                x=super().arr2str(tips[:,2]),
                y=super().arr2str(tips[:,3]),
            )
        )

        script = self._script_init + buildmesh + tip_information + \
                    self._script_problem + self._script_adaptmesh + \
                        self._script_tip_integration

        return script

    def solve_PDE(self, network, is_out_freefem_returned=False):
        """Solve the PDE for the field around the network.

        Prepare a FreeFEM script, export it to a temporary file and run.
        Then, import the a1a2a3 coefficients to ``self.flux_info``.

        Parameters
        ----------
        network : object of class Network
            Network around which the field will be calculated.

        Returns
        -------
        None.

        """
        script = self.prepare_script(network)
        
        out_freefem = super().run_freefem(script)
        if out_freefem.returncode or "nan" in out_freefem.stdout.decode():
            logger.warning("Trying again...")
            # script_perturbed = script.replace("nvAroundTips.min < 250","nvAroundTips.min < 350")
            lookfor = "boxN(0:1023)=["
            ind = script.find(lookfor) + len(lookfor)
            ind2 = ind + script[ind:].find(",")
            boxN_0 = script[ind:ind2]
            script_perturbed = script.replace(lookfor+boxN_0,lookfor+str(int(boxN_0)+1))
            out_freefem = super().run_freefem(script_perturbed)
            
        ai_coeffs_flat = self.array_from_string(out_freefem.stdout, "kopytko")
        self.flux_info = ai_coeffs_flat.reshape(len(ai_coeffs_flat) // 3, 3)
        
        with np.printoptions(formatter={"float": "{:.6e}".format}):
            logger.info("a1a2a3")
            for i, branch in enumerate(network.active_branches):
                logger.info("Branch %d: %s, l=%.3g", branch.ID, self.flux_info[i], branch.length())
        
        if self.is_backward:
            return self.flux_info.copy()
        if is_out_freefem_returned:
            return out_freefem
        
class FreeFEM_ThinFingers_Boundary(FreeFEM_ThinFingers):
    r"""A PDE solver based on the finite element method implemented in FreeFEM [Ref2]_.
    Solves PDE, computes a1a2a3 coefficients and forges it into tip trajectory with the streamline algorithm [Ref1]_.
    Extension for co-evolving boundary case.
    
    Attributes
    ----------
    equation, eta, ds, is_script_saved inherited from FreeFEM

    flux_info, bifurcation_type, bifurcation_thresh, 
        bifurcation_angle, distance_from_bif_thresh, 
        sleep_frac_thresh, crit_shields_param inherited from FreeFEM_ThinFingers

    flux_info_boundary : array
        A 1-n array of fluxes at each point of the moving-boundary.
    boundary_pts_sep_min : float, default 0.015
        Minimum separation between boundary points. If too small introduces numerical noise.
    boundary_pts_sep_max : float, default 0.03
        Maximum separation between boundary points.
    box_history : list, default []
        History of the box at each step.
    v_rim : float, default 1.0
        How fast the moving-boundary grows. The factor: $\sqrt{R/L}*\alfa_{boundary} / \alfa_{tip}$ in Maciej's Pawlus thesis.
    response_func : str, default "linear"
        Function to map fluxes to boundary growth velocity.
    response_func_params : dict, default {}
        Parameters for the response function.        

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
            is_script_saved=False,
            bifurcation_type=0,
            bifurcation_thresh=None,
            bifurcation_angle=2 * np.pi / 5,
            distance_from_bif_thresh=None,
            sleep_frac_thresh=0.01,
            crit_shields_param=0,
            boundary_pts_sep_min=0.019,
            boundary_pts_sep_max=0.021,
            box_history=None,
            v_rim=1,
            response_func="linear",
            response_func_params={},
        ):
        """Initialize FreeFEM_ThinFingers_Boundary.

        Parameters
        ----------
        network : Network
        equation : int, default 0
        eta : float, default 1.0
        ds : float, default 0.01
        bifurcation_type : int, default 0
        bifurcation_thresh : float, default 0
        bifurcation_angle : float, default 2pi/5
        distance_from_bif_thresh : float, None
        sleep_frac_thresh : float, default 0.01
        crit_shields_param : float, default 0
        boundary_pts_sep_min : float, default 0.015
        boundary_pts_sep_max : float, default 0.03
        box_history : list, default []
        v_rim : float, default 1.0
        response_func : str, default "linear"
        response_func_params : dict, default {}        

        Returns
        -------
        None.

        """
        super().__init__(network, equation, eta, ds, is_script_saved, \
                         bifurcation_type, bifurcation_thresh, bifurcation_angle, \
                         distance_from_bif_thresh, sleep_frac_thresh, crit_shields_param)
        self.boundary_pts_sep_min = boundary_pts_sep_min
        self.boundary_pts_sep_max = boundary_pts_sep_max
        if box_history is None:
            self.box_history = [network.box.copy()]
        else:
            self.box_history = box_history
        self.v_rim = v_rim
        if response_func == "linear":
            self.response_func = self.linear
        if response_func == "linear_sigmoid":
            self.response_func = self.linear_sigmoid

        self.response_func_params = response_func_params        
        
        add_after = lambda text, after_what, add_what: text.replace(after_what, after_what+add_what)
        
        script_distance_0 = textwrap.dedent("""
                        // Distance from the rim and required edges
                        fespace Vh1(Th,P1);
                        Vh1 u1,v1,dist;
                        varf vb(u1,v1) = on(2,u1=1); // Defines a variational form vb that imposes Dirichlet condition u1=1 on boundary label 2
                        Vh1 ub=vb(0, Vh1, tgv=1); // Solves the var. prob. to create boundary marker function ub; tgv=1 enables strong imposition of BC; ub=1 on boundary label 2, 0 elsewhere
                        ub[]=ub[] ? 0:1; //  inverts the marker values; Now ub marks interior points as 1, boundary 2 as 0
                        distance(Th,ub,dist[],distmax=100);
                        // plot(dist,wait=1,fill=1);
                        Vh1 distExp=1 - 0.3*erf(1) + 0.3*erf(sqrt(dist^2/(2*(R)^2)));
                        plot(distExp, wait=true, fill=1);
                        """
                        )
        
        script_distance_1 = textwrap.dedent("""
                        u=u; ub=ub;dist=dist;distExp=distExp;
                        varf vbLoop(u1,v1) = on(2, u1=1);
                        ub[] = vbLoop(0, Vh1, tgv=1);
                        ub[] = ub[] ? 0 : 1;
                        distance(Th, ub, dist[], distmax=100);
                        distExp = 1 - 0.3*erf(1) + 0.3*erf(sqrt(dist^2/(2*R^2)));        
                        """
                        )

        self._script_init = """load "distance"\n""" + self._script_init

        self._script_adaptmesh = self._script_adaptmesh.replace("\n// Solving the problem", \
                                          script_distance_0+"\n// Solving the problem")
        self._script_adaptmesh = self._script_adaptmesh.replace("nvAroundTips.min < 250", "nvAroundTips.min < 100")
        self._script_adaptmesh = self._script_adaptmesh.replace("adaptCounter<=3", "adaptCounter<=2")
        self._script_adaptmesh = self._script_adaptmesh.replace("u=u;", script_distance_1)
        # self._script_adaptmesh = self._script_adaptmesh.replace("Th = adaptmesh(Th,1,", "// Th = adaptmesh(Th,1,")
        self._script_adaptmesh = add_after(self._script_adaptmesh, "keepbackvertices=1",",requirededges=reqEdgs")
        self._script_adaptmesh = add_after( self._script_adaptmesh, \
                                            "// counting cells around the tips",
                                            f"\nint[int] reqEdgs=[{DIRICHLET_1}];" )
        self._script_adaptmesh = add_after(self._script_adaptmesh, \
                                            "*R,nbTips)", ", 20*distExp")
    
        script_flux_rim = textwrap.dedent(f"""
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // CALCULATE FLUXES AT THE RIM
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            // Counting vertices on the rim
            func int countNvOnRim( mesh th, int tLabels)
            {{
                int nvOnRim=0;
                int1d(th, tLabels, qfe=qf1pElump)( (nvOnRim++)*1.);
                return nvOnRim;
            }}
            int nvOnRim=countNvOnRim(Th, {DIRICHLET_1});

            // Calculating gradient
            fespace Wh(Th, P1);
            // Marker = 1 on label-{DIRICHLET_1} vertices, 0 everywhere else
            Wh rimMarker;
            {{
                varf vMark(uu, vv) = on({DIRICHLET_1}, uu=1);
                rimMarker[] = vMark(0, Wh, tgv=1);
            }}
            // Boundary mass on the rim and bulk-gradient RHS
            varf vMassRim(uu, vv)  = int1d(Th, {DIRICHLET_1})(uu * vv);
            varf vBulkGrad(uu, vv) = int2d(Th)(dx(u)*dx(vv) + dy(u)*dy(vv));
            matrix Mrim     = vMassRim(Wh, Wh);
            real[int] rhsB  = vBulkGrad(0, Wh);
            // Lock every non-rim dof to 0 by overwriting its diagonal with tgv
            // and zeroing the corresponding RHS entry. This handles interior dofs,
            // label-1 dofs, and label-4 dofs all in one pass.
            real tgvLock = 1e30;
            real[int] diag = Mrim.diag;
            for (int i = 0; i < diag.n; i++)
            {{
                if (rimMarker[][i] < 0.5)
                {{
                    diag[i] = tgvLock;
                    rhsB[i] = 0;
                }}
            }}
            Mrim.diag = diag;
            set(Mrim, solver=sparsesolver);
            Wh fluxN;
            fluxN[] = Mrim^-1 * rhsB;
            Wh fluxAbs = abs(fluxN);

            // Alternatively use FreeFEM 'solve' to get smoothed gradients (less precise)
            // fespace Wh(Th, P1);
            // Wh dxs, dys, ww;
            // solve projX(dxs, ww) = int2d(Th)(dxs*ww) - int2d(Th)(dx(u)*ww);
            // solve projY(dys, ww) = int2d(Th)(dys*ww) - int2d(Th)(dy(u)*ww);
            // Wh fluxAbs = sqrt(dxs*dxs + dys*dys);



            // Determining the flux coming from the rim
            // More on reading the field values in specific points:
            // https://www.ljll.math.upmc.fr/pipermail/freefempp/2013-July/002798.html
            // https://ljll.math.upmc.fr/pipermail/freefempp/2009/000337.html

            int ndof=countNvOnRim(Th, {DIRICHLET_1}), n=0;
            real[int] xs(ndof), ys(ndof), fluxes(ndof);
            int1d(Th, {DIRICHLET_1}, qfe=qf1pElump)( (xs(n++)=x)*1.
                                      +(ys(n)=y)*1.
                                       // +(fluxes(n)=abs(dxu*N.x+dyu*N.y))*1.);
                                        +(fluxes(n)=fluxAbs)*1.);
            // cout<<"tip"<<tipLabels(k)<<endl;
            cout<<"xs"<<xs<<"xs"<<"end"<<endl;
            cout<<"ys"<<ys<<"ys"<<"end"<<endl;
            cout<<"fluxes"<<fluxes<<"fluxes"<<"end"<<endl;
            real totGrad =  int1d(Th, {DIRICHLET_1})( fluxAbs ); // abs([dxu,dyu]'*[N.x,N.y]) );
            cout<<"tot_flux"<<"1 "<<totGrad<<"tot_flux"<<"end"<<endl;

            real[int] indices(ndof);
            for(int i=0; i<ndof; i++) indices[i] = i;
            plot([indices, fluxes], wait=true);
            """
        )
        self._script_tip_integration = self._script_tip_integration + script_flux_rim
    
    def linear(self, fluxes):
        return self.v_rim*fluxes
    def linear_sigmoid(self, fluxes, sig_shift, sig_rate):
        return self.v_rim * fluxes * sigmoid(fluxes, sig_shift=sig_shift, sig_rate=sig_rate, sig_h=1)

    def control_point_density(self, network, boundary_pts):
        ###### CONTROL POINT DENSITY ######
        if network.box.initial_condition==350: # full circle
            boundary_pts = np.vstack((boundary_pts, boundary_pts[0]))
        processed_points = [boundary_pts[0]]
        for i in range(1, len(boundary_pts)):
            last_kept_point = processed_points[-1]
            current_point = boundary_pts[i]
            distance = np.linalg.norm(current_point - last_kept_point)
            # Too far apart -> Insert interpolated points
            if distance > self.boundary_pts_sep_max:
                # Calculate how many segments of self.boundary_pts_sep_max length fit between the points
                num_segments = int(np.ceil(distance / self.boundary_pts_sep_max))
                # Use linear interpolation to generate the intermediate points
                t_values = np.linspace(0, 1, num_segments + 1)
                interpolated_points = (1 - t_values[:, np.newaxis]) * last_kept_point + t_values[:, np.newaxis] * current_point
                # Add all new points to the list, except the first (which is last_kept_point)
                processed_points.extend(interpolated_points[1:])
            # Accepted separation range -> Keep the point
            elif distance >= self.boundary_pts_sep_min:
                processed_points.append(current_point)
            # If the last two points are too close, replace the penultimate point with the last
            elif i==len(boundary_pts)-1:
                processed_points[-1] = current_point
        if network.box.initial_condition==350: # full circle
            processed_points.pop(-1)
                    
        processed_points = np.array(processed_points)
        x, y = processed_points[:,0], processed_points[:,1]
            
        ###### UPDATE BOX ######
        n_seeds = len(network.box.seeds_connectivity)
        
        # POINTS
        if network.box.initial_condition==300: # rectangle, seeds vertically at the bottom
            # shift top right/left corner vertically only
            network.box.points = np.vstack(( network.box.points[0], np.stack((x,y)).T, network.box.points[-1-n_seeds:] ))
            network.box.points[1,0] = network.box.points[0,0]
            network.box.points[-2-n_seeds,0] = 0
        if network.box.initial_condition==301: # semiellipse, seeds vertically at the bottom boundary
            # shift bottom right/left 'corner' horizontally only
            network.box.points = np.vstack(( np.stack((x,y)).T, network.box.points[-n_seeds:] ))
            network.box.points[0,1] = 0
            network.box.points[-1-n_seeds,1] = 0
        if network.box.initial_condition==350: # full circle, seeds in the center
            network.box.points = np.stack((x,y)).T
        if network.box.initial_condition==351: # slice, seeds in the center
            # keep angular width
            ang_width = np.atan2(*network.box.points[0])*2
            network.box.points = np.vstack(( np.stack((x,y)).T, [0,0] ))
            x_right = network.box.points[0,0]
            y_right = network.box.points[0,1]
            network.box.points[0,0] = ((1-np.cos(ang_width))*x_right + np.sin(ang_width)*y_right)/2
            network.box.points[0,1] = ((1+np.cos(ang_width))*y_right + np.sin(ang_width)*x_right)/2
            x_left = network.box.points[-2,0]
            y_left = network.box.points[-2,1]
            network.box.points[-2,0] = ((1-np.cos(ang_width))*x_left - np.sin(ang_width)*y_left)/2
            network.box.points[-2,1] = ((1+np.cos(ang_width))*y_left - np.sin(ang_width)*x_left)/2
        # SEEDS_CONNECTIVITY and CONNECTIONS
        network.box.seeds_connectivity = np.column_stack(
                    (
                        len(network.box.points) - n_seeds + np.arange(n_seeds),
                        np.arange(n_seeds),
                    )
                )
        network.box.connections = np.vstack(
            [np.arange(len(network.box.points)), np.roll(
                np.arange(len(network.box.points)), -1)]
        ).T
        # BOUNDARY_CONDITIONS
        network.box.boundary_conditions = DIRICHLET_1 * np.ones(len(network.box.connections), dtype=int)
        if network.box.initial_condition==300:
            network.box.boundary_conditions[0] = NEUMANN_0
            network.box.boundary_conditions[-2-n_seeds] = NEUMANN_0
            network.box.boundary_conditions[-1-n_seeds:] = DIRICHLET_0
        if network.box.initial_condition==301 or (network.box.initial_condition==351):
            network.box.boundary_conditions[-1-n_seeds:] = NEUMANN_0

    def find_test_dRs_boundary(self, network, dt):
        # moving-boundary points
        mask = network.box.boundary_conditions==DIRICHLET_1
        mask[1:] |= mask[:-1].copy() # "spill" True on point i+1 (i-last detected)
        x = network.box.points[mask,0]
        y = network.box.points[mask,1]

        # VELOCITY
        velocity = self.response_func(self.flux_info_boundary, **self.response_func_params)

        ####### find dRs ######
        vx = np.diff(x,prepend=2*x[0]-x[1],append=2*x[-1]-x[-2]) # warunki na brzegach = symetria względem ostatniego punktu
        vy = np.diff(y,prepend=2*y[0]-y[1],append=2*y[-1]-y[-2])
        if network.box.initial_condition==300:
            vy = np.diff(y,prepend=y[1],append=y[-2]) # warunki na brzegach = odbicie względem osi pionowej (tylko dla prostokątów)
        if network.box.initial_condition==350:
            vx = np.diff(x,prepend=x[-1],append=x[0]) # warunki na brzegach = cykliczne (tylko dla pełnego koła)
            vy = np.diff(y,prepend=y[-1],append=y[0])
        alfa = (np.arctan2(-vy[:-1],-vx[:-1])+np.arctan2(vy[1:],vx[1:]))/2 # kąt nachylenia dwusiecznej (między 1->0 a 1->2)
        s = velocity*dt
        sx = s*np.cos(alfa) # definicja dwusiecznej i wartość przesunięcia z fluxów
        sy = s*np.sin(alfa)

        # x += (2*(vx[1:]*sy<vy[1:]*sx)-1)*sx # przesuwanie punktów (zmiana znaku nierówności zmieni zwrot)
        # y += (2*(vx[1:]*sy<vy[1:]*sx)-1)*sy        
        dRs_boundary = np.array([(2*(vx[1:]*sy<vy[1:]*sx)-1)*sx, \
                         (2*(vx[1:]*sy<vy[1:]*sx)-1)*sy]).T

        return dRs_boundary

    def prepare_script(self, network):
        """Return a FreeFEM script with ``network`` geometry."""

        self._script_border_box, self._script_inside_buildmesh_box = \
            super().prepare_script_box(network.box.connections_bc(), \
                                        network.box.points)
        
        script = super().prepare_script(network)

        return script

    def solve_PDE(self, network):
        """Solve the PDE for the field around the network.

        Additionally, compute fluxes at the growing boundary.

        Parameters
        ----------
        network : object of class Network
            Geometry and boundary conditions to solve PDE.

        Returns
        -------
        None.

        """

        out_freefem = super().solve_PDE(network, is_out_freefem_returned=True)

        rim_xs = self.array_from_string(out_freefem.stdout,"xs")
        fluxes = self.array_from_string(out_freefem.stdout,"fluxes")

        rim_xs2 = np.concatenate(([rim_xs[1]], (rim_xs[:-2:2]+rim_xs[3::2])/2, [rim_xs[-2]]))
        self.flux_info_boundary = np.concatenate(([fluxes[1]], (fluxes[:-2:2]+fluxes[3::2])/2, [fluxes[-2]]))
        
        if network.box.initial_condition==350:
            self.flux_info_boundary = self.flux_info_boundary[:-1]

        # import matplotlib.pyplot as plt
        # plt.plot(self.flux_info_boundary); plt.show;
        # print(1)
