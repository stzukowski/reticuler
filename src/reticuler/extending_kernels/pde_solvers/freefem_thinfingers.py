import textwrap
import numpy as np

from reticuler import DIRICHLET_1, DIRICHLET_0, NEUMANN_1
from reticuler import rotation_matrix

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
    inflow_thresh : float, default 0.05
        Threshold to put asleep the tips with less than ``inflow_thresh``
        of max flux/velocity.
    distance_from_bif_thresh : float, default 2.1*``ds``
        A minimal distance the tip has to move from the previous bifurcations
        to split again.

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
            inflow_thresh=0.05,
            distance_from_bif_thresh=None,
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
        inflow_thresh : float, default 0.05
        distance_from_bif_thresh : float, default 2.1*``ds``
        is_backward : bool, default False

        Returns
        -------
        None.

        """
        super().__init__(equation, eta, ds, is_script_saved)
        
        self.is_backward = is_backward
        
        self.bifurcation_type = bifurcation_type  # no bifurcations, a1, a3/a1, random
        self.bifurcation_thresh = bifurcation_thresh
        if bifurcation_thresh is None:
            if self.bifurcation_type == 1:
                self.bifurcation_thresh = 0.8  # a1 bifurcations
            elif self.bifurcation_type == 2:
                self.bifurcation_thresh = -0.1  # a3/a1 bifurcations
            elif self.bifurcation_type == 3:
                self.bifurcation_thresh = 3 * ds # random bifurcations: bif_probability
            else:
                self.bifurcation_thresh = 0
        self.bifurcation_angle = bifurcation_angle  # 2*np.pi/5

        # less than ``inflow_thresh`` of max flux/velocity puts branches asleep
        self.inflow_thresh = inflow_thresh
        self.distance_from_bif_thresh = 2.1 * ds if distance_from_bif_thresh is None else distance_from_bif_thresh
        
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
                plot(Th, wait=true);
            }
            
            // cout << endl << "Adaptation finished." << " h[].min = " << h[].min;
            // cout << ", nvAroundTip.min = " << nvAroundTips.min << endl;
            
            // solving with adapted mesh
            potential;
            // cout << "Problem solved." << endl;
            plot(Th, wait=true);
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
            print("\nTrying again...\n")
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
            print("a1a2a3") # , self.flux_info)
            for i, branch in enumerate(network.active_branches):
                print(f"Branch {branch.ID}: {self.flux_info[i]}, l={branch.length():.3g}")
        
        if self.is_backward:
            return self.flux_info.copy()
        if is_out_freefem_returned:
            return out_freefem
        
class FreeFEM_ThinFingers_Boundary(FreeFEM_ThinFingers):
    """A PDE solver based on the finite element method implemented in FreeFEM [Ref2]_.
    Solves PDE, computes a1a2a3 coefficients and forges it into tip trajectory with the streamline algorithm [Ref1]_.
    Extension for co-evolving boundary case.
    
    Attributes
    ----------
    equation, eta, ds, is_script_saved inherited from FreeFEM

    flux_info, bifurcation_type, bifurcation_thresh, 
        bifurcation_angle, inflow_thresh, distance_from_bif_thresh inherited from FreeFEM_ThinFingers

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
            inflow_thresh=0.05,
            distance_from_bif_thresh=None,
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
        inflow_thresh : float, default 0.05
        distance_from_bif_thresh=None,

        Returns
        -------
        None.

        """
        super().__init__(network, equation, eta, ds, is_script_saved, \
                         bifurcation_type, bifurcation_thresh, bifurcation_angle, \
                         inflow_thresh, distance_from_bif_thresh)
        
        add_after = lambda text, after_what, add_what: text.replace(after_what, after_what+add_what)
        
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

        self._script_init = """load "distance"\n""" + self._script_init

        self._script_adaptmesh = self._script_adaptmesh.replace("\n// Solving the problem", \
                                          script_distance+"\n// Solving the problem")
        self._script_adaptmesh = self._script_adaptmesh.replace("nvAroundTips.min < 250", "nvAroundTips.min < 100")
        self._script_adaptmesh = self._script_adaptmesh.replace("adaptCounter<=3", "adaptCounter<=2")
        # self._script_adaptmesh = self._script_adaptmesh.replace("Th = adaptmesh(Th,1,", "// Th = adaptmesh(Th,1,")
        self._script_adaptmesh = add_after(self._script_adaptmesh, "keepbackvertices=1",",requirededges=reqEdgs")
        self._script_adaptmesh = add_after( self._script_adaptmesh, \
                                            "// counting cells around the tips",
                                            "\nint[int] reqEdgs=[{DIRICHLET_1}];".format(DIRICHLET_1=DIRICHLET_1) )
        self._script_adaptmesh = add_after(self._script_adaptmesh, \
                                            "u=u;", "distExp=distExp;")
        self._script_adaptmesh = add_after(self._script_adaptmesh, \
                                            "*R,nbTips)", ", distExp")
    
        script_flux_rim = textwrap.dedent("""
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
            Vh dxu,dyu;
            dxu=dx(u);
            dyu=dy(u);

            // Deteremining the flux coming to the tip
            // More on reading the field values in specific points:
            // https://www.ljll.math.upmc.fr/pipermail/freefempp/2013-July/002798.html
            // https://ljll.math.upmc.fr/pipermail/freefempp/2009/000337.html

            int ndof=countNvOnRim(Th, {DIRICHLET_1}), n=0;
            real[int] xs(ndof), ys(ndof), fluxes(ndof); // angles with X axis
            int1d(Th, {DIRICHLET_1}, qfe=qf1pElump)( (xs(n++)=x)*1.
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
        self._script_tip_integration = self._script_tip_integration + script_flux_rim

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
            Network around which the field will be calculated.

        Returns
        -------
        None.

        """

        out_freefem = super().solve_PDE(network, is_out_freefem_returned=True)

        rim_xs = self.array_from_string(out_freefem.stdout,"xs")
        rim_ys = self.array_from_string(out_freefem.stdout,"ys")
        rim_fluxes = self.array_from_string(out_freefem.stdout,"fluxes")
        rim_xy_flux = np.stack((rim_xs, rim_ys, rim_fluxes)).T

        return rim_xy_flux               