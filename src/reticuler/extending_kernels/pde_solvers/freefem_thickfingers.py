import scipy
import textwrap
import numpy as np

import shapely
from shapely.ops import linemerge 
from shapely.geometry import LineString, MultiLineString, LinearRing, Polygon

from reticuler.utilities.misc import LEFT_WALL_PBC, RIGHT_WALL_PBC, DIRICHLET_1, DIRICHLET_0, NEUMANN_0, NEUMANN_1, DIRICHLET_0_GLOB_FLUX
from reticuler.utilities.misc import rotation_matrix

from reticuler.extending_kernels.pde_solvers.freefem import FreeFEM

class FreeFEM_ThickFingers(FreeFEM):
    """A PDE solver based on the finite element method implemented in FreeFEM [Ref2]_.
    Prepares thick fingers, solves PDE and computes fluxes at the tips and maximal flux direction [Ref3]_.

    Attributes
    ----------
    equation, eta, ds, is_script_saved inherited from FreeFEM

    flux_info : array
        A 2-n array with total flux and angle of highest flux direction 
        for each tip in the network.

    finger_width : float, default 0.02
        The width of the fingers.
    mobility_ratio : float, default 1e4
        Diffusive case (equation=0, 1):
            Mobility ratio between the inside and outside of the fingers.
            mobility_outside = 1, mobilty_inside = ``mobility_ratio``
        Elastic case (equation=2):
            Young's modulus of the canals (with E=100 for the tissue).


    References
    ----------
    .. [Ref2] https://freefem.org/
    .. [Ref3] "Breakthrough-induced loop formation in evolving transport networks",
        S. Żukowski, A. J. M. Cornelissen, F. Osselin, S. Douady, and P. Szymczak, PNAS 121, e2401200121 (2024).
        https://doi.org/10.1073/pnas.2401200121

    """

    def __init__(
            self, 
            network,
            equation=0,
            eta=1.0,
            ds=0.01,
            is_script_saved=False,
            finger_width=0.02, 
            mobility_ratio=1e4,
        ):
        """Initialize FreeFEM_ThickFingers.

        Parameters
        ----------
        network : Network
        finger_width : float, default 0.02
        mobility_ratio : float, default 1e4
        equation : int, default 0
        eta : float, default 1.0
        ds : float, default 0.01
        is_script_saved : bool, default False

        Returns
        -------
        None.

        """
        super().__init__(equation, eta, ds, is_script_saved)
        
        self.finger_width = finger_width
        self.mobility_ratio = mobility_ratio
        
        # parts of the script
        DIRICHLET_0_GLOB_FLUX_script = ""
        if (network.box.boundary_conditions==DIRICHLET_0_GLOB_FLUX).any(): 
            DIRICHLET_0_GLOB_FLUX_script = f"""
            // Normalize "u" if DIRICHLET_0_GLOB_FLUX BC
            real globFlux=int1d(Th, {DIRICHLET_0_GLOB_FLUX})( abs([dxu,dyu]'*[N.x,N.y])*mobility );
            u=u/globFlux;
            // Recalculate gradients
            dxu=dx(u);
            dyu=dy(u);
            """
        
        self.pbc = "" if network.box.boundary_conditions[0]!=2 else ", periodic=PBC"
        
        self._script_init = textwrap.dedent(
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
        box_ring, _, _, _, _ = \
            self.fingers_and_box_contours(network)
        self._script_border_box, self._script_inside_buildmesh_box = \
            self.prepare_script_box(network, box_ring)
        
        self._script_regions = textwrap.dedent(
            """
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // DEFINING REGIONS
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            """
            )
        
        self._script_problem = textwrap.dedent(
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
                                +on({DIRICHLET_0_GLOB_FLUX},u=0) // constant flux (global)
                                +on({DIRICHLET_0},u=0) // constant field
                                +on({DIRICHLET_1},u=1);
            """
        ).format(pbc=self.pbc, NEUMANN_1=NEUMANN_1, DIRICHLET_0_GLOB_FLUX=DIRICHLET_0_GLOB_FLUX, DIRICHLET_1=DIRICHLET_1, DIRICHLET_0=DIRICHLET_0)

        if self.equation==1:
            self._script_problem = self._script_problem.replace(
                                            "// -int2d(Th)(v) // rain in domain", 
                                            "-int2d(Th)(v) // rain in domain")

        self._script_adaptmesh = textwrap.dedent(
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
            cout << "Problem solved." << endl;

            real adaptTime=clock() - adaptTime0;
            
            plot(Th, wait=true);
            plot(u, wait=true, fill=true, value=true);
            """
        ).format(pbc=self.pbc)

        self._script_tip_integration = textwrap.dedent(
            """
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // CALCULATE FLUXES AND EXPORT RESULTS
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            
            // Calculating gradient
            dxu=dx(u);
            dyu=dy(u);
            // du=(dxu^2+dyu^2)^0.5;
            // plot(du, wait=true, fill=true);
            {DIRICHLET_0_GLOB_FLUX_script}            
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
                                                +(fluxes(n)=abs([dxu,dyu]'*[N.x,N.y]))*1.);
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
            """.format(DIRICHLET_0_GLOB_FLUX_script=DIRICHLET_0_GLOB_FLUX_script)
        )

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
    
    def fingers_and_box_contours(self, network):
        """ Prepares contours of the thickened tree using shapely library. """
        pts = []
        pts_in = [] # points inside subdomains with higher mobility
        tips_all = []; tips_active = [];
        for branch in network.branches:
            
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
        lines_to_merge = [] 
        br_diff_tree = box_ring.difference(thick_tree)
        br_intersec_tree = box_ring.intersection(thick_tree)
        lines_to_merge = [g.simplify(tolerance=1e-4) for g in br_diff_tree.geoms] + \
                            [g.simplify(tolerance=1e-4) for g in br_intersec_tree.geoms]
        box_ring_simp = linemerge(lines_to_merge)

        box_polygon = Polygon(box_ring)
        box_ring = linemerge( [*box_ring.difference(thick_tree).geoms,
                          *box_ring.intersection(thick_tree).geoms])
        thick_tree = box_polygon.intersection(thick_tree)
        
        # polygons to contours_tree
        polygons = [thick_tree] if thick_tree.geom_type=="Polygon" else thick_tree.geoms
        contours_tree = []
        for poly in polygons:
            pts_in.append(poly.representative_point().coords[0])
            poly = shapely.geometry.polygon.orient(poly) # now, exterior is ccw, but interiors are cw
            
            # exteriors
            # poly_exterior = poly.exterior.simplify(tolerance=1e-4)
            # poly_exterior = poly_exterior.difference(box_ring)
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
                mask = np.convolve(mask, [1,1], mode='valid')>1
                if mask.any():
                    contours_tree_bc[-1] = ~mask*888 + mask*b_label
        
        return box_ring_simp, contours_tree, contours_tree_bc, tips_active, pts_in

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
        border_nodes_inds2 = np.where(super().inNd(box_ring_pts,border_nodes))[0] # ?assumes that the points are ordered (if not we need to use a solution like for the seeds)
        boundary_conditions = np.ones(len(connections_to_add), dtype=int)
        boundary_conditions[:border_nodes_inds2[1]] = network.box.boundary_conditions[0]
        bcs0 = network.box.boundary_conditions[1:][border_nodes_mask]
        for i, ind in enumerate(border_nodes_inds2[:-1]):
            boundary_conditions[ind:border_nodes_inds2[i+1]] = bcs0[i]    
        boundary_conditions[border_nodes_inds2[-1]:] = bcs0[-1] 
        # points_to_plot = box_ring_pts[connections_to_add]
        # for i, pts in enumerate(points_to_plot):
        #     plt.plot(*pts.T, '.-', ms=1, lw=5, \
        #     color=f"{boundary_conditions[i]/6}")
        # for p in network.box.points[1:][border_nodes_mask]:
        #     plt.plot(*p, '.',ms=20, c='r')
        
        border_box, inside_buildmesh_box = \
            super().prepare_script_box( 
                                        np.column_stack((connections_to_add, boundary_conditions)), \
                                        box_ring_pts, \
                                        points_per_unit_len=0.5)  
        
        return border_box, inside_buildmesh_box

    def prepare_script_network(self, network, contours_tree, contours_tree_bc, tips):
        """Return parts of the FreeFEM script with the geometry of the ``network``."""
        # contours_tree to border
        border_contour = ""
        inside_buildmesh = ""
        for i, points in enumerate(contours_tree):
            # points = np.flip(points0, axis=0)
            border_contour, inside_buildmesh = \
                super().prepare_contour_list(border_contour, inside_buildmesh, i, points, \
                                    label=contours_tree_bc[i], border_name="contour" )
        
        inside_buildmesh = self._script_inside_buildmesh_box + inside_buildmesh[:-2]
        
        buildmesh = (
            textwrap.dedent(
                """
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // BUILDING MESH
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            """
            )
            + self._script_border_box
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
                tip_labels=super().arr2str(tips[:,0]),
                x=super().arr2str(tips[:,1]),
                y=super().arr2str(tips[:,2]),
            )
        )

        return buildmesh, tip_information    

    def prepare_script(self, network):
        """Return a full FreeFEM script with the ``network`` geometry."""
        # contours based on the thickened tree
        box_ring, contours_tree, contours_tree_bc, tips, points_in = \
            self.fingers_and_box_contours(network)
           
        self._script_border_box, self._script_inside_buildmesh_box = \
            self.prepare_script_box(network, box_ring)

        buildmesh, tip_information = self.prepare_script_network(network, \
                                            contours_tree, contours_tree_bc, tips)
        
        # Regions 
        regions = ""
        script_regions = self._script_regions
        for i, p in enumerate(points_in):
            script_regions = script_regions + \
                textwrap.dedent("""\nint indRegion{i} = Th({x}, {y}).region;""".format(
                    i=i, x=p[0], y=p[1] ))
            regions = regions + "region==indRegion{i} || ".format(i=i)
        
        #-> Diffusive specific
        script_regions = script_regions + textwrap.dedent(
            """
            fespace Vh0(Th, P0{pbc});
            Vh0 mobility = {mobilityOutside}*!({regions}) + {mobilityInside}*({regions});
            plot(mobility, wait=true, cmm="mobility", fill=true, value=true);
            """.format(mobilityOutside=1, mobilityInside=self.mobility_ratio, 
                        regions = regions[:-4], pbc=self.pbc)
            )
        #<-
        
        # whole script
        script = self._script_init + buildmesh + tip_information + script_regions
        #-> Diffusive specific
        script = script + self._script_problem
        #<-
        script = script + self._script_adaptmesh + self._script_tip_integration

        return script
    
    
    def solve_PDE(self, network):
        """Solve the PDE for the field around the network.

        Prepare a FreeFEM script, export it to a temporary file and run.
        Then, import the results to ``self.flux_info``.

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
            print("\n\n\nTrying again...")
            lookfor = "boxN(0:1023)=["
            ind = script.find(lookfor) + len(lookfor)
            ind2 = ind + script[ind:].find("];")
            boxN = np.fromstring(script[ind:ind2], sep=",", dtype=int)
            boxN_1 = np.array2string(boxN*2, separator=',', max_line_width=1000)[1:-1]
            script_perturbed = script.replace(script[ind:ind2],boxN_1)
            out_freefem = super().run_freefem(script_perturbed)

        # flux_info calculated in FreeFem:
        # flux_info = np.fromstring(out_freefem.stdout[out_freefem.stdout.find(b"kopytko")+7:], sep=",")
        # self.flux_info = flux_info.reshape(len(flux_info) // 2, 2)
        
        # determining flux_info 
        angles=[]; fluxes=[]; 
        self.flux_info = np.zeros((len(network.active_branches), 2))
        for i, branch in enumerate(network.active_branches):
            tip_label = 1000+branch.ID
            angles.append(self.array_from_string(out_freefem.stdout, f"angles{tip_label}"))
            fluxes.append(self.array_from_string(out_freefem.stdout, f"fluxes{tip_label}"))
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
            self.flux_info[i,0] = self.array_from_string(out_freefem.stdout, f"tot_flux{tip_label}")
            # Highest gradiend direction (angle with X axis)
            ang_max = xx[np.argmax(smoothed)]
            self.flux_info[i,1] = (ang_max + np.pi) % (2*np.pi) - np.pi
            
        with np.printoptions(formatter={"float": "{:.6g}".format}):
            print("flux_info: ", self.flux_info[...,0])
            print("angles: ", self.flux_info[...,1]*180/np.pi)

class FreeFEM_ThickFingers_Elasticity(FreeFEM_ThickFingers):
    """A PDE solver based on the finite element method implemented in FreeFEM [Ref2]_.
    Prepares thick fingers, solves PDE and computes fluxes at the tips and maximal flux direction [Ref3]_.
    Extension for the elastic case.

    Attributes
    ----------
    equation, eta, ds, is_script_saved inherited from FreeFEM

    flux_info, finger_width inherited from FreeFEM_ThickFingers
    
    youngs_modulus_inside : float, default 10
        Young's modulus inside the network.
    youngs_modulus_outside : float, default 100
        Young's modulus outside of the network.
    poissons_ratio_inside : float, default 0.3
        Poisson's inside the network.
    poissons_ratio_outside : float, default 0.49
        Poisson's outside of the network.
        
    References
    ----------
    .. [Ref2] https://freefem.org/
    .. [Ref3] "Breakthrough-induced loop formation in evolving transport networks",
        S. Żukowski, A. J. M. Cornelissen, F. Osselin, S. Douady, and P. Szymczak, PNAS 121, e2401200121 (2024).
        https://doi.org/10.1073/pnas.2401200121

    """

    def __init__(
            self, 
            network,
            equation=2,
            eta=1.0,
            ds=0.01,
            is_script_saved=False,
            finger_width=0.02, 
            youngs_modulus_inside=10,
            youngs_modulus_outside=100,
            poissons_ratio_inside=0.3,
            poissons_ratio_outside=0.49
        ):
        """Initialize FreeFEM_ThickFingers.

        Parameters
        ----------
        network : Network
        equation : int, default 2
        eta : float, default 1.0
        ds : float, default 0.01
        is_script_saved : bool, default False

        finger_width : float, default 0.02
        youngs_modulus_inside : float, default 10
        youngs_modulus_outside : float, default 100
        poissons_ratio_inside : float, default 0.3
        poissons_ratio_outside : float, default 0.49        

        Returns
        -------
        None.

        """
        super().__init__(network, equation, eta, ds, is_script_saved, finger_width)

        self.youngs_modulus_inside = youngs_modulus_inside
        self.youngs_modulus_outside = youngs_modulus_outside
        self.poissons_ratio_inside = poissons_ratio_inside
        self.poissons_ratio_outside = poissons_ratio_outside

        self._script_problem = textwrap.dedent(
            """
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // DEFINING PROBLEM AND EQUATION TO SOLVE
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            
            fespace Displacement(Th,P2);
            fespace Stress(Th, P2);

            Displacement ur, uq, vr, vq;
            Stress sigmarr, sigmaqq, sigmarq, sigmavM;
            Displacement r, q, cosq, sinq;
            r=sqrt((x-{radius})^2+(y-{radius})^2);q=atan2(y-{radius},x-{radius}); 
            cosq=(x-{radius})/r; sinq=(y-{radius})/r;

            macro dr(u) (dx(u)*cosq+dy(u)*sinq) // EOM (end of macro)
            macro dq(u) ((dy(u)*cosq-dx(u)*sinq)*r) // EOM
            // macro for strain
            macro strain(ur,uq)
                [
                    dr(ur),
                    (1/r*dq(ur)+dr(uq)-uq/r)/2,
                    (1/r*dq(ur)+dr(uq)-uq/r)/2,
                    ur/r+1/r*dq(uq)
                ]//eps_rr, eps_rq , eps_qr , eps_qq
            // macro for stress 
            macro stress(ur,uq)
                [
                    E/(1-nu^2)*(strain(ur,uq)[0]+nu*strain(ur,uq)[3]),
                    E/(1+nu)*strain(ur,uq)[1],
                    E/(1+nu)*strain(ur,uq)[2],
                    E/(1-nu^2)*(strain(ur,uq)[3]+nu*strain(ur,uq)[0])
                ] //stress s_rr, s_rq, s_qr, s_qq
                
            macro vonMises(sigmarr,sigmaqq,sigmarq) ( sqrt(sigmarr^2+sigmaqq^2-sigmarr*sigmaqq+3*sigmarq^2) ) // EOM von Mises stress
                
            //	Equation to solve
            problem	Elasticity([ur,uq],[vr,vq]) = 
                int2d(Th)(stress(ur,uq)'*strain(vr,vq)) 
                + on({NEUMANN_1}, ur={radial_deformation})
                + on({NEUMANN_0}, uq=0)
                ;
            """
        )

        self._script_adaptmesh = self._script_adaptmesh.replace("potential;", \
                                "Elasticity;\n    sigmavM = vonMises(stress(ur,uq)[0],stress(ur,uq)[3],stress(ur,uq)[1]);")
        self._script_adaptmesh = self._script_adaptmesh.replace("u/u[]", "sigmavM/sigmavM[]")
        self._script_adaptmesh = self._script_adaptmesh.replace("u=u;", "ur=ur; uq=uq;")
        self._script_adaptmesh = self._script_adaptmesh.replace("mobility=mobility;", "E=E; nu=nu;")
        self._script_adaptmesh = self._script_adaptmesh.replace("plot(u, wait=true, fill=true, value=true);", \
                                    textwrap.dedent("""
                                        // Stresses 
                                        sigmarr=stress(ur,uq)[0];sigmaqq=stress(ur,uq)[3];sigmarq=stress(ur,uq)[1];
                                        sigmavM = vonMises(stress(ur,uq)[0],stress(ur,uq)[3],stress(ur,uq)[1]);

                                        plot(sigmarr,fill=1, cmm="Stress sigmarr",wait=1,value=true);
                                        plot(sigmaqq,fill=1, cmm="Stress sigmaqq",wait=1,value=true);
                                        plot(sigmavM,fill=1, cmm="von Misses stress",wait=1,value=true,dim=2);

                                        // plot on the deformed surface
                                        // Cartesian displacements
                                        // Displacement ux, uy;
                                        // ux = cosq*ur - sinq*uq;
                                        // uy = sinq*ur + cosq*uq;
                                        // plot([ux,uy],coef=10,cmm="Displacement field",wait=1,value=true);

                                        // real ampfactor = 1;
                                        // mesh Th2=movemesh(Th,[x+ampfactor*ux,y+ampfactor*uy]);
                                        // plot(Th,Th2,cmm="Deformed configuration/r",wait=1);
                                        """)                          
                                )
        
        self._script_tip_integration = self._script_tip_integration.replace(
                                        textwrap.dedent("""
                                        // Calculating gradient
                                        dxu=dx(u);
                                        dyu=dy(u);
                                        // du=(dxu^2+dyu^2)^0.5;
                                        // plot(du, wait=true, fill=true);"""), \
                                        ""
                                    )
        self._script_tip_integration = self._script_tip_integration.replace("[dxu,dyu]'*[N.x,N.y]", "sigmavM")

    def prepare_script(self, network):
        """Return a full FreeFEM script with the ``network`` geometry."""
        # contours based on the thickened tree
        box_ring, contours_tree, contours_tree_bc, tips, points_in = \
            self.fingers_and_box_contours(network)
           
        self._script_border_box, self._script_inside_buildmesh_box = \
            super().prepare_script_box(network, box_ring)

        buildmesh, tip_information = self.prepare_script_network(network, \
                                            contours_tree, contours_tree_bc, tips)
        
        # Regions 
        regions = ""
        script_regions = self._script_regions
        for i, p in enumerate(points_in):
            script_regions = script_regions + \
                textwrap.dedent("""\nint indRegion{i} = Th({x}, {y}).region;""".format(
                    i=i, x=p[0], y=p[1] ))
            regions = regions + "region==indRegion{i} || ".format(i=i)
        
        #-> Elastic specific
        script_regions = script_regions + textwrap.dedent(
            """
            fespace Vh0(Th, P0{pbc});
            Vh0 E = {EOutside}*!({regions}) + {EInside}*({regions});
            Vh0 nu = {nuOutside}*!({regions}) + {nuInside}*({regions});
            plot(E, wait=true, cmm="E", fill=true, value=true);
            plot(nu, wait=true, cmm="nu", fill=true, value=true);
            """.format(EOutside=self.youngs_modulus_outside, EInside=self.youngs_modulus_inside,
                        nuOutside=self.poissons_ratio_outside, nuInside=self.poissons_ratio_inside,
                            regions = regions[:-4], pbc=self.pbc)
            )
        #<-

        # Whole script
        script = self._script_init + buildmesh + tip_information + script_regions
        #-> Elastic specific
        radius = (network.box.points[:,0].min()+network.box.points[:,0].max())/2
        script = script + self._script_problem.format(NEUMANN_1=NEUMANN_1,NEUMANN_0=NEUMANN_0, \
                                                        radius=radius, radial_deformation=-0.04*radius)
        #<-
        script = script + self._script_adaptmesh + self._script_tip_integration

        return script