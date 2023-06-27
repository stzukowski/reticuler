"""Solvers for PDEs describing the field around the network.

Classes:
    FreeFEM
    
"""

import numpy as np
import subprocess
import os.path
import os
from platform import system
from tempfile import NamedTemporaryFile
import textwrap

from reticuler.system import LEFT_WALL_PBC, RIGHT_WALL_PBC, DIRICHLET

class FreeFEM:
    """PDE solver based on the finite element method implemented in FreeFEM [Ref2]_.

    Attributes
    ----------
    equation : int, default 0
        - 0: Laplace
        - 1: Poisson
    flux_info : array
        An array of a1a2a3 coefficients for each tip in the network.


    References
    ----------
    .. [Ref2] https://freefem.org/

    """

    def __init__(self, network, equation=0):
        """Initialize FreeFEM.

        Parameters
        ----------
        network : Network
        equation : int, default 0

        Returns
        -------
        None.

        """
        self.equation = equation
        self.flux_info = []
        
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
            func real tipfield( real[int] X, real[int] Y,real sigma,int nTips)
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
            
            problem potential(u,v,solver=sparsesolver)=
                     int2d(Th)(dx(u)*dx(v) + dy(u)*dy(v))
                         -int1d(Th,3)(v)  // constant flux
            			 // +on(3,u=50) // constant field
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
            
            problem potential(u,v,solver=sparsesolver)=
                     int2d(Th)(dx(u)*dx(v) + dy(u)*dy(v))
                         -int2d(Th)(v) 
            			 -int1d(Th,3)(v)  // constant flux
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
            		*BaseVector(order,exp(-angle(i)*1i)*( (x-x0) + (y-y0)*1i) ) ) /
                    (int2d(Ph)(exp(-(sqrt((x-x0)^2 + (y-y0)^2)/R)^exponant)
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

    def prepare_script_box(self, connections_bc, points, points_per_unit_len=0.5):
        """Return part of the FreeFEM script with the geometry of the box."""
        border_box = "\nreal buildTime0=clock();\n\n"
        inside_buildmesh_box = ""
        for i, triple in enumerate(connections_bc):
            x0 = points[triple[0], 0]
            y0 = points[triple[0], 1]
            x1 = points[triple[1], 0]
            y1 = points[triple[1], 1]
            boundary_condition = triple[2]
            n_points = np.max((1, int(points_per_unit_len*np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2))))

            border_box = (
                border_box
                + "border box{i}(t=0, 1){{x={x0:.12g}+t*({ax:.12g});y={y0:.12g}+t*({ay:.12g}); label={bc};}}\n".format(
                    i=i, x0=x0, ax=x1 - x0, y0=y0, ay=y1 - y0, bc=boundary_condition
                )
            )

            inside_buildmesh_box = inside_buildmesh_box + " box{i}({n}) +".format(
                i=i, n=n_points
            )
        return border_box, inside_buildmesh_box
    
    def __prepare_script(self, network):
        """Return a FreeFEM script with `network` geometry."""

        border_network = ""
        inside_buildmesh = self.__script_inside_buildmesh_box
        for i, branch in enumerate(network.branches):
            for j, pair in enumerate(zip(branch.points, branch.points[1:])):
                x0 = pair[0][0]
                y0 = pair[0][1]
                x1 = pair[1][0]
                y1 = pair[1][1]

                border_network = (
                    border_network
                    + "border branch{i}connection{j}(t=0, 1){{x={x0:.12g}+t*({ax:.12g});y={y0:.12g}+t*({ay:.12g}); label={bc};}}\n".format(
                        i=i, j=j, x0=x0, ax=x1 - x0, y0=y0, ay=y1 - y0, bc=DIRICHLET
                    )
                )

                inside_buildmesh = (
                    inside_buildmesh + " branch{i}connection{j}(1) +".format(i=i, j=j)
                )
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
            + "\nmesh Th = buildmesh({inside_buildmesh});\n".format(
                inside_buildmesh=inside_buildmesh
            )
            + "\nreal buildTime=clock() - buildTime0;\n// plot(Th, wait=true);\n"
        )

        tip_information = textwrap.dedent(
            """
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // TIP INFORMATION
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            
            int nbTips={n_tips}; 
            real[int] angle(nbTips);
            real[int] X(nbTips); 
            real[int] Y(nbTips);
            """.format(
                n_tips=len(network.active_branches)
            )
        )
        for i, branch in enumerate(network.active_branches):
            tip_information = (
                tip_information
                + "\nX({j})={x:.12g};".format(j=i, x=branch.points[-1, 0])
                + "\nY({j})={y:.12g};".format(j=i, y=branch.points[-1, 1])
                + "\nangle({j})={angle:.12g};".format(
                    j=i, angle=branch.tip_angle()
                )
            )  # angle with X axis
        tip_information = tip_information + "\n"

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
        if result.returncode:
            print("\nFreeFem++ failed.\n")
            print("stdout:", result.stdout.decode())
            print("stderr:", result.stderr.decode())

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
        # print("stdout:", result.stdout.decode())
        # print("stderr:", result.stderr.decode())
        if result.returncode:
            print("\nFreeFem++ failed.\n")

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
        script = self.__prepare_script(network)
        if system() == 'Windows':
            out_freefem = self.run_freefem(script) # useful for debugging
        else:
            out_freefem = self.run_freefem_temp(script)
        ai_coeffs_flat = np.fromstring(out_freefem.stdout[out_freefem.stdout.find(b'kopytko')+7:], sep=",")
        self.flux_info = ai_coeffs_flat.reshape(len(ai_coeffs_flat) // 3, 3)
        # with np.printoptions(formatter={'float': '{:.6e}'.format}):
            # print('a1a2a3: \n', self.flux_info)
            

class FreeFEM_ThickFingers:
    """PDE solver based on the finite element method implemented in FreeFEM [Ref2]_.

    Attributes
    ----------
    equation : int, default 0
        - 0: Laplace
        - 1: Poisson
    flux_info : array
        An array of a1a2a3 coefficients for each tip in the network.
    finger_width : float, default 0.02
        The width of the fingers.
    mobility_ratio : float, default 1e4
        Mobility ratio between inside and outside of the fingers.
        mobility_outside = 1, mobilty_inside = `mobility_ratio`


    References
    ----------
    .. [Ref2] https://freefem.org/

    """

    def __init__(self, network, equation=0, finger_width=0.02, mobility_ratio=1e4, n_adapt=2):
        """Initialize FreeFEM.

        Parameters
        ----------
        network : Network
        equation : int, default 0
        finger_width : float, default 0.02
        mobility_ratio : float, default 1e4

        Returns
        -------
        None.

        """
        self.equation = equation
        self.finger_width = finger_width
        self.mobility_ratio = mobility_ratio 
        self.n_adapt = n_adapt
        self.flux_info = []
         
        # parts of the script
        self.__script_init = textwrap.dedent(
            """
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // INITIALISATION
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            verbosity = 0;
            
            real time0=clock();
            
            // Adaptation around the tip
            func real tipfield( real X, real Y, real sigma)
            {
            real rr=((x-X)^2 + (y-Y)^2)^0.5;
            real err=0;
            if (rr>0.99*sigma & rr<1.01*sigma)
                err=1;
            return err;
            }
            """
        )
        
        self.__script_border_box, self.__script_inside_buildmesh_box = \
            self.prepare_script_box(network)
        
        self.__script_mobility = textwrap.dedent(
            """
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // DEFINING MOBILITY
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            
            int indRegionOut=Th({x_outside},{y_outside}).region;""").format(
                x_outside=network.box.points[1,0]*0.9,
                y_outside=network.box.points[1,1]*0.9
                )
        
        self.__script_problem_Laplace = textwrap.dedent(
            """
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // DEFINING PROBLEM AND equation TO SOLVE
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            
            fespace Vh(Th,P2, periodic=PBC);
            Vh u,v,dxu,dyu,du;
            
            problem potential(u,v,solver=sparsesolver)=
                     int2d(Th)(mobility*(dx(u)*dx(v) + dy(u)*dy(v)))
                                // -int1d(Th,3)(v)  // constant flux
                                +on(3,u=0) // constant field
                                // -int2d(Th)(v) // rain in domain
                                +on(1,u=1);
            """
        )

        self.__script_problem_Poisson = textwrap.dedent(
            """ 
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // DEFINING PROBLEM AND equation TO SOLVE
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            fespace Vh(Th,P2, periodic=PBC);
            Vh u,v,dxu,dyu,du;

            problem potential(u,v,solver=sparsesolver)=
                     int2d(Th)(mobility*(dx(u)*dx(v) + dy(u)*dy(v)))
                                -int2d(Th)(v) // Poissonian source term
                                // -int1d(Th,3)(v)  // constant flux
                                +on(3,u=0) // constant field
                                // -int2d(Th)(v) // rain in domain
                                +on(1,u=1);            
            """
        )

        self.__script_adaptmesh = textwrap.dedent(
            """
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // ADAPTING THE MESH AND SOLVING FOR THE FIELD
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////        

            // First adaptation
            real firstAdaptTime0=clock();
            // Th = adaptmesh(Th,5.*tipfield(X,Y,3.*R,nbTips),nbvx=500000,nbsmooth=100,iso=true);
            // Th = adaptmesh(Th,1,nbvx=500000,hmax=0.05,nbsmooth=100,iso=true,ratio=1.8,periodic=PBC);
            real firstAdaptTime=clock() - firstAdaptTime0;
            // plot(Th, wait=true);
            
            // Solving the problem for the first time
            real firstRunTime0=clock();
            potential;
            // cout<<"First solve completed."<<endl;
            real firstRunTime=clock() - firstRunTime0;
            // dxu=dx(u);
            // dyu=dy(u);
            // du=(dxu^2+dyu^2)^0.5;
            // plot(du, wait=true, fill=true);
            
            // Adaptation loop
            real adaptTime0=clock();
            // cout << endl << endl << "Adaptation..." << endl;
            real error=0.01;
            for(int i=0;i<{n_adapt};i++){{
            Th=adaptmesh(Th, u, err=error, nbvx=500000, periodic=PBC, verbosity=0, nbsmooth=100,iso=true,ratio=1.8); // Adapting mesh according to the first solution
            u=u;
            error=error/2;
            potential; // Solving one more time with adapted mesh
            }}
            // cout << "Problem solved." << endl;
            // plot(u, wait=true, fill=true);
            
            real adaptTime=clock() - adaptTime0;
            
            // Calculating gradient
            dxu=dx(u);
            dyu=dy(u);
            // du=(dxu^2+dyu^2)^0.5;
            // plot(du, wait=true, fill=true);
            """
        ).format(n_adapt=self.n_adapt)

        self.__script_tip_integration = textwrap.dedent(
            """
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // EXPORT RESULTS
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////        

            cout.precision(12);
            cout << "kopytko ";
            int ndof=0, n=0;
            real totGrad=0;
            real sumGrad=0, sumAng=0;
            int avgWindow = 2;
            real maxGrad=0, maxAngle=0;
            """
        )

    def prepare_script_box(self, network):
        """Return parts of the FreeFEM script with the geometry of the `network.box`."""
        
        points = network.box.points.copy()
        boundary_conditions = network.box.boundary_conditions.copy()
        for i, seed_ind in enumerate(network.box.seeds_connectivity[:,0]):
            point_seed = network.box.points[seed_ind]
            point_left = network.box.points[seed_ind-1]
            point_right = network.box.points[(seed_ind+1) % len(network.box.points)]
            v1 = point_left - point_seed
            v1 = point_seed + v1 / np.linalg.norm(v1) * self.finger_width/2
            v2 = point_right - point_seed
            v2 = point_seed + v2 / np.linalg.norm(v2) * self.finger_width/2
            
            points[seed_ind+i] = v1
            points = np.insert(points, seed_ind+i+1, v2, axis=0)
            boundary_conditions = np.insert(boundary_conditions, seed_ind+i+1, 1)
    
        connections = np.vstack(
                        [np.arange(len(points)), np.roll(
                            np.arange(len(points)), -1)]
                    ).T
        connections_bc = np.column_stack((connections, boundary_conditions))
        
        border_box, inside_buildmesh_box = \
            FreeFEM.prepare_script_box(self, connections_bc, points, points_per_unit_len=10)
        
        return border_box, inside_buildmesh_box

    
    def __prepare_script(self, network):
        """Return a full FreeFEM script with the `network` geometry."""
        pt_between = lambda pt1, pt2 : pt1 + (pt2-pt1)/np.linalg.norm(pt2-pt1)*self.finger_width/2
        
        tip_integration_template = textwrap.dedent(
            """
            ndof=0; n=0; maxGrad=0; maxAngle=pi/2;
            int1d(Th, {i}, qfe=qf1pE)( (ndof++)*1.);
            real[int] angles{i}(ndof), gradAbsNormal{i}(ndof), gradAbs{i}(ndof);
            int1d(Th, {i}, qfe=qf1pE)( (angles{i}(n++)=atan2(y-{y0}, x-{x0}))*1.
                                            +(gradAbs{i}(n)=(dxu^2+dyu^2)^0.5)*1.
                                            +(gradAbsNormal{i}(n)=abs(dxu*N.x+dyu*N.y))*1.);
            // cout << "angles{i} "<<angles{i}<<"angles{i}end "<<endl;
            // cout << "gradAbs{i} "<<gradAbs{i}<<"gradAbs{i}end "<<endl;
            // cout << "gradAbsNormal{i} "<<gradAbsNormal{i}<<"gradAbsNormal{i}end "<<endl;
            
            real[int] gradAbsNormalMvAvg{i}(ndof-avgWindow+1), anglesMvAvg{i}(ndof-avgWindow+1);
            for (int i=0; i<=(ndof-avgWindow); i++){{
                sumGrad=0;
                sumAng=0;
                for (int j=i; j<i+avgWindow; j++){{
                    sumGrad += gradAbsNormal{i}[j];
                    sumAng += angles{i}[j];
                }}
                	gradAbsNormalMvAvg{i}(i) = sumGrad / avgWindow;
                	anglesMvAvg{i}(i) = sumAng / avgWindow;
                	if (gradAbsNormalMvAvg{i}(i)>maxGrad){{
                    maxGrad=gradAbsNormalMvAvg{i}(i);
                    maxAngle=anglesMvAvg{i}(i);
                }}
            }}
            totGrad =  int1d(Th, {i})( abs([dxu,dyu]'*[N.x,N.y]) );
            cout << totGrad << "," << maxAngle << ",";
            """
            )
        
        
        inside_mobility = ""
        border_contour = ""
        inside_buildmesh = self.__script_inside_buildmesh_box
        script_mobility = self.__script_mobility
        script_tip_integration = self.__script_tip_integration
        for i, branch in enumerate(network.branches):
            if branch.ID in network.box.seeds_connectivity[:,1]:
                ind_seed = network.box.seeds_connectivity[network.box.seeds_connectivity[:,1]==branch.ID, 0]
                # box (boundary)
                box_pt_right = pt_between(network.box.points[ind_seed], \
                                      network.box.points[(ind_seed+1) % len(network.box.points)])
                box_pt_left = pt_between(network.box.points[ind_seed], \
                                      network.box.points[ind_seed-1])
                    

                script_mobility = script_mobility + \
                            textwrap.dedent("""\nint indRegion{i} = Th({x}, {y}).region;""".format(
                                i=branch.ID, \
                                x=branch.points[-1][0],
                                y=branch.points[-1][1]
                                ))
                inside_mobility = inside_mobility + "region==indRegion{i} || ".format(i=branch.ID)
            
            skeleton = branch.points
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
            
            if len(network.branch_connectivity)==0 or branch.ID not in network.branch_connectivity[:,0]:
                # if the branch is a dead end -> tip with a semi-circular cap
                angle_tip_right = angles_down[-1] + np.pi/2
                angle_tip_left = angle_tip_right + np.pi
                tip_right = skeleton[-1] + self.finger_width/2 * \
                                np.array([np.cos(angle_tip_right), np.sin(angle_tip_right)])
                tip_left = skeleton[-1] + self.finger_width/2 * \
                        np.array([np.cos(angle_tip_left), np.sin(angle_tip_left)])
                
                contour_right = np.vstack(( contour_right, tip_right ))
                contour_left = np.vstack(( tip_left, contour_left ))    
                
                # semi-circular cap
                tilt_shift = tip_right-skeleton[-1]
                tilt_ang = np.arctan2(tilt_shift[1], tilt_shift[0])
                border_contour = (
                    border_contour
                    + "border tip{i}(t=0, 1){{x={x0:.12g}+{f_w_half:.12g}*cos(t*pi+{phi0});y={y0:.12g}+{f_w_half:.12g}*sin(t*pi+{phi0}); label={bc};}}\n".format(
                        i=branch.ID, x0=skeleton[-1,0], y0=skeleton[-1,1], 
                        f_w_half=self.finger_width/2, bc=1000+branch.ID, phi0=tilt_ang
                    )
                )

                inside_buildmesh = (
                    inside_buildmesh + " tip{i}(101) +".format(i=branch.ID)
                )
                
                script_tip_integration = script_tip_integration + tip_integration_template.format( \
                                                i=1000+branch.ID, 
                                                x0=skeleton[-1,0], 
                                                y0=skeleton[-1,1])
        
            for k, points in enumerate([contour_right, contour_left]):
                # building mesh for contours on the right and left of the finger
                for j, pair in enumerate(zip(points, points[1:])):
                    x0 = pair[0][0]
                    y0 = pair[0][1]
                    x1 = pair[1][0]
                    y1 = pair[1][1]
    
                    border_contour = (
                        border_contour
                        + "border branch{i}connection{j}side{k}(t=0, 1){{x={x0:.12g}+t*({ax:.12g});y={y0:.12g}+t*({ay:.12g}); label=888;}}\n".format(
                            i=branch.ID, j=j, k=k, x0=x0, ax=x1 - x0, y0=y0, ay=y1 - y0
                        )
                    )
    
                    inside_buildmesh = (
                        inside_buildmesh + " branch{i}connection{j}side{k}(1) +".format(i=branch.ID, j=j, k=k)
                    )
        inside_buildmesh = inside_buildmesh[:-1]
        script_mobility = script_mobility + textwrap.dedent(
            """
            fespace Vh0(Th, P0, periodic=PBC);
            Vh0 mobility = {mobilityOutside}*(region==indRegionOut) + {mobilityInside}*({inside_mobility});
            """.format(mobilityOutside=1, mobilityInside=self.mobility_ratio, inside_mobility = inside_mobility[:-4])
            )

        buildmesh = (
            textwrap.dedent(
                """
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // BUILDING MESH
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            """
            )
            + self.__script_border_box
            + border_contour
            + "func PBC=[[{left},y],[{right},y]];".format( left=LEFT_WALL_PBC, right=RIGHT_WALL_PBC)
            + "\nmesh Th = buildmesh({inside_buildmesh}, fixedborder=true);\n".format(
                inside_buildmesh=inside_buildmesh
            )
            + "\nreal buildTime=clock() - buildTime0;\n// plot(Th, wait=true);\n"
        )

        script = self.__script_init + buildmesh + script_mobility
        if self.equation:
            script = script + self.__script_problem_Poisson
        else:
            script = script + self.__script_problem_Laplace
        script = script + self.__script_adaptmesh + script_tip_integration

        return script
    
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
        script = self.__prepare_script(network)
        if system() == "Windows":
            out_freefem = FreeFEM.run_freefem(self, script) # useful for debugging
        else:
            out_freefem = FreeFEM.run_freefem_temp(self, script)
            
        flux_info = np.fromstring(out_freefem.stdout[out_freefem.stdout.find(b'kopytko')+7:], sep=",")
        self.flux_info = flux_info.reshape(len(flux_info) // 2, 2)
        # return s_o
    
    
        # ai_coeffs_flat = np.fromstring(out_freefem.stdout[out_freefem.stdout.find(b'kopytko')+7:], sep=",")
        # self.flux_info = ai_coeffs_flat.reshape(len(ai_coeffs_flat) // 3, 3)
        # with np.printoptions(formatter={'float': '{:.6e}'.format}):
            # print('a1a2a3: \n', self.flux_info)
