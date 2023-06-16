"""Solvers for PDEs describing the field around the network.

Classes:
    FreeFEM
    
"""

import numpy as np
import subprocess
import os.path
import os
from tempfile import NamedTemporaryFile
import textwrap


class FreeFEM:
    """PDE solver based on the finite element method implemented in FreeFEM [Ref2]_.

    Attributes
    ----------
    equation : int, default 0
        - 0: Laplace
        - 1: Poisson
    a1a2a3_coefficients : array
        An array of a1a2a3 coefficients for each tip in the network.


    References
    ----------
    .. [Ref2] https://freefem.org/

    """

    def __init__(self, equation=0):
        """Initialize FreeFEM.

        Parameters
        ----------
        equation : int, default 0

        Returns
        -------
        None.

        """
        self.equation = equation

        self.a1a2a3_coefficients = []

    def __prepare_script(self, network):
        """Return a FreeFEM script with `Network` geometry."""

        initialisation = textwrap.dedent(
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

        border_box = "\nreal buildTime0=clock();\n\n"
        inside_buildmesh = ""
        for i, triple in enumerate(network.box.connections_bc()):
            x0 = network.box.points[triple[0], 0]
            y0 = network.box.points[triple[0], 1]
            x1 = network.box.points[triple[1], 0]
            y1 = network.box.points[triple[1], 1]
            boundary_condition = triple[2]
            n_points = np.max((1, int(np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2) / 2)))

            border_box = (
                border_box
                + "border box{i}(t=0, 1){{x={x0}+t*({ax});y={y0}+t*({ay}); label={bc};}}\n".format(
                    i=i, x0=x0, ax=x1 - x0, y0=y0, ay=y1 - y0, bc=boundary_condition
                )
            )

            inside_buildmesh = inside_buildmesh + " box{i}({n}) +".format(
                i=i, n=n_points
            )

        border_network = ""
        for i, branch in enumerate(network.branches):
            for j, pair in enumerate(zip(branch.points, branch.points[1:])):
                x0 = pair[0][0]
                y0 = pair[0][1]
                x1 = pair[1][0]
                y1 = pair[1][1]

                border_network = (
                    border_network
                    + "border branch{i}connection{j}(t=0, 1){{x={x0}+t*({ax});y={y0}+t*({ay}); label=1;}}\n".format(
                        i=i, j=j, x0=x0, ax=x1 - x0, y0=y0, ay=y1 - y0
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
            + border_box
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
                + "\nX({j})={x};".format(j=i, x=branch.points[-1, 0])
                + "\nY({j})={y};".format(j=i, y=branch.points[-1, 1])
                + "\nangle({j})={angle};".format(
                    j=i, angle=np.pi / 2 - branch.tip_angle()
                )
            )  # angle with X axis
        tip_information = tip_information + "\n"

        problem_Laplace = textwrap.dedent(
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

        problem_Poisson = textwrap.dedent(
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

        adaptmesh = textwrap.dedent(
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

        integrate_a1a2a3 = textwrap.dedent(
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
            		*BaseVector(order,exp(-angle(i)*1i)*( (x-x0) + (y-y0)*1i) ) ) /
                    (int2d(Ph)(exp(-(sqrt((x-x0)^2 + (y-y0)^2)/R)^exponant)
            		*square(BaseVector(order,exp(-angle(i)*1i)*( (x-x0) + (y-y0)*1i) ) )));
            		
            		cout << a[order-1] << ",";
                    // cout << "a(" << order << ") = " << a[order-1] << endl;
                }}
            	// freefemOutput << Th.nv << " ";
            	// freefemOutput << Ph.nv << " ";
            	// freefemOutput << adaptCounter << " ";
            	
            	// cout << endl;
            }};
            
            // cout << endl << endl << "Building mesh took: " << buildTime; 
            // cout << endl << "First adapt took: " << firstAdaptTime; 
            // cout << endl << "First run took: " << firstRunTime; 
            // cout << endl << "Adaptation took: " << adaptTime; 
            // cout << endl << "Calculating coefficients took: " << clock()- coeffTime0;
            // cout << endl << "Total time: " << clock()-time0 << endl << endl;
            """
        )

        script = initialisation + buildmesh + tip_information
        if self.equation:
            script = script + problem_Poisson
        else:
            script = script + problem_Laplace
        script = script + adaptmesh + integrate_a1a2a3

        return script

    def __run_freefem_temp(self, script):
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

        ai_coeffs_flat = np.fromstring(result.stdout, sep=",")
        self.a1a2a3_coefficients = ai_coeffs_flat.reshape(len(ai_coeffs_flat) // 3, 3)
        # print(self.a1a2a3_coefficients)

        # close temporary files
        for tmp_file in temporary_files:
            tmp_file.close()
            os.unlink(tmp_file.name)

    def __run_freefem(self, script):
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
        print("stdout:", result.stdout.decode())
        print("stderr:", result.stderr.decode())
        if result.returncode:
            print("\nFreeFem++ failed.\n")

        ai_coeffs_flat = np.fromstring(result.stdout, sep=",")
        self.a1a2a3_coefficients = ai_coeffs_flat.reshape(len(ai_coeffs_flat) // 3, 3)
        print(self.a1a2a3_coefficients)

    def solve_PDE(self, network):
        """Solve the PDE for the field around the network.

        Prepare a FreeFEM script, export it to a temporary file and run.
        Then, import the a1a2a3 coefficients to ``self.a1a2a3_coefficients``.

        Parameters
        ----------
        network : object of class Network
            Network around which the field will be calculated.

        Returns
        -------
        None.

        """
        script = self.__prepare_script(network)
        self.__run_freefem_temp(script)
        # self.__run_freefem(script) # useful for debugging
        print('a1a2a3: ', self.a1a2a3_coefficients)