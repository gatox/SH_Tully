# SH_Tully
We share our own implementation of the trajectory surface hopping method based on Tully's approach using the velocity Verlet algorithm. This approach is often called fewest switches surface hopping (FSSH) because distributes the trajectories between the electronic states according to the computed state probabilities using a minimum number of hops[1].
This implementation works as a pluging of [PySurf](https://github.com/MFSJMenger/pysurf); a software framework for data science applications in computational chemistry[2].
## Requirements
Create a python enviroment (Python3.8+). Clone the [PySurf](https://github.com/MFSJMenger/pysurf) repository and its requirements. Clone the `pysurf_plugins` folder at home directory. 
## Settings
Go to `pysurf/pusurf` folder. Open `__init__.py` file, add the following line: `home = os.path.expanduser("~")` and change the following line: `user_plugins = os.path.join(base, "plugins")` to `user_plugins = os.path.join(home, "pysurf_plugins")`. 
## References
[1] Tully, J. C. Molecular dynamics with electronic transitions. J. Chem. Phys. 1990, 93,
1061–1071.

[2] Menger, M. F. S. J.; Ehrmaier, J.; Faraji, S. PySurf: A Framework for Database
Accelerated Direct Dynamics. J. Chem. Theory Comput. 2020, 16, 7681–7689.
