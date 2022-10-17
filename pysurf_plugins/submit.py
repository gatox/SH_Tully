import os
from subprocess import run, CalledProcessError

class SubmitTrajectories:

    def read_sampling(self):
        prop = open("sampling.inp", 'r+')
        for line in prop:
            if "model = Tully_1" in line:
                model = str(line.split()[2])
            else:
                model = "molecule"
        return model 

    def read_spp(self):
        prop = open("spp.inp", 'r+')
        for line in prop:
            if "use_db =" in line:
                use_db = str(line.split()[2])
        return use_db 

    def read_prop(self):
        prop = open("prop.inp", 'r+')
        for line in prop:
            if "method" in line:
                method = str(line.split()[2])
        return method 

    def trajectories(self):
        model = self.read_sampling()
        use_db = self.read_spp()
        method = self.read_prop()
        for traj in os.listdir("prop"):
            subfolder = os.path.join("prop",traj)
            try:
                if method == "LandauZener":
                    if use_db == "yes":
                        run(['sbatch db_lz_run.sh'], cwd=subfolder, check=True, shell=True)
                    else:
                        run(['sbatch lz_run.sh'], cwd=subfolder, check=True, shell=True)
                elif method == "Surface_Hopping":
                    if model == "Tully_1":
                        run(['sbatch model_fssh_run.sh'], cwd=subfolder, check=True, shell=True)
                    else:
                        run(['sbatch fssh_run.sh'], cwd=subfolder, check=True, shell=True)
            except KeyboardInterrupt or CalledProcessError:
                break
            print("Submitting", subfolder)            


if __name__=='__main__':
    all_traj = SubmitTrajectories()
    all_traj.trajectories()
