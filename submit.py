import os
from subprocess import run, CalledProcessError

class SubmitTrajectories:

    def trajectories(self):
        for traj in os.listdir("prop"):
            subfolder = os.path.join("prop",traj)
            try:
                run(['sbatch run.sh'], cwd=subfolder, check=True, shell=True)
            except KeyboardInterrupt or CalledProcessError:
                break
            print("Submitting", subfolder)            

if __name__=='__main__':
    all_traj = SubmitTrajectories()
    all_traj.trajectories()

