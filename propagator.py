class VelocityVerletPropagator:


    def __init__(self, state):
        self.state = state

    def run(self):

        state = self.state

        if state.current_time > state.end_time:
            raise SystemExit("Nothing to be done")

        while (state.current_time <= state.end_time):
            self.update_coordinates(state)
            self.compute_gradient(state)
            self.update_velocities(state)
            self.rescale_velocities(state)
            update_state(state)
        return state



class BornOppenheimer:

    needed_properties = ["energy","gradient"]

    def __init__(self, spp, nstates):
        self.nstates = nstates
        self.spp = spp

    def get_gradient(self, crd):
        result = self.spp.request(crd, ['energy', 'gradient'])
        return result['energy'], result['gradient']

    def rescale_velocity(self, state):
        pass


class SurfaceHopping(BornOppenheimer):

    needed_properties = ["energy", "gradient", "coupling"]

    def __init__(self, spp, nstates):
        super().__init__(spp, nstates)
        self.nstates = nstates

    def rescale_velocity(self, state):
        self._electronic_propagation(state)
        hopped = self.surface_hopping(state)
        if hopped is True:
            self._rescale_velocity(state)
        ...
