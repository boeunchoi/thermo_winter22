import numpy as np
import tkinter as tk           # simple gui package for python


class particle():
    def __init__(self, size, pid, init_v=5, rad=3):
        """Initialise the particles

        Parameters
        ----------
        size : int
            Size of the box
        pid : int
            Unique particle ID
        init_v : int, optional
            Initial velocity for particles, by default 5
        rad : int, optional
            Radius of each particle, by default 3
        """
        # choose random x and y positions within the grid (padded by radius of particles)
        self.x = np.random.uniform(0 + rad, size - rad)
        self.y = np.random.uniform(0 + rad, size - rad)

        # set random velocities for each particle (randomly distributed between x and y speed)
        self.vx = np.random.uniform(0, init_v) * np.random.choice([-1, 1])
        self.vy = np.sqrt(init_v**2 - self.vx**2) * np.random.choice([-1, 1])

        # set the radius of the particle
        self.rad = rad

        # assign a particle id to each particle
        self.pid = pid

    def update_x(self, val):
        self.x = val

    def update_y(self, val):
        self.y = val

    def update_vx(self, val):
        self.vx = val

    def update_vy(self, val):
        self.vy = val


class Simulation():  # this is where we will make them interact
    def __init__(self, N, E, size, rad, delay=20):
        """Simulation class initialisation. This class handles the entire particle
        in a box thing.

        Parameters
        ----------
        N : `int`
            Total number of particles
        E : `int`
            Kinetic energy to start with
        size : `int`
            Size of the box
        rad : `int`
            Radius of the particles
        delay : `int`
            Delay in milliseconds between showing/running timesteps
        """
        self.N = N
        self.E = E
        self.size = size
        self.rad = rad
        self.delay = delay

        # initialise N particle classes
        self.particles = [particle(size=size, pid=i, init_v=E, rad=rad) for i in range(N)]

        self.canvas = None
        self.root = None
        self.particle_handles = {}

        self._init_visualization()
        self.root.update()

    def _init_visualization(self):
        # start the visualisation box
        self.root = tk.Tk()
        self.root.title("Particles in a Box!")

        # create a canvas with the right size
        self.canvas = tk.Canvas(self.root, width=self.size, height=self.size)
        self.canvas.pack()

        # add a close button
        self.button = tk.Button(self.root, text='Close', command=self._quit_visualisation)
        self.button.place(x=self.size, y=10, anchor="e")

        self.timestep_message = self.canvas.create_text(self.size // 2, 10, text="Timestep = 0")

        # add all of the particles
        for p in self.particles:
            self.particle_handles[p.pid] = self._draw_particle(p)

        # update this all on the canvas
        self.root.update()

    def _quit_visualisation(self):
        self.root.destroy()

    def _draw_particle(self, particle):
        """Draw a circle on the canvas corresponding to particle

        Returns the handle of the tkinter circle element"""
        x0 = particle.x - particle.rad
        y0 = particle.y - particle.rad
        x1 = particle.x + particle.rad
        y1 = particle.y + particle.rad
        return self.canvas.create_oval(x0, y0, x1, y1, fill='black', outline='black')

    def _move_particle(self, particle):
        xx = particle.x + particle.vx
        yy = particle.y + particle.vy
        particle.update_x(xx)
        particle.update_y(yy)
        self.canvas.move(self.particle_handles[particle.pid], particle.vx, particle.vy)

    def resolve_particle_collisions(self):
        raise NotImplementedError

    def resolve_wall_collisions(self):
        # check whether each particle hits the wall
        # for each collider reflect its velocity (account for ones that hit both walls)
        raise NotImplementedError

    def run_simulation(self, steps=1000):
        for i in range(steps):
            # 1. update all particle positions based on current speeds
            for particle in self.particles:
                self._move_particle(particle)

            # 2. resolve whether any hit the wall and reflect them
            # self.resolve_wall_collisions()

            # 3. resolve any particle collisions and transfer momentum
            # self.resolve_particle_collisions()

            # update visualization with a delay
            self.root.after(self.delay, self.root.update())

            # change the timestep message as well
            self.canvas.itemconfig(self.timestep_message, text="Timestep = {}".format(i))

        self.root.mainloop()

    def get_velocities(self):
        raise NotImplementedError
