#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tkinter as tk           # simple gui package for python


# In[27]:


class particle():
    def __init__(self, size, pid, T, mass, rad):
        """Initialise the particles

        Parameters
        ----------
        size : int
            1/2 Size of the box (rectangle)
            
        pid : int
            Unique particle ID
            
        T : `float`
            Temperature of the system [K]
        
        mass : float
            Mass of each particle
        
        rad : float, optional
            Radius of each particle
        """
        
        kB = 1.38e-16     # Boltzmann const. [cm^2 g s^-2 K^-1]
        
        #Initial velocity for particles, determined by the temperature of the box
        init_v = np.sqrt(3. * kB * T / mass)   
        
        # Frame origin: center of the box
        self.size = size
        
        # Assign a particle ID to each particle
        self.pid = pid

        # Set the radius of the particle
        self.rad = rad
    
        # Set the mass of the particle
        self.mass = mass        
    
        # Choose random x and y positions within the grid (padded by radius of particles)
        self.x = 2. * np.random.uniform(0 + rad, size - rad) - size
        self.y = 2. * np.random.uniform(0 + rad, size - rad) - size

        # set random velocities for each particle (randomly distributed between x and y speed)
        self.vx = np.random.uniform(0, init_v) * np.random.choice([-1, 1])
        self.vy = np.sqrt(init_v**2 - self.vx**2) * np.random.choice([-1, 1])

        self.px = mass * self.vx
        self.py = mass * self.vy

        
    def update_x(self, val):
        self.x = val

    def update_y(self, val):
        self.y = val

    def update_vx(self, val):
        self.vx = val

    def update_vy(self, val):
        self.vy = val

    def get_kinetic_energy(self):
        return 0.5 * self.mass * (self.vx**2 + self.vy**2)


# In[32]:


class Simulation():  # this is where we will make them interact
    def __init__(self, N, T, size, mass, rad, delay=20):
        """Simulation class initialisation. This class handles the entire particle
        in a box thing.

        Parameters
        ----------
        N : `int`
            Total number of particles
        T : `float`
            Temperature of the system [K]
        size : `int`
            1/2 size of the box (rectangle) [km]
        rad : `int`
            Radius of the particles
            
        mass : 'int'
            Mass number of each particle
        
        delay : `int`
            Delay in milliseconds between showing/running timesteps
        """
        
        # Physical contants
        kB = 1.38e-16   # Boltzmann const. [cm^2 g s^-2 K^-1]
        mH = 1.67e-24   # Atomic Hydrogen mass [g]
        
        
        self.N = N
        self.T = T
        self.E = 3. / 2. * N * kB * T      # Total energy of the system
        self.epsilon = 3. / 2. * kB * T    # Initial kinetic energy of each particle
        self.size = size * 1e5             # Box size in [cm]
        self.mass = mass * mH
        self.rad = rad
        self.delay = delay

        
        # Initialise N particle classes
        self.particles = [particle(size=size, pid=i, T=T, mass=mass, rad=rad) for i in range(N)]

        
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

    def resolve_particle_collisions(self, particle):

        for i in range(self.N):
            for j in range(i + 1, self.N):
                distance = self._calculate_distance(self.particles[i], self.particles[j])
                if distance <= self.particles[i].rad + self.particles[j].rad:
                    # handle the collision

                    # calculate the velocity of the center of mass
                    cm_vx = (self.particles[i].mass * self.particles[i].vx 
                             + self.particles[j].mass * self.particles[j].vx) \
                                 / (self.particles[i].mass + self.particles[j].mass)

                    cm_vy = (self.particles[i].mass * self.particles[i].vy 
                             + self.particles[j].mass * self.particles[j].vy) \
                                / (self.particles[i].mass + self.particles[j].mass)
                    
                    # convert velocities to center of mass frame
                    v1_xi = -(self.particles[i].vx - cm_vx)
                    v2_xi = -(self.particles[j].vx - cm_vx)

                    v1_yi = -(self.particles[i].vy - cm_vy)
                    v2_yi = -(self.particles[j].vy - cm_vy)

                    self.particles[i].update_vx(v1_xi)
                    self.particles[i].update_vy(v1_yi)

                    self.particles[j].update_vx(v2_xi)
                    self.particles[j].update_vy(v2_yi)

                    total_ke = self.particles[i].get_kinetic_energy() + self.particles[j].get_kinetic_energy()
                    
                    print("total kinetic energy INITIAL", total_ke)


                    # calculate momenta in center of mass frame
                    p_xi = self.particles[i].mass * v1_xi + self.particles[j].mass * v2_xi
                    p_yi = self.particles[i].mass * v1_yi + self.particles[j].mass * v2_yi

                    # draw random angles in center of mass frame
                    cos_1 = np.random.uniform(-1, 1)
                    sin_1 = np.sqrt(1. - cos_1**2) * np.random.choice([-1, 1])
                    cos_2 = np.random.uniform(-1, 1)
                    sin_2 = np.sqrt(1 - cos_2**2) * np.random.choice([-1, 1])

                    # get final velocities IN CENTER OF MASS FRAME
                    v1_f = (p_xi * sin_2 - p_yi * cos_2) / (cos_1 * sin_2 - cos_2 * sin_1) / self.particles[i].mass
                    v2_f = (p_xi * sin_1 - p_yi * cos_1) / (cos_2 * sin_1 - cos_1 * sin_2) / self.particles[j].mass

                    # convert velocities back to lab frame 


                    


                    #updating the particle velocities 
                    self.particles[i].update_vx(np.absolute(v1_f) * cos_1)
                    self.particles[i].update_vy(np.absolute(v1_f) * sin_1)

                    self.particles[j].update_vx(np.absolute(v2_f) * cos_2)
                    self.particles[j].update_vy(np.absolute(v2_f) * sin_2)

                    total_ke = self.particles[i].get_kinetic_energy() + self.particles[j].get_kinetic_energy()

                    print("total kinetic energy FINAL", total_ke)
                    print()
    
    def resolve_particle_collisions(self, particle):                    



    def _calculate_distance(self, particle1, particle2):
        dist = np.sqrt((particle1.x - particle2.x)**2 + (particle1.y - particle2.y)**2)
        return dist


    def resolve_wall_collisions(self, particle):
        # check whether each particle hits the wall
        # for each collider reflect its velocity (account for ones that hit both walls)
        if (particle.size - np.absolute(particle.x)) <= particle.rad:
            particle.update_vx(-particle.vx)
            
        if (particle.size - np.absolute(particle.y)) <= particle.rad:
            particle.update_vy(-particle.vy)

    def run_simulation(self, steps=1000):
        for i in range(steps):
            # 1. update all particle positions based on current speeds
            for particle in self.particles:
                self._move_particle(particle)

            # 2. resolve whether any hit the wall and reflect them
                self.resolve_wall_collisions(particle)

            
            
            # 3. resolve any particle collisions and transfer momentum
            self.resolve_particle_collisions(particle)

            # update visualization with a delay
            self.root.after(self.delay, self.root.update())

            # change the timestep message as well
            self.canvas.itemconfig(self.timestep_message, text="Timestep = {}".format(i))

        self.root.mainloop()

    def get_velocities(self):
        raise NotImplementedError


# In[33]:


# import the code for the simulation
import matplotlib.pyplot as plt

# create a new class for the simulation with some randomish variable choices
sim = Simulation(N=100, T=3000, size=500, mass = 3e10, rad=5, delay=200)

# run the simulation
sim.run_simulation(steps=100)


# In[ ]:




