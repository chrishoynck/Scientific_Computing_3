# **Eigenmodes of drums or membranes of different shapes, Direct methods for solving steady state problems, and The leapfrog method**
By Bart Koedijk (15756785), Charlotte Koolen (15888592), & Chris Hoynck van Papendrecht (15340791)

## **Project Overview**
- **Eigenmodes of drums or membranes of different shapes**  
  - Illustrating how eigenvalues and eigenfunctions depend on domain geometry.
- **Direct methods for solving steady state problems**  
  - Steady-state diffusion problems solved using direct matrix factorization methods.
- **The leapfrog method**  
  - Demonstration of a second-order explicit scheme for time-dependent PDEs.

The focus is on understanding how boundary conditions, geometry, and chosen numerical schemes influence the resulting solutions.

## **Main Components**
- **main.ipynb** – A Jupyter Notebook demonstrating 
- **src/solutions/direct_diffusion.py** – Functions for solving steady-state diffusion equations using direct methods.
- **src/solutions/eigenmodes_part1.py** – Functions to discretize domains to compute eigenmodes/eigenfrequencies of membranes.
- **src/solutions/leapfrog.py** – Implementation of the leapfrog scheme.
- **src/visualizations.py** – Plotting utilities.
- **plots/** – Directory for saving generated figures and animations. The GIF of the eigenmodes in time (`eigenmode_animation.gif`) is saved here.
- **README.md** – This file, describing the project’s structure and usage.

## **Installation & Setup**
1. **Clone the repository:**
   
bash
   git clone https://github.com/chrishoynck/Scientific_Computing_3
   cd Scientific_Computing_3

2. **Create and activate a virtual environment:**
   
bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate

3. **Install dependencies:**
   
bash
   pip install -r requirements.txt


## **Usage**
1. **Run the Jupyter Notebook:**
   
bash
   jupyter notebook main.ipynb

2. **Follow the notebook cells** to see how each simulation is set up, how parameters are varied, and to render results.  change this to this assignment with this repo + add that the gif for eigenmodes simulation is added in the plots
