from firedrake import *
from firedrake.petsc import PETSc
from tracer.options import TracerOptions


n = 2**8
class TracerProblem():
    def __init__(self,
                 op=TracerOptions(),
                 stab='no',
                 mesh=RectangleMesh(50*n, 10*n, 50, 10),
                 fe=FiniteElement("Lagrange", triangle, 1),
                 high_order=False):
        
        # Mesh and function spaces
        assert(fe.family() == 'Lagrange')  # TODO: DG option if finite_element.family() == 'DG'
        self.mesh = mesh
        self.V = FunctionSpace(self.mesh, fe)
        self.P0 = FunctionSpace(self.mesh, "DG", 0)
        self.P1 = FunctionSpace(self.mesh, "CG", 1)
#         self.P1_vec = VectorFunctionSpace(self.mesh, "CG", 1)
        self.n = FacetNormal(self.mesh)
        self.h = CellSize(self.mesh)
        
        # Parameters
        self.op = op
#         self.op.target_vertices = self.mesh.num_vertices() * 0.85
        self.op.target_vertices = 2121. * 0.85
#         self.op.region_of_interest = [(op.loc_x, op.loc_y, op.loc_r)]
        self.op.region_of_interest = [(20., 7.5, 0.5)]
        self.x0 = 1.
        self.y0 = 5.
        self.r0 = 0.457
        self.nu = Constant(0.1)
        self.u = Constant([1., 0.])
        self.params = {'pc_type': 'lu', 'mat_type': 'aij' , 'ksp_monitor': True}
        self.stab = stab
        self.high_order = high_order
        
        # Outputting
        self.di = 'plots/'
        self.ext = ''
        if self.stab == 'SU':
            self.ext = '_su'
        elif self.stab == 'SUPG':
            self.ext = '_supg'
        self.sol_file = File(self.di + 'stationary_tracer' + self.ext + '.pvd')
        
    def source_term(self):
        x, y = SpatialCoordinate(self.mesh)
        bell = 1 + cos(pi * min_value(sqrt(pow(x - self.x0, 2) + pow(y - self.y0, 2)) / self.r0, 1.0))
        return interpolate(0. + conditional(ge(bell, 0.), bell, 0.), self.P1)
    
    def setup_equation(self):
        u = self.u
        nu = self.nu
        n = self.n
        f = self.source_term()

        # Finite element problem
        phi = TrialFunction(self.V)
        psi = TestFunction(self.V)
        a = psi*dot(u, grad(phi))*dx
        a += nu*inner(grad(phi), grad(psi))*dx
        a += - nu*psi*dot(n, nabla_grad(phi))*ds(1)
        a += - nu*psi*dot(n, nabla_grad(phi))*ds(2)
        L = f*psi*dx

        # Stabilisation
        if self.stab in ("SU", "SUPG"):
            tau = self.h / (2*sqrt(inner(u, u)))
            stab_coeff = tau * dot(u, grad(psi))
            R_a = dot(u, grad(phi))         # LHS component of strong residual
            if self.stab == 'SUPG':
                R_a += - div(nu*grad(phi))
                R_L = f                     # RHS component of strong residual
                L += stab_coeff*R_L*dx
            a += stab_coeff*R_a*dx
        
        self.lhs = a
        self.rhs = L
        self.bc = DirichletBC(self.V, 0, 1)

    def solve(self):
        phi = Function(self.V, name='Tracer concentration')
        solve(self.lhs == self.rhs, phi, bcs=self.bc, solver_parameters=self.params)
        self.sol = phi

    def objective_functional(self):
        ks = interpolate(self.op.indicator(self.mesh), self.P0)
        return assemble(self.sol * ks * dx)


tp = TracerProblem(stab='SUPG')
tp.setup_equation()
tp.solve()
PETSc.Sys.Print("%d cells, objective %.4e" % (tp.mesh.num_cells(), tp.objective_functional()), comm=COMM_WORLD)
