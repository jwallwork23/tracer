# External imports
from thetis_adjoint import *
from fenics_adjoint.solving import SolveBlock       # For extracting adjoint solutions
from fenics_adjoint.projection import ProjectBlock  # Exclude projections from tape reading
from firedrake.petsc import PETSc
import pyadjoint
from copy import copy

# Adaptivity imports
from adapt.adaptivity import *
from adapt.interpolation import interp
from adapt.misc import index_string

# Tracer imports
from tracer.callbacks import TracerCallback
from tracer.options import TracerOptions


def solve_tracer(prev_sol=None, iteration=0, op=TracerOptions()):
    """
    Solve tracer transport problem using Thetis.
    """

    ### SETUP

    # N.B. Known quantities (such as bathymetry and source term) should be regenerated from data
    #      rather than interpolated, wherever possible

    # Mesh and function spaces
    if prev_sol is None:
        mesh = RectangleMesh(30 * op.nx, 5 * op.nx, 60, 10)
    else:
        mesh = prev_sol.function_space().mesh()
    x, y = SpatialCoordinate(mesh)
    P1 = FunctionSpace(mesh, "CG", 1)
    P1DG = FunctionSpace(mesh, "DG", 1)
    P1_vec = VectorFunctionSpace(mesh, "CG", 1)

    # Initial and source conditions
    u0 = Function(P1_vec).interpolate(as_vector([1., 0.]))
    eta0 = Function(P1)
    b = Function(P1).assign(1.)
    bell = conditional(
        ge((1 + cos(pi * min_value(sqrt(pow(x - op.bell_x0, 2) + pow(y - op.bell_y0, 2)) / op.bell_r0, 1.0))), 0.),
        (1 + cos(pi * min_value(sqrt(pow(x - op.bell_x0, 2) + pow(y - op.bell_y0, 2)) / op.bell_r0, 1.0))),
        0.)
    source = Function(P1).interpolate(0. + bell)

    # Inflow boundary condition
    BCs = {'shallow water': {}, 'tracer': {1: {'value': Constant(0.)}}}

    # Artificial 'sponge' boundary condition
    nu = Function(P1).interpolate(op.viscosity + op.sponge_scaling * pow(max_value(0, x - op.sponge_start), 2))


    ### SOLVER
    solver_obj = solver2d.FlowSolver2d(mesh, b)
    options = solver_obj.options
    options.simulation_export_time = op.dt * op.dt_per_export
    options.simulation_end_time = op.end_time
    options.timestepper_type = op.timestepper
    options.timestep = op.dt
    options.output_directory = op.directory()
    options.fields_to_export = ['tracer_2d']
    options.fields_to_export_hdf5 = ['tracer_2d', 'uv_2d', 'elev_2d']
    options.compute_residuals_tracer = False
    options.solve_tracer = True
    options.tracer_only = True  # Hold shallow water variables fixed
    options.horizontal_diffusivity = nu
    options.tracer_source_2d = source
    if prev_sol is None:
        solver_obj.assign_initial_conditions(elev=eta0, uv=u0)
    else:
        # solver_obj.assign_initial_conditions(elev=eta0, uv=u0, tracer=prev_sol)
        solver_obj.load_state(iteration)
    cb1 = TracerCallback(solver_obj, parameters=op)
    solver_obj.add_callback(cb1, 'timestep')
    solver_obj.bnd_functions = BCs

    print("\nSolving forward problem...")
    solver_obj.iterate()
    J = cb1.get_val()  # Assemble objective functional for adjoint solver

    if op.solve_adjoint:
        print("\nSolving adoint problem...")
        compute_gradient(J, Control(nu))
        return solver_obj, get_working_tape()
    else:
        return solver_obj


def store_adjoint(solver_obj, tape, op=TracerOptions()):
    """
    Solve (discrete) adjoint equations using pyadjoint.
    """
    solve_blocks = [block for block in tape._blocks if isinstance(block, SolveBlock) and not isinstance(block, ProjectBlock) and block.adj_sol is not None]
    op.adjoint_steps = len(solve_blocks)
    remainder = op.adjoint_steps % op.dt_per_export   # Number of extra tape annotations in setup
    adjoint = Function(solver_obj.function_spaces.Q_2d, name='adjoint_2d')
    for i in range(op.adjoint_steps - 1, remainder - 2, -op.dt_per_export):  # FIXME: Why -2?
        adjoint.assign(solve_blocks[i].adj_sol)
        idx = int((i+1) / op.dt_per_export) - remainder
        index_str = index_string(idx)
        with DumbCheckpoint(op.directory() + 'hdf5/Adjoint2d_' + index_str, mode=FILE_CREATE) as sa:
            sa.store(adjoint)
            sa.close()
        time = (i+1)*op.dt
        op.adjoint_outfile.write(adjoint, time=time)
        line = ('{iexp:5d} {it:5d} T={t:10.2f} tracer norm: {q:10.4f}')
        print_output(line.format(iexp=idx, it=i+1, t=time, q=norm(adjoint)))
    tape.clear_tape()


def solve_and_estimate_error(prev_sol=None, counter=0, iteration=0, op=TracerOptions()):
    """
    Solve tracer transport problem using Thetis and estimate error using data stored to HDF5.
    """

    ### SETUP

    # N.B. Known quantities (such as bathymetry and source term) should be regenerated from data
    #      rather than interpolated, wherever possible

    # Mesh and function spaces
    if prev_sol is None:
        mesh = RectangleMesh(30 * op.nx, 5 * op.nx, 60, 10)
    else:
        mesh = prev_sol.function_space().mesh()
    x, y = SpatialCoordinate(mesh)
    P0 = FunctionSpace(mesh, "DG", 0)
    P1 = FunctionSpace(mesh, "CG", 1)
    P1DG = FunctionSpace(mesh, "DG", 1)
    P1_vec = VectorFunctionSpace(mesh, "CG", 1)

    # Initial and source conditions
    u0 = Function(P1_vec).interpolate(as_vector([1., 0.]))
    eta0 = Function(P1)
    b = Function(P1).assign(1.)
    bell = conditional(
        ge((1 + cos(pi * min_value(sqrt(pow(x - op.bell_x0, 2) + pow(y - op.bell_y0, 2)) / op.bell_r0, 1.0))), 0.),
        (1 + cos(pi * min_value(sqrt(pow(x - op.bell_x0, 2) + pow(y - op.bell_y0, 2)) / op.bell_r0, 1.0))),
        0.)
    source = Function(P1).interpolate(0. + bell)

    # Inflow boundary condition
    BCs = {'shallow water': {}, 'tracer': {1: {'value': Constant(0.)}}}

    # Artificial 'sponge' boundary condition
    nu = Function(P1).interpolate(op.viscosity + op.sponge_scaling * pow(max_value(0, x - op.sponge_start), 2))


    ### SOLVER
    solver_obj = solver2d.FlowSolver2d(mesh, b)
    options = solver_obj.options
    options.simulation_export_time = op.dt * op.dt_per_export
    options.simulation_end_time = op.end_time
    options.timestepper_type = op.timestepper
    options.timestep = op.dt
    options.output_directory = op.directory()
    options.fields_to_export = ['tracer_2d']
    options.fields_to_export_hdf5 = ['tracer_2d', 'uv_2d', 'elev_2d']
    options.compute_residuals_tracer = op.approach == 'DWR'
    options.solve_tracer = True
    options.tracer_only = True  # Hold shallow water variables fixed
    options.horizontal_diffusivity = nu
    options.tracer_source_2d = source
    solver_obj.assign_initial_conditions(elev=eta0, uv=u0, tracer=prev_sol)
    solver_obj.i_export = its
    solver_obj.next_export_t = its * options.simulation_export_time
    solver_obj.iteration = its * op.dt_per_export
    solver_obj.simulation_time = solver_obj.iteration * op.dt
    for e in solver_obj.exporters.values():
        e.set_next_export_ix(solver_obj.i_export)
    solver_obj.bnd_functions = BCs

    print("\nSolving forward problem and estimating errors...")
    solver_obj.iterate()

    tracer_ts = solver_obj.timestepper.timesteppers['tracer']
    adjoint = Function(P1DG, name='adjoint_2d')
    index_str = index_string(counter)
    with DumbCheckpoint(op.directory() + 'hdf5/Adjoint2d_' + index_str, mode=FILE_READ) as la:
        la.load(adjoint)
        la.close()

    if op.approach == 'DWR':
        cell_res = tracer_ts.cell_residual(adjoint)
        print("Cell residual: {:.4e}".format(norm(cell_res)))

        edge_res = tracer_ts.edge_residual(adjoint)
        if edge_res == 0:
            edge_res = Constant(0., domain=mesh)
        print("Edge residual: {:.4e}".format(norm(edge_res)))

        I = TestFunction(P0)
        h = CellSize(mesh)
        epsilon = project(sqrt(assemble(I * (h * h * cell_res * cell_res + h * edge_res * edge_res) * dx)), P0)
    elif op.approach == 'DWP':
        epsilon = project(solver_obj.fields.tracer_2d * adjoint, P0)
    epsilon.rename('error_estimate_2d')
    op.estimator_outfile.write(epsilon, time=solver_obj.simulation_time)

    return solver_obj


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", help="Choose adaptive approach from {'HessianBased', 'DWP', 'DWR'} (default 'FixedMesh')")
    args = parser.parse_args()

    op = TracerOptions(approach='FixedMesh' if args.a is None else args.a)
    #op.end_time = 5. - 0.5*op.dt

    # DWP and DWR estimators both use this workflow
    if op.solve_adjoint:
        solver_obj, tape = solve_tracer(op=op)
        store_adjoint(solver_obj, tape, op=op)
        with pyadjoint.stop_annotating():
            restart_time = op.dt * op.dt_per_export
            T_end = copy(op.end_time)
            t = 0.
            its = 0
            cnt = int(op.end_time/(op.dt*op.dt_per_export))
            op.end_time = copy(restart_time)
            sol = None
            while t < T_end:
                solver_obj = solve_and_estimate_error(prev_sol=sol, counter=cnt, iteration=its, op=op)
                sol = solver_obj.fields.tracer_2d
                its = solver_obj.i_export
                cnt -= 1
                op.end_time += restart_time
                t = solver_obj.simulation_time
            # FIXME: Why does restarted step have a different norm? - times not in sync?
            # TODO: Adapt mesh
    elif op.approach == 'HessianBased':
        restart_time = op.dt * op.dt_per_export
        T_end = copy(op.end_time)
        t = 0.
        op.end_time = copy(restart_time)
        sol = None
        while t < T_end:
            solver_obj = solve_tracer(prev_sol=sol, op=op)
            sol = solver_obj.fields.tracer_2d
            # TODO: Compute Hessian
            t += restart_time
    else:
        solver_obj = solve_tracer(op=op)
