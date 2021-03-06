# Passive tracer transport advection-diffusion problem

Based on "Point discharge with diffusion" test case by D. Lebris, from TELEMAC-2D validation
document version 7.0 (2015), pp. 178-184.

Anisotropic mesh adaptivity is applied to the tracer transport problem in the coastal,
estuarine and ocean modelling solver, [Thetis][1]. Thetis builds upon the [Firedrake][2]
project, which enables efficient FEM solution in Python by automatic generation of C code. Anisotropic mesh adaptivity is achieved using [PRAgMaTIc][3].

This is research of the Applied Modelling and Computation Group ([AMCG][4]) at
Imperial College London.

* Download the [Firedrake][1] install script, set
    * ``export PETSC_CONFIGURE_OPTIONS="--download-pragmatic --with-cxx-dialect=C++11"``

    and install with option parameters ``--install pyadjoint`` and ``--install thetis``.

* Fetch and checkout the remote branches
    * ``https://bitbucket.org/dolfin-adjoint/pyadjoint/branch/linear-solver`` for pyadjoint;
    * ``https://github.com/thetisproject/thetis/tree/residuals`` for thetis;
    * ``https://github.com/taupalosaurus/firedrake`` for firedrake, fork ``barral/meshadapt``
    and call ``make`` in ``firedrake/src/firedrake`` to enable pragmatic drivers.
    * ``https://github.com/jwallwork23/adapt_utils`` and add this to ``PYTHONPATH`` environment variable.

#### For feedback, comments and questions, please email j.wallwork16@imperial.ac.uk.

[1]: http://thetisproject.org/index.html "Thetis"
[2]: http://firedrakeproject.org/ "Firedrake"
[3]: https://github.com/meshadaptation/pragmatic "PRAgMaTIc"
[4]: http://www.imperial.ac.uk/earth-science/research/research-groups/amcg/ "AMCG"

