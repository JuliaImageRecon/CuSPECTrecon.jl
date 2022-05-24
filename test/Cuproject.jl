# Cuproject.jl

using SPECTrecon: SPECTplan, project!


@testset "Cuproject" begin
    T = Float32
    nx = 8; ny = nx
    nz = 6
    nview = 7

    mumap = rand(T, nx, ny, nz)

    px = 5
    pz = 3
    psfs = rand(T, px, pz, ny, nview)
    psfs = psfs .+ mapslices(reverse, psfs, dims = [1, 2]) # symmetrize
    psfs = psfs ./ mapslices(sum, psfs, dims = [1, 2])

    dy = T(4.7952)
    plan = SPECTplan(mumap, psfs, dy; T, interpmeth = :two, mode = :fast)
    x = randn(T, nx, ny, nz)
    views = zeros(T, nx, nz, nview)

    Cumumap = CuArray(mumap)
    Cupsfs = CuArray(psfs)
    Cuplan = CuSPECTplan(Cumumap, Cupsfs, dy; T)
    Cux = CuArray(x)
    Cuviews = CuArray(zeros(T, nx, nz, nview))
    project!(views, x, plan)
    Cuproject!(Cuviews, Cux, Cuplan)
    @test isapprox(Array(Cuviews), views; rtol = 1e-2)
end
