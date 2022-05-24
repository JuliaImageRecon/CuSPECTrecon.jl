# Cubackproject.jl

using SPECTrecon: SPECTplan, backproject!
using LinearAlgebra: dot
using Random: seed!

@testset "Cubackproject" begin
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
    image = zeros(T, nx, ny, nz) # images must be initialized to zero
    views = rand(T, nx, nz, nview)
    Cuimage = CuArray(zeros(T, nx, ny, nz))
    Cuviews = CuArray(views)

    Cumumap = CuArray(mumap)
    Cupsfs = CuArray(psfs)
    Cuplan = CuSPECTplan(Cumumap, Cupsfs, dy; T)

    backproject!(image, views, plan)
    Cubackproject!(Cuimage, Cuviews, Cuplan)
    @test isapprox(Array(Cuimage), image; rtol = 5e-2) 
end


@testset "adjoint-Cuproject" begin
    seed!(0)
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

    Cumumap = CuArray(mumap)
    Cupsfs = CuArray(psfs)
    Cuplan = CuSPECTplan(Cumumap, Cupsfs, dy; T)

    image = CuArray(rand(T, nx, ny, nz))
    backimage = CuArray(zeros(T, nx, ny, nz))
    views = CuArray(rand(T, nx, nz, nview))
    forviews = CuArray(zeros(T, nx, nz, nview))

    Cuproject!(forviews, image, Cuplan)
    Cubackproject!(backimage, views, Cuplan)
    @test isapprox(dot(forviews, views), dot(backimage, image); rtol = 1e-2)
end
