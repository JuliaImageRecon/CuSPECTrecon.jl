# Curotatez.jl

using SPECTrecon: imrotate!, imrotate_adj!, plan_rotate
using LinearAlgebra: dot


@testset "rotate-gpu" begin
    nx = 20
    θ_list = (1:23) / 12 * π
    T = Float32
    image2 = rand(T, nx, nx)
    plan = plan_rotate(nx; T, method = :two)[1]
    result = similar(image2)
    Cuimage2 = CuArray(image2)
    Curesult = similar(Cuimage2)
    Cuplan = CuPlanRotate(nx; T)
    for θ in θ_list
        imrotate!(result, image2, θ, plan)
        Cuimrotate!(Curesult, Cuimage2, θ, Cuplan)
        @test isapprox(result, Array(Curesult); rtol = 2e-3)
        imrotate_adj!(result, image2, θ, plan)
        Cuimrotate!(Curesult, Cuimage2, -θ, Cuplan) # rotate the image by "-angle"
        @test isapprox(result, Array(Curesult); rtol = 2e-1)
    end
end


@testset "Cuadjoint-rotate" begin
    nx = 20
    θ_list = (1:23) / 12 * π
    T = Float32
    x = CuArray(rand(T, nx, nx))
    y = CuArray(rand(T, nx, nx))
    out_x = similar(x)
    out_y = similar(y)
    Cuplan = CuPlanRotate(nx; T)
    for θ in θ_list
        Cuimrotate!(out_x, x, θ, Cuplan)
        Cuimrotate!(out_y, y, -θ, Cuplan)
        @test isapprox(dot(out_x, y), dot(out_y, x); rtol = 1e-2)
    end
end
