# Cuhelper.jl

using SPECTrecon: rot180!, rot_f90!, rotl90!, rotr90!
using OffsetArrays
using ImageFiltering: BorderArray, Fill, Pad


@testset "Cupadzero!" begin
    T = Float32
    x = randn(T, 7, 5)
    Cux = CuArray(x)
    y = randn(T, 3, 3)
    Cuy = CuArray(y)
    Cupadzero!(Cux, Cuy, (2, 2, 1, 1))
    z = OffsetArrays.no_offset_view(BorderArray(y, Fill(0, (2, 1), (2, 1))))
    x_cpu = Array(Cux)
    @test x_cpu == z
end


@testset "Cupadrepl!" begin
    T = Float32
    x = randn(T, 10, 9)
    Cux = CuArray(x)
    y = randn(T, 5, 4)
    Cuy = CuArray(y)
    Cupadrepl!(Cux, Cuy, (1, 4, 3, 2)) # up, down, left, right
    z = OffsetArrays.no_offset_view(BorderArray(y, Pad(:replicate, (1, 3), (4, 2)))) # up, left, down, right
    x_cpu = Array(Cux)
    @test x_cpu == z
end


@testset "Cupad2sizezero!" begin
    T = Float32
    x = reshape(Int16(1):Int16(15), 5, 3)
    Cux = CuArray(x)
    padsize = (8, 6)
    z = CuArray(randn(T, padsize))
    Cupad2sizezero!(z, Cux, padsize)
    tmp = OffsetArrays.no_offset_view(BorderArray(x, Fill(0, (2, 2), (1, 1))))
    z_cpu = Array(z)
    @test tmp == z_cpu
end


@testset "Cuplus1di!" begin
    T = Float32
    x = randn(T, 4, 9)
    Cux = CuArray(x)
    v = randn(T, 9)
    Cuv = CuArray(v)
    y = x[2, :] .+ v
    Cuplus1di!(Cux, Cuv, 2)
    x_cpu = Array(Cux)
    @test x_cpu[2, :] == y
end


@testset "Cuplus1dj!" begin
    T = Float32
    x = randn(T, 9, 4)
    Cux = CuArray(x)
    v = randn(T, 9)
    Cuv = CuArray(v)
    y = x[:, 2] .+ v
    Cuplus1dj!(Cux, Cuv, 2)
    x_cpu = Array(Cux)
    @test x_cpu[:, 2] == y
end


@testset "Cuplus2di!" begin
    T = Float32
    x = randn(9)
    Cux = CuArray(x)
    v = randn(4, 9)
    Cuv = CuArray(v)
    y = x .+ v[2, :]
    Cuplus2di!(Cux, Cuv, 2)
    x_cpu = Array(Cux)
    @test x_cpu == y
end


@testset "Cuplus2dj!" begin
    T = Float32
    x = randn(T, 9)
    Cux = CuArray(x)
    v = randn(T, 9, 4)
    Cuv = CuArray(v)
    y = x .+ v[:, 2]
    Cuplus2dj!(Cux, Cuv, 2)
    x_cpu = Array(Cux)
    @test x_cpu == y
end


@testset "Cuplus3di!" begin
    T = Float32
    x = randn(T, 9, 7)
    Cux = CuArray(x)
    v = randn(T, 4, 9, 7)
    Cuv = CuArray(v)
    y = x .+ v[2, :, :]
    Cuplus3di!(Cux, Cuv, 2)
    x_cpu = Array(Cux)
    @test x_cpu == y
end


@testset "Cuplus3dj!" begin
    T = Float32
    x = randn(T, 9, 7)
    Cux = CuArray(x)
    v = randn(T, 9, 4, 7)
    Cuv = CuArray(v)
    y = x .+ v[:, 2, :]
    Cuplus3dj!(Cux, Cuv, 2)
    x_cpu = Array(Cux)
    @test x_cpu == y
end


@testset "Cuplus3dk!" begin
    T = Float32
    x = randn(T, 9, 7)
    Cux = CuArray(x)
    v = randn(T, 9, 7, 4)
    Cuv = CuArray(v)
    y = x .+ v[:, :, 2]
    Cuplus3dk!(Cux, Cuv, 2)
    x_cpu = Array(Cux)
    @test x_cpu == y
end


@testset "Cuscale3dj!" begin
    T = Float32
    x = randn(T, 9, 7)
    Cux = CuArray(x)
    v = randn(T, 9, 4, 7)
    Cuv = CuArray(v)
    s = -0.5
    y = s * v[:, 2, :]
    Cuscale3dj!(Cux, Cuv, 2, s)
    x_cpu = Array(Cux)
    @test x_cpu == y
end


@testset "Cumul3dj!" begin
    T = Float32
    x = randn(T, 9, 4, 7)
    Cux = CuArray(x)
    v = randn(T, 9, 7)
    Cuv = CuArray(v)
    y = x[:,2,:] .* v
    Cumul3dj!(Cux, Cuv, 2)
    x_cpu = Array(Cux[:,2,:])
    @test x_cpu == y
end


@testset "Cucopy3dj!" begin
    T = Float32
    x = randn(T, 9, 7)
    Cux = CuArray(x)
    v = randn(T, 9, 4, 7)
    Cuv = CuArray(v)
    y = v[:,2,:]
    Cucopy3dj!(Cux, Cuv, 2)
    x_cpu = Array(Cux)
    @test x_cpu == y
end


@testset "Curotl90!" begin
    T = Float32
    N = 20
    A = rand(T, N, N)
    B = similar(A, N, N)
    CuA = CuArray(A)
    CuB = CuArray(B)
    Curotl90!(CuB, CuA)
    rotl90!(B, A)
    @test isequal(B, Array(CuB))
end


@testset "Curotr90!" begin
    T = Float32
    N = 20
    A = rand(T, N, N)
    B = similar(A)
    CuA = CuArray(A)
    CuB = CuArray(B)
    Curotr90!(CuB, CuA)
    rotr90!(B, A)
    @test isequal(B, Array(CuB))
end


@testset "Curot180!" begin
    T = Float32
    N = 20
    A = rand(T, N, N)
    B = similar(A)
    CuA = CuArray(A)
    CuB = CuArray(B)
    Curot180!(CuB, CuA)
    rot180!(B, A)
    @test isequal(B, Array(CuB))
end


@testset "Curot_f90!" begin
    T = Float32
    N = 20
    A = CuArray(rand(T, N, N))
    B = CuArray(rand(T, N, N))
    @test_throws String Curot_f90!(A, B, -1)
    @test_throws String Curot_f90!(A, B, 4)
end
