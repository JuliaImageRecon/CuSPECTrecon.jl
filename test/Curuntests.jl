# Curuntests.jl

include("../src/CuSPECTrecon.jl")
using Main.CuSPECTrecon
using Test: @test, @testset, @test_throws, @inferred, detect_ambiguities
using CUDA
CUDA.allowscalar(false)

include("Cuhelper.jl")
include("Curotate.jl")
include("Cufftconv.jl")
include("Cupsf-gauss.jl")
include("Cuproject.jl")
include("Cubackproject.jl")

@testset "CuSPECTrecon" begin
    @test isempty(detect_ambiguities(CuSPECTrecon))
end
