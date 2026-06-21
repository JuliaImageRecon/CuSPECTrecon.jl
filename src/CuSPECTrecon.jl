"""
    CuSPECTrecon
GPU version of system matrix (forward and back-projector) for SPECT image reconstruction.
"""
module CuSPECTrecon

    include("Cuhelper.jl")
    include("Cuplan-rotate.jl")
    include("Curotate.jl")
    include("Cuplan-fft.jl")
    include("Cupsf-gauss.jl")
    include("Cufft_convolve.jl")
    include("CuSPECTplan.jl")
    include("Cuproject.jl")
    include("Cubackproject.jl")

end # module
