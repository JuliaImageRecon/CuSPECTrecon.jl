# Cupsf-gauss.jl


@testset "Cupsf-gauss" begin
    psf = @inferred Cupsf_gauss()
    @test Array(psf) isa Array{Float32,3}

    ny = 4
    px = 5
    pz = 3
    psf = @inferred Cupsf_gauss(; ny, px, pz, fwhm = zeros(ny))
    tmp = zeros(px,pz)
    tmp[(end+1)÷2,(end+1)÷2] = 1 # Kronecker impulse
    tmp = repeat(tmp, 1, 1, ny)
    @test Array(psf) == tmp

    ny = 4
    px = 5
    pz = 3
    psf = @inferred Cupsf_gauss(; ny, px, pz,
        fwhm_x = fill(Inf, ny),
        fwhm_z = zeros(ny),
    )
    tmp = zeros(px,pz)
    tmp[:,(end+1)÷2] .= 1/px # wide in x
    tmp = repeat(tmp, 1, 1, ny)
    @test Array(psf) ≈ tmp
end
