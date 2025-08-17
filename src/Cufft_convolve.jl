# Cufft_convolve.jl

export Cufft_conv!, Cufft_conv_adj!
export Cufft_conv, Cufft_conv_adj

"""
    Cuimfilterz!(plan)
GPU version of FFT-based convolution of `plan.img_compl`
and kernel `plan.ker_compl` (not centered),
storing result in `plan.workmat`.
"""
function Cuimfilterz!(plan::CuPlanFFT)
    mul!(plan.img_compl, plan.fft_plan, plan.img_compl)
    mul!(plan.ker_compl, plan.fft_plan, plan.ker_compl)
    broadcast!(*, plan.img_compl, plan.img_compl, plan.ker_compl)
    mul!(plan.img_compl, plan.ifft_plan, plan.img_compl)
    fftshift!(plan.ker_compl, plan.img_compl)
    plan.workmat .= real.(plan.ker_compl)
    return plan.workmat
end


"""
    Cufft_conv!(output, img, ker, plan)
Convolve 2D image `img` with 2D (symmetric!) kernel `ker` using FFT,
storing the result in `output`.
"""
function Cufft_conv!(
    output::AbstractMatrix{<:RealU},
    img::AbstractMatrix{<:RealU},
    ker::AbstractMatrix{<:RealU},
    plan::CuPlanFFT,
)
    @boundscheck size(output) == size(img) || throw("size output")
    @boundscheck size(img) == (plan.nx, plan.nz) || throw("size img")
    @boundscheck size(ker) == (plan.px, plan.pz) ||
        throw("size ker $(size(ker)) $(plan.px) $(plan.pz)")

    # filter image with a kernel, using replicate padding and fft convolution
    Cupadrepl!(plan.img_compl, img, plan.padsize)

    Cupad2sizezero!(plan.ker_compl, ker, size(plan.ker_compl)) # zero pad kernel

    Cuimfilterz!(plan)

    (M, N) = size(img)
    copyto!(output, (@view plan.workmat[plan.padsize[1] .+ (1:M),
                                        plan.padsize[3] .+ (1:N)]))
    return output
end


"""
    Cufft_conv(img, ker; T)
GPU version of convolving 2D image `img` with 2D (symmetric!) kernel `ker` using FFT.
"""
function Cufft_conv(
    img::AbstractMatrix{I},
    ker::AbstractMatrix{K};
    T::DataType = promote_type(I, K, Float32),
) where {I <: Number, K <: Number}

    ker ≈ reverse(ker, dims=:) || throw("asymmetric kernel")
    nx, nz = size(img)
    px, pz = size(ker)
    plan = CuPlanFFT( ; nx, nz, px, pz, T)
    output = similar(img)
    Cufft_conv!(output, img, ker, plan)
    return output
end


"""
    Cufft_conv_adj!(output, img, ker, plan)
GPU version of the adjoint of `fft_conv!`.
"""
function Cufft_conv_adj!(
    output::AbstractMatrix{<:RealU},
    img::AbstractMatrix{<:RealU},
    ker::AbstractMatrix{<:RealU},
    plan::CuPlanFFT{T},
) where {T}

    @boundscheck size(output) == size(img) || throw("size output")
    @boundscheck size(img) == (plan.nx, plan.nz) || throw("size img")
    @boundscheck size(ker) == (plan.px, plan.pz) ||
        throw("size ker $(size(ker)) $(plan.px) $(plan.pz)")

    Cupadzero!(plan.img_compl, img, plan.padsize) # pad the image with zeros
    Cupad2sizezero!(plan.ker_compl, ker, size(plan.ker_compl)) # pad the kernel with zeros

    Cuimfilterz!(plan)
    (M, N) = size(img)
    # adjoint of replicate padding
    plan.workvecz .= zero(T)
    for i = 1:plan.padsize[1]
        Cuplus2di!(plan.workvecz, plan.workmat, i)
    end
    Cuplus1di!(plan.workmat, plan.workvecz, 1+plan.padsize[1])

    plan.workvecz .= zero(T)
    for i = (plan.padsize[1]+M+1):size(plan.workmat, 1)
        Cuplus2di!(plan.workvecz, plan.workmat, i)
    end
    Cuplus1di!(plan.workmat, plan.workvecz, M+plan.padsize[1])

    plan.workvecx .= zero(T)
    for j = 1:plan.padsize[3]
        Cuplus2dj!(plan.workvecx, plan.workmat, j)
    end
    Cuplus1dj!(plan.workmat, plan.workvecx, 1+plan.padsize[3])

    plan.workvecx .= zero(T)
    for j = (plan.padsize[3]+N+1):size(plan.workmat, 2)
        Cuplus2dj!(plan.workvecx, plan.workmat, j)
    end
    Cuplus1dj!(plan.workmat, plan.workvecx, N+plan.padsize[3])

    copyto!(output,
        (@view plan.workmat[(plan.padsize[1]+1):(plan.padsize[1]+M),
                            (plan.padsize[3]+1):(plan.padsize[3]+N)]),
    )

    return output
end


"""
    Cufft_conv_adj(img, ker; T)
GPU version of the adjoint of `fft_conv`.
"""
function Cufft_conv_adj(
    img::AbstractMatrix{I},
    ker::AbstractMatrix{K};
    T::DataType = promote_type(I, K, Float32),
) where {I <: Number, K <: Number}

    ker ≈ reverse(ker, dims=:) || throw("asymmetric kernel")
    nx, nz = size(img)
    px, pz = size(ker)
    plan = CuPlanFFT( ; nx, nz, px, pz, T)
    output = similar(Matrix{T}, size(img))
    Cufft_conv_adj!(output, img, ker, plan)
    return output
end
