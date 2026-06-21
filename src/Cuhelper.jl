# Cuhelper.jl
# A lot of helper functions

export Cupadzero!, Cupadrepl!, Cupad2sizezero!
export fftshift!, ifftshift!
export Cuplus1di!, Cuplus1dj!
export Cuplus2di!, Cuplus2dj!
export Cuplus3di!, Cuplus3dj!, Cuplus3dk!
export Cuscale3dj!, Cumul3dj!, Cucopy3dj!
export Curotl90!, Curotr90!, Curot180!, Curot_f90!
using LinearAlgebra
using FFTW

const RealU = Number
Power2 = x -> 2^(ceil(Int, log2(x)))
_padup(nx, px)    =  ceil(Int, (Power2(nx + px - 1) - nx) / 2)
_paddown(nx, px)  = floor(Int, (Power2(nx + px - 1) - nx) / 2)
_padleft(nz, pz)  =  ceil(Int, (Power2(nz + pz - 1) - nz) / 2)
_padright(nz, pz) = floor(Int, (Power2(nz + pz - 1) - nz) / 2)


"""
    Cupadzero!(output, img, padsize)
GPU version of padzero!
"""
function Cupadzero!(
    output::AbstractMatrix{T},
    img::AbstractMatrix,
    padsize::NTuple{4, <:Int}, # up, down, left, right
    ) where {T}

    @boundscheck size(output) ==
        size(img) .+ (padsize[1] + padsize[2], padsize[3] + padsize[4]) || throw("size")
    M, N = size(img)
    output .= zero(T)
    (@view output[padsize[1] + 1:padsize[1] + M, padsize[3] + 1:padsize[3] + N]) .= img
    return output

end


"""
    Cupadrepl!(output, img, padsize)
GPU version of padrepl!
"""
function Cupadrepl!(output::AbstractMatrix{T},
                    img::AbstractMatrix,
                    padsize::NTuple{4, <:Int}
                    ) where {T}

    @boundscheck size(output) ==
        size(img) .+ (padsize[1] + padsize[2], padsize[3] + padsize[4]) || throw("size")
    M, N = size(img)
    output .= zero(T)
    (@view output[padsize[1] + 1:padsize[1] + M, padsize[3] + 1:padsize[3] + N]) .= img
    (@view output[1:padsize[1], padsize[3] + 1:padsize[3] + N]) .= (@view img[1:1, :])
    (@view output[padsize[1] + M + 1 : end, padsize[3] + 1 : padsize[3] + N]) .= (@view img[end:end, :])
    (@view output[:, 1:padsize[3]]) .= (@view output[:, padsize[3] + 1 : padsize[3] + 1])
    (@view output[:, padsize[3] + N + 1: end]) .= (@view output[:, padsize[3] + N : padsize[3] + N])
    return output
end


"""
    Cupad2sizezero!(output, img, padsize)
GPU version of pad2sizezero!
"""
function Cupad2sizezero!(output::AbstractMatrix{T},
                         img::AbstractMatrix,
                         padsize::Tuple{<:Int, <:Int},
                         ) where {T}

        @boundscheck size(output) == padsize || throw("size")
        dims = size(img)
        pad_dims = ceil.(Int, (padsize .- dims) ./ 2)
        output .= zero(T)
        (@view output[pad_dims[1] + 1:pad_dims[1] + dims[1], pad_dims[2] + 1:pad_dims[2] + dims[2]]) .= img
        return output

end


"""
    fftshift!(dst, src), ifftshift!(dst, src)
fftshift! and ifftshift! work for both cpu and gpu
"""
fftshift!(dst::AbstractArray, src::AbstractArray) = circshift!(dst, src, size(src) .÷ 2)

ifftshift!(dst::AbstractArray, src::AbstractArray) = circshift!(dst, src, size(src) .÷ -2)


"""
    Cuplus1di!(mat2d, mat1d, i)
GPU version of `mat2d[i, :] += mat1d`
"""
Base.@propagate_inbounds function Cuplus1di!(
    mat2d::AbstractMatrix,
    mat1d::AbstractVector,
    i::Int,
)
    @boundscheck (size(mat1d, 1) == size(mat2d, 2) || throw("size2"))
    @boundscheck (1 ≤ i ≤ size(mat2d, 1) || throw("bad i"))
    (@view mat2d[i, :]) .+= mat1d
    return mat2d
end


"""
    Cuplus1dj!(mat2d, mat1d, j)
GPU version of `mat2d[:, j] += mat1d`
"""
Base.@propagate_inbounds function Cuplus1dj!(
    mat2d::AbstractMatrix,
    mat1d::AbstractVector,
    j::Int,
)
    @boundscheck (size(mat1d, 1) == size(mat2d, 1) || throw("size1"))
    @boundscheck (1 ≤ j ≤ size(mat2d, 2) || throw("bad j"))
    (@view mat2d[:, j]) .+= mat1d
    return mat2d
end


"""
    Cuplus2di!(mat1d, mat2d, i)
GPU version of `mat1d += mat2d[i,:]`
"""
Base.@propagate_inbounds function Cuplus2di!(
    mat1d::AbstractVector,
    mat2d::AbstractMatrix,
    i::Int,
)
    @boundscheck (size(mat1d, 1) == size(mat2d, 2) || throw("size2"))
    @boundscheck (1 ≤ i ≤ size(mat2d, 1) || throw("bad i"))
    mat1d .+= (@view mat2d[i, :])
    return mat1d
end


"""
    Cuplus2dj!(mat1d, mat2d, j)
GPU version of plus2dj!
"""
Base.@propagate_inbounds function Cuplus2dj!(
    mat1d::AbstractVector,
    mat2d::AbstractMatrix,
    j::Int,
)
    @boundscheck (size(mat1d, 1) == size(mat2d, 1) || throw("size1"))
    @boundscheck (1 ≤ j ≤ size(mat2d, 2) || throw("bad j"))
    mat1d .+= (@view mat2d[:, j])
    return mat1d
end


"""
    Cuplus3di!(mat2d, mat3d, i)
GPU version of plus3di!
"""
Base.@propagate_inbounds function Cuplus3di!(
    mat2d::AbstractMatrix,
    mat3d::AbstractArray{<:Any, 3},
    i::Int,
)
    @boundscheck (size(mat2d, 1) == size(mat3d, 2) || throw("size2"))
    @boundscheck (size(mat2d, 2) == size(mat3d, 3) || throw("size3"))
    @boundscheck (1 ≤ i ≤ size(mat3d, 1) || throw("bad i"))
    mat2d .+= (@view mat3d[i, :, :])
    return mat2d
end


"""
    Cuplus3dj!(mat2d, mat3d, j) # use sum!
GPU version of plus3dj!
"""
Base.@propagate_inbounds function Cuplus3dj!(
    mat2d::AbstractMatrix,
    mat3d::AbstractArray{<:Any, 3},
    j::Int,
)
    @boundscheck (size(mat2d, 1) == size(mat3d, 1) || throw("size1"))
    @boundscheck (size(mat2d, 2) == size(mat3d, 3) || throw("size3"))
    @boundscheck (1 ≤ j ≤ size(mat3d, 2) || throw("bad j"))
    mat2d .+= (@view mat3d[:, j, :])
    return mat2d
end


"""
    Cuplus3dk!(mat2d, mat3d, k)
GPU version of plus3dk!
"""
Base.@propagate_inbounds function Cuplus3dk!(
    mat2d::AbstractMatrix,
    mat3d::AbstractArray{<:Any, 3},
    k::Int,
)
    @boundscheck (size(mat2d, 1) == size(mat3d, 1) || throw("size1"))
    @boundscheck (size(mat2d, 2) == size(mat3d, 2) || throw("size2"))
    @boundscheck (1 ≤ k ≤ size(mat3d, 3) || throw("bad k"))
    mat2d .+= (@view mat3d[:, :, k])
    return mat2d
end


"""
    Cuscale3dj!(mat2d, mat3d, j, s)
GPU version of scale3dj!
"""
Base.@propagate_inbounds function Cuscale3dj!(
    mat2d::AbstractMatrix,
    mat3d::AbstractArray{<:Any, 3},
    j::Int,
    s::RealU,
)
    @boundscheck (size(mat2d, 1) == size(mat3d, 1) || throw("size1"))
    @boundscheck (size(mat2d, 2) == size(mat3d, 3) || throw("size3"))
    @boundscheck (1 ≤ j ≤ size(mat3d, 2) || throw("bad j"))
    copyto!(mat2d, (@view mat3d[:, j, :]))
    mat2d .*= s
    return mat2d
end


"""
    Cumul3dj!(mat3d, mat2d, j)
GPU version of mul3dj!
"""
Base.@propagate_inbounds function Cumul3dj!(
    mat3d::AbstractArray{<:Any, 3},
    mat2d::AbstractMatrix,
    j::Int,
)
    @boundscheck (size(mat3d, 1) == size(mat2d, 1) || throw("size1"))
    @boundscheck (size(mat3d, 3) == size(mat2d, 2) || throw("size3"))
    @boundscheck (1 ≤ j ≤ size(mat3d, 2) || throw("bad j"))
    (@view mat3d[:, j, :]) .*= mat2d
    return mat3d
end


"""
    Cucopy3dj!(mat2d, mat3d, j)
GPU version of copy3dj!
"""
Base.@propagate_inbounds function Cucopy3dj!(
    mat2d::AbstractMatrix,
    mat3d::AbstractArray{<:Any, 3},
    j::Int,
)
    @boundscheck (size(mat3d, 1) == size(mat2d, 1) || throw("size1"))
    @boundscheck (size(mat3d, 3) == size(mat2d, 2) || throw("size3"))
    @boundscheck (1 ≤ j ≤ size(mat3d, 2) || throw("bad j"))
    copyto!(mat2d, (@view mat3d[:, j, :]))
    return mat2d
end


"""
    Curotl90!(B::AbstractMatrix, A::AbstractMatrix)
In place GPU version of `rotl90`, returning rotation of `A` in `B`.
"""
function Curotl90!(B::AbstractMatrix, A::AbstractMatrix)
    N = size(B, 2)
    for i = 1:N
        copyto!((@view B[:, i]), (@view A[i, end:-1:1]))
    end
    return B
end


"""
    Curotr90!(B::AbstractMatrix, A::AbstractMatrix)
In place GPU version of `rotr90`, returning rotation of `A` in `B`.
"""
function Curotr90!(B::AbstractMatrix, A::AbstractMatrix)
    N = size(B, 2)
    for i = 1:N
        copyto!((@view B[:, N-i+1]), (@view A[i, :]))
    end
    return B
end


"""
    Curot180!(B::AbstractMatrix, A::AbstractMatrix)
In place GPU version of `rot180`, returning rotation of `A` in `B`.
"""
function Curot180!(B::AbstractMatrix, A::AbstractMatrix)
    N = size(B, 2)
    for i = 1:N
        copyto!((@view B[N-i+1, end:-1:1]), (@view A[i, :]))
    end
    return B
end


"""
    Curot_f90!(output, img, m)
In-place GPU version of rotating an image by 90/180/270 degrees
"""
function Curot_f90!(output::AbstractMatrix, img::AbstractMatrix, m::Int)
    if m == 0
        output .= img
    elseif m == 1
        Curotl90!(output, img)
    elseif m == 2
        Curot180!(output, img)
    elseif m == 3
        Curotr90!(output, img)
    else
        throw("invalid m!")
    end
    return output
end
