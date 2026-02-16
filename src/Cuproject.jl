# Cuproject.jl

export Cuproject, Cuproject!

"""
    Cuproject!(view, plan, image, viewidx)
GPU version of SPECT projection of `image` into a single `view` with index `viewidx`.
The `view` must be pre-allocated but need not be initialized to zero.
"""
function Cuproject!(
    view::AbstractMatrix{<:RealU},
    image::AbstractArray{<:RealU, 3},
    plan::CuSPECTplan,
    viewidx::Int,
)

    for z = 1:plan.imgsize[3]
        # rotate images
        Cuimrotate!(
            (@view plan.imgr[:, :, z]),
            (@view image[:, :, z]),
            plan.viewangle[viewidx],
            plan.planrot)
        # rotate mumap
        Cuimrotate!(
            (@view plan.mumapr[:, :, z]),
            (@view plan.mumap[:, :, z]),
            plan.viewangle[viewidx],
            plan.planrot)
    end

    for y = 1:plan.imgsize[2] # 1:ny

        Cugen_attenuation!(plan, y)
        # apply depth-dependent attenuation
        Cumul3dj!(plan.imgr, plan.exp_mumapr, y)

        Cufft_conv!(plan.add_view,
                   (@view plan.imgr[:, y, :]),
                   (@view plan.psfs[:, :, y, viewidx]),
                    plan.planpsf)

        view .+= plan.add_view
    end

    return view
end


"""
    Cuproject!(views, image, plan; viewlist)
Project `image` into multiple `views` with indexes `index` (defaults to `1:nview`).
The 3D `views` array must be pre-allocated, but need not be initialized.
"""
function Cuproject!(
    views::AbstractArray{<:RealU,3},
    image::AbstractArray{<:RealU,3},
    plan::CuSPECTplan;
    viewlist::AbstractVector{<:Int} = 1:plan.nview, # all views
)

    # loop over each view index
    for viewidx in viewlist
        Cuproject!((@view views[:,:,viewidx]), image, plan, viewidx)
    end

    return views
end


"""
    views = Cuproject(image, plan ; kwargs...)
GPU version of a convenience method for SPECT forward projector that allocates and returns views.
"""
function Cuproject(
    image::AbstractArray{<:RealU,3},
    plan::CuSPECTplan;
    kwargs...,
)
    views = CuArray{plan.T}(undef, plan.imgsize[1], plan.imgsize[3], plan.nview)
    Cuproject!(views, image, plan; kwargs...)
    return views
end


"""
    views = Cuproject(image, mumap, psfs, dy; kwargs...)
GPU version of a convenience method for SPECT forward projector that does all allocation
including initializing `plan`.

In
* `image` : 3D array `(nx,ny,nz)`
* `mumap` : `(nx,ny,nz)` 3D attenuation map, possibly zeros()
* `psfs` : 4D PSF array
* `dy::RealU` : pixel size
"""
function Cuproject(
    image::AbstractArray{<:RealU, 3},
    mumap::AbstractArray{<:RealU, 3}, # (nx,ny,nz) 3D attenuation map
    psfs::AbstractArray{<:RealU, 4}, # (px,pz,ny,nview)
    dy::RealU;
    kwargs...,
)
    size(mumap) == size(image) || throw(DimensionMismatch("image/mumap size"))
    size(image,2) == size(psfs,3) || throw(DimensionMismatch("image/psfs size"))
    plan = CuSPECTplan(mumap, psfs, dy; kwargs...)
    return Cuproject(image, plan; kwargs...)
end
