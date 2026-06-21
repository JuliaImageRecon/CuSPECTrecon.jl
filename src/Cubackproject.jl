# Cubackproject.jl

export Cubackproject, Cubackproject!


"""
    Cubackproject!(image, view, plan, viewidx)
Backproject a single view.
"""
function Cubackproject!(
    image::AbstractArray{<:RealU, 3},
    view::AbstractMatrix{<:RealU},
    plan::CuSPECTplan,
    viewidx::Int,
)

    for z = 1:plan.imgsize[3] # 1:nz
        # rotate mumap by "-angle"
        Cuimrotate!((@view plan.mumapr[:, :, z]),
                    (@view plan.mumap[:, :, z]),
                    plan.viewangle[viewidx],
                    plan.planrot)
    end

    # adjoint of convolving img with psf and applying attenuation map
    for y = 1:plan.imgsize[2] # 1:ny

        Cugen_attenuation!(plan, y)

        Cufft_conv_adj!((@view plan.imgr[:, y, :]),
                         view,
                        (@view plan.psfs[:, :, y, viewidx]),
                         plan.planpsf)

        Cumul3dj!(plan.imgr, plan.exp_mumapr, y)
    end

    # adjoint of rotating image, again, rotating by "-angle"
    for z = 1:plan.imgsize[3] # 1:nz

        Cuimrotate!((@view image[:, :, z]),
                    (@view plan.imgr[:, :, z]),
                    -plan.viewangle[viewidx],
                     plan.planrot)
    end

    return image
end


"""
    Cubackproject!(image, views, plan ; viewlist)
Backproject multiple views into `image`.
Users must initialize `image` to zero.
"""
function Cubackproject!(
    image::AbstractArray{<:RealU, 3},
    views::AbstractArray{<:RealU, 3},
    plan::CuSPECTplan;
    viewlist::AbstractVector{<:Int} = 1:plan.nview, # all views
)

    # loop over each view index
    for viewidx in viewlist
        Cubackproject!(plan.add_img, (@view views[:, :, viewidx]), plan, viewidx)
        broadcast!(+, image, image, plan.add_img)
    end
    return image
end


"""
    image = Cubackproject(views, plan ; kwargs...)
SPECT backproject `views`; this allocates the returned 3D array.
"""
function Cubackproject(
    views::AbstractArray{<:RealU, 3},
    plan::CuSPECTplan;
    kwargs...,
)
    image = CuArray(zeros(plan.T, plan.imgsize))
    Cubackproject!(image, views, plan; kwargs...)
    return image
end


"""
    image = Cubackproject(views, mumap, psfs, dy; kwargs...)
SPECT backproject `views` using attenuation map `mumap` and PSF array `psfs` for pixel size `dy`.
This method initializes the `plan` as a convenience.
Most users should use `Cubackproject!` instead after initializing those, for better efficiency.
"""
function Cubackproject(
    views::AbstractArray{<:RealU, 3}, # [nx,nz,nview]
    mumap::AbstractArray{<:RealU, 3}, # [nx,ny,nz] attenuation map, must be 3D, possibly zeros()
    psfs::AbstractArray{<:RealU, 4},
    dy::RealU;
    kwargs...,
)

    size(mumap,1) == size(mumap,1) == size(views,1) ||
        throw(DimensionMismatch("nx"))
    size(mumap,3) == size(views,2) || throw(DimensionMismatch("nz"))
    plan = CuSPECTplan(mumap, psfs, dy; kwargs...)
    return Cubackproject(views, plan; kwargs...)
end
