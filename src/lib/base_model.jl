using Parameters

# applies to super-sets of model parameters, rn only for Optim_Params
abstract type ParameterCollection end
# applies to structs that collect model parameters
abstract type ModelParameters <: ParameterCollection end
floatOrVec = Union{Float64, Vector{Float64}}

(::Type{T})(tuples::AbstractArray{<:Tuple}) where {T<:ParameterCollection} = map(t->T(t...), tuples)

function Base.length(it::ParameterCollection)
    return 1
end

function Base.iterate(it::ParameterCollection)
    return (it, nothing)
end

function Base.iterate(it::ParameterCollection, nothing)
    return nothing
end

function Base.:(==)(a::T, b::T) where {T <: ModelParameters}
    for key in fieldnames(T)
        getproperty(a,key) == getproperty(b,key) || return false
    end
    return true
end

function range_of_model_parameters(
    model_type::Type{T}, args...
) where {T <: ModelParameters}
    return vec([ T(param_set...) for param_set in Iterators.product(args...) ])
end