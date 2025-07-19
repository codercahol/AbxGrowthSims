using DifferentialEquations
using JuMP
import Ipopt
using InfiniteOpt
using DifferentialEquations
using ForwardDiff
using Parameters
using Statistics
using NLopt
using LinearAlgebra
using DiffEqCallbacks

import HomotopyContinuation as HC
import BifurcationKit
const BK = BifurcationKit
using Accessors
using Memoization
using NumericalIntegration

using Dates
using DataFrames

abstract type ExpandedPrecursorModels <: ModelParameters end
abstract type PrecursorModelFamily <: ExpandedPrecursorModels end
abstract type tRNAModelFamily <: ModelParameters end
abstract type ABXAnticipationFamily <: ParameterCollection end

abstract type AbstractIntegrationRule end
struct RectangularIntegrationRule <: AbstractIntegrationRule end
struct TrapezoidalIntegrationRule <: AbstractIntegrationRule end

abstract type SolverType end
struct SingleSolver <: SolverType end
struct MultiSolver <: SolverType end


include("./utils.jl")

function mutate(p::ModelParameters; kwargs...)
    new_properties = Dict{Symbol,Any}()
    # make sure all the provided kwargs are valid properties
    if !all([prop in propertynames(p) for prop in keys(kwargs)])
        throw(ArgumentError("Invalid property name"))
    end
    # if there are multiple properties provided for a single keyword
    # return a vector of mutated models
    num_props = Dict(prop => length(kwargs[prop]) for prop in keys(kwargs))
    if any(values(num_props) .> 1)
        ks = keys(kwargs)
        vals = [kwargs[prop] for prop in ks]
        value_sets = [i for i in Iterators.product(vals...)]
        dict_sets = [Dict(zip(ks, vs)) for vs in value_sets]
        return [mutate(p; d...) for d in dict_sets]
    end
    for prop in propertynames(p)
        if prop in keys(kwargs)
            new_properties[prop] = kwargs[prop]
        else
            new_properties[prop] = getproperty(p, prop)
        end
    end
    p_type = typeof(p)
    return p_type(; new_properties...)
end

function Base.length(f::Function)
    return 1
end

function Base.iterate(f::Function)
    return (f, nothing)
end

function Base.iterate(f::Function, nothing)
    return nothing
end

function Base.copy(s::ModelParameters)
    # copy all parameters to the new instance
    props = Dict(k => getproperty(s, k) for k in propertynames(s))
    # new instance of the same type
    p_type = typeof(s)
    return p_type(; props...)
end

function Base.copy(s::ABXAnticipationFamily)
    # Create a new instance of the same type
    p_type = typeof(s)
    # copy all the the parameter tables to the new instance
    abx = copy(s.abx)
    ant = copy(s.ant)
    # return the new instance
    return p_type(abx, ant)
end

Numba = Union{Real,JuMP.VariableRef}

@with_kw struct PrecursorModelConstantRiboFraction <: PrecursorModelFamily
    γmax::Numba
    νmax::Numba
    Km::Numba
    φ_0::Numba
    α_R::Numba
end

@with_kw struct PrecursorModel <: PrecursorModelFamily
    γmax::Numba
    νmax::Numba
    Km::Numba
    φ_0::Numba
end
PrecursorModels = Union{PrecursorModel,Vector{PrecursorModel}}

#########################
# Main model parameters #
#########################
@with_kw struct PrecursorABXBaseModel <: ExpandedPrecursorModels
    γmax::Numba
    νmax::Numba
    Km::Numba
    φ_0::Numba
    C_max 
    kp::Numba
    ke::Numba
    k_on::Numba
    k_off::Numba
    k_ind::Numba
    n_prot::Numba
    Km_abx::Numba
    n_CAT::Numba
end

@with_kw struct PrecursorABXConstantRiboBase <: ExpandedPrecursorModels
    γmax::Numba
    νmax::Numba
    Km::Numba
    φ_0::Numba
    C_max # ::Numba or Function
    kp::Numba
    ke::Numba
    k_on::Numba
    k_off::Numba
    k_ind::Numba
    n_prot::Numba
    Km_abx::Numba
    n_CAT::Numba
    α_R::Numba
end

struct tRNAModel <: tRNAModelFamily
    γmax::Numba
    νmax::Numba
    Km::Numba
    φ_0::Numba
    τ::Numba
    κmax::Numba
end

struct tRNAModelConstantRiboFraction <: tRNAModelFamily
    γmax::Numba
    νmax::Numba
    Km::Numba
    φ_0::Numba
    τ::Numba
    κmax::Numba
    α_R::Numba
end

@with_kw struct AbxBaseModel <: ModelParameters
    γmax::Numba
    νmax::Numba
    Km::Numba
    φ_0::Numba
    τ::Numba
    κmax::Numba
    C_max::Numba
    kp::Numba
    ke::Numba
    k_on::Numba
    k_off::Numba
    k_ind::Numba
    n_prot::Numba
end

@with_kw struct AbxAnticipationParams <: ModelParameters
    αmax::Numba # maximum φ_AR
    αmin::Numba # AR protein reserve in ABX model
end

struct PrecursorABXwAnticipation <: ABXAnticipationFamily
    abx::PrecursorABXBaseModel
    ant::AbxAnticipationParams
end

mutable struct Perturbation
    amplitude::Numba
    duty::Numba
    period::Numba
    current_value::Numba
end

struct PcABXAntWPerturbation <: ABXAnticipationFamily
    abx::PrecursorABXBaseModel
    ant::AbxAnticipationParams
    perturbation::Perturbation
end

struct PrecursorMultiABXwAnticipation <: ABXAnticipationFamily
    abx::PrecursorABXBaseModel
    ant::Vector{AbxAnticipationParams}
end

function simple_mutate(p::ABXAnticipationFamily; kwargs...)
    # split kwargs into abx and ant
    abx_kwargs = Dict(k => v for (k, v) in kwargs if k in propertynames(p.abx))
    ant_kwargs = Dict(k => v for (k, v) in kwargs if k in propertynames(p.ant))
    abx = mutate(p.abx; abx_kwargs...)
    ant = mutate(p.ant; ant_kwargs...)
    p_type = typeof(p)
    return p_type(abx, ant)
end

function mutate(p::ABXAnticipationFamily; kwargs...)
    # Split kwargs into abx and ant groups
    abx_kwargs = Dict(k => v for (k, v) in kwargs if k in propertynames(p.abx))
    
    # Handle ABX params (same for both types)
    abx = mutate(p.abx; abx_kwargs...)
    
    # Different handling for single vs multi ant structs
    if p.ant isa AbxAnticipationParams
        # Simple case: single anticipation struct
        ant_kwargs = Dict(k => v for (k, v) in kwargs if k in propertynames(p.ant))
        ant = mutate(p.ant; ant_kwargs...)
    elseif p.ant isa Vector{<:AbxAnticipationParams}
        # Complex case: vector of anticipation structs
        ant = copy(p.ant)  # Create a copy to modify
        
        # Collect ant parameters
        ant_param_names = p.ant[1] isa AbxAnticipationParams ? 
                         propertynames(p.ant[1]) : Symbol[]
        
        for (key, value) in kwargs
            if key in ant_param_names
                if value isa Tuple{Int, Any}
                    # Case 1: Tuple of (index, value)
                    idx, val = value
                    if 1 <= idx <= length(ant)
                        ant[idx] = mutate(ant[idx]; Dict(key => val)...)
                    else
                        @warn "Index $idx out of bounds, skipping $key update"
                    end
                elseif value isa Vector{<:Tuple{Int, Any}}
                    # Case 2: Vector of (index, value) tuples
                    for (idx, val) in value
                        if 1 <= idx <= length(ant)
                            ant[idx] = mutate(ant[idx]; Dict(key => val)...)
                        else
                            @warn "Index $idx out of bounds, skipping $key update"
                        end
                    end
                elseif value isa Vector && length(value) == length(ant)
                    # Case 3: Vector of values matching the number of antibiotic structs
                    for i in 1:length(ant)
                        ant[i] = mutate(ant[i]; Dict(key => value[i])...)
                    end
                else
                    # Case 4: Single scalar value to apply to all
                    for i in 1:length(ant)
                        ant[i] = mutate(ant[i]; Dict(key => value)...)
                    end
                end
            end
        end
    else
        throw(ArgumentError("Invalid type for ant: $(typeof(p.ant))"))
    end

    # check if the new ModelParameters for abx and ant are a higher dimensional array
    # if p is a PrecursorMultiABXwAnticipation, then we expect ant to be a vector
    # and don't want to generate a tensor product. 
    if length(abx) > 1 || (length(ant) > 1 && !(p isa PrecursorMultiABXwAnticipation))
        # return a cartesian product of the two
        pairs = collect(Iterators.product(abx, ant))
        return typeof(p)(pairs)
    else
        # Construct the appropriate type
        return typeof(p)(abx, ant)
    end
end

struct PrecursorABXConstantRiboWAnt <: ABXAnticipationFamily
    abx::PrecursorABXConstantRiboBase
    ant::AbxAnticipationParams
end

struct ABXwAnticipation <: ABXAnticipationFamily
    abx::AbxBaseModel
    ant::AbxAnticipationParams
end

struct AbxAntANDRiboRegulation
    abx::AbxBaseModel
    ant::AbxAnticipationParams
    α_R::Function
    ppGpp::Function
end

AbxModels = Union{ABXwAnticipation,AbxAntANDRiboRegulation}

@with_kw struct JumpConfig{T<:AbstractIntegrationRule} <: ModelParameters
    n::Int   # mesh points
    Δt::Numba  # time step
    integration_rule::T
end


function model_partial_hypercube(
    max_model::PrecursorABXwAnticipation, min_model::PrecursorABXwAnticipation,
    length::Int
)
    """
    Only computes the hypercube for the abx parameters
    """
    @assert max_model.ant == min_model.ant "Anticipation parameters must be the same"

    range_abx_models = model_hypercube(
        max_model.abx, min_model.abx, length
    )
    return [PrecursorABXwAnticipation(abx, max_model.ant) for abx in range_abx_models]
end

function monod(v, Km)
    f(x) = v * x / (x + Km)
    return f
end

function integration_JuMP!(model, u, du, jc, ::RectangularIntegrationRule)
    @variable(model, δt[1:jc.n] == jc.Δt)

    for t in 2:jc.n
        s = t - 1
        for k in 1:length(u)
            @constraint(model, u[k][t] == u[k][s] + δt[t] * du[k][s])
        end
    end
end

function add_nutrient_shift!(model, νmax, new_ν, jc, shift_index)
    n = jc.n
    ν = @expression(model, ν[j=1:n], j < shift_index ? νmax : new_ν)
    return ν
end

function add_translation_shift!(model, γmax, new_γ_max, jc, shift_index)
    n = jc.n
    γ_max = @expression(model, γ_max[j=1:n], j < shift_index ? γmax : new_γ_max)
    return γ_max
end

# <1s (up to 2s) runtime
function simulate(
    u0, p::PrecursorModel, jc::JumpConfig;
    t_shift=nothing, ν_2=nothing, γ_max_2=nothing
)
    @unpack_PrecursorModel p
    @unpack_JumpConfig jc
    model = Model(Ipopt.Optimizer)

    if length(u0) < 3
        throw(ArgumentError("Initial conditions must be (at least) of length 3"))
    end

    # dynamical variables
    u = JuMP.@variables(model, begin
        0 <= φ_R[1:n] <= 1 - φ_0
        0 <= c_pc[1:n]
        0 <= M[1:n]
    end)

    # optimization knob
    @variable(model, 0 <= α_R[1:n] <= 1 - φ_0)

    # Shifts
    if isnothing(t_shift)
        # shift at halfway point by default
        shift_index = round(Int, n / 2)
    else
        shift_index = round(Int, t_shift / Δt)
    end
    # nutrient shift
    if !isnothing(ν_2)
        ν = add_nutrient_shift!(model, νmax, ν_2, jc, shift_index)
    else
        @variable(model, ν[1:n] == νmax)
    end
    # translation shift
    if !isnothing(γ_max_2)
        γ_max = add_translation_shift!(model, γmax, γ_max_2, jc, shift_index)
    else
        @variable(model, γ_max[1:n] == γmax)
    end

    # helper functions
    @expression(model, γ[j=1:n], γ_max[j] * c_pc[j] / (c_pc[j] + Km))
    @expression(model, λ[j=1:n], γ[j] * φ_R[j])

    du = Vector{Vector{JuMP.NonlinearExpr}}(undef, length(u))
    # growth dynamics
    du[1] = @expression(model, δφ_R[j=1:n], (α_R[j] - φ_R[j]) * λ[j])
    du[2] = @expression(model, δc_pc[j=1:n], ν[j] * (1 - φ_0 - φ_R[j]) - (1 + c_pc[j]) * λ[j])
    du[3] = @expression(model, δM[j=1:n], M[j] * λ[j])

    # integration
    integration_JuMP!(model, u, du, jc, integration_rule)

    @objective(model, Max, M[n])
    set_silent(model)

    # one set of initial conditions
    if length(u0) == 3
        for i in 1:length(u)
            fix(u[i][1], u0[i]; force=true)
        end

        optimize!(model)
        Ls = value.(λ)
        Rs = value.(φ_R)
        aRs = value.(α_R)
        cpcs = value.(c_pc)

        # if we have set of ICs for φ_R, c_pc, M
    elseif u0 isa Array{Tuple{Numba,Numba,Numba},3}
        m = length(u0)
        Ls = zeros(n, m)
        Rs = zeros(n, m)
        aRs = zeros(n, m)
        cpcs = zeros(n, m)
        for i in 1:m
            for j in 1:length(u)
                fix(u[j][1], u0[i][j]; force=true)
            end
            optimize!(model)

            Ls[:, i] = value.(λ)
            Rs[:, i] = value.(φ_R)
            aRs[:, i] = value.(α_R)
            cpcs[:, i] = value.(c_pc)
        end
    end
    return Rs, cpcs, aRs, Ls
end

function sim_inf(
    u0, p::PrecursorABXwAnticipation, jc::JumpConfig;
    smoothness=10
)
    @unpack_PrecursorABXBaseModel p.abx
    @unpack_AbxAnticipationParams p.ant
    @unpack_JumpConfig jc
    model = InfiniteModel(Ipopt.Optimizer)
    set_silent(model)

    T = Δt * n
    @infinite_parameter(model, t ∈ [0, T], num_supports = n)

    # dynamical variables
    vars = JuMP.@variables(model, begin
        0 <= φ_R <= 1 - φ_0, Infinite(t)
        0 <= c_pc, Infinite(t)
        0 <= C_bnd, Infinite(t)
        0 <= C_in, Infinite(t)
        # optimization knob
        0 <= α_R <= 1 - φ_0, Infinite(t)
    end)
    # initial conditions
    @constraint(
        model,
        [i = 1:4],
        vars[i](0) == u0[i]
    )

    @expressions(model, begin
        # concentration of free ribosomes
        C_Rf, n_prot .* φ_R .- C_bnd
        # mass fraction of free ribosomes
        φ_Rf, φ_R .- C_bnd ./ n_prot
        # metabolic protein mass fraction
        φ_M, 1 - φ_0 .- φ_R
        # translation rate
        γ, γmax .* c_pc ./ (c_pc .+ Km)
        # growth rate
        λ, γ .* φ_Rf
    end)
    @constraint(model, φ_Rf .>= 0)

    @constraint(model, ∂(φ_R, t) == (α_R - φ_R) * λ)
    @constraint(model, ∂(c_pc, t) == νmax .* (1 - φ_0 .- φ_R) .- (1 .+ c_pc) .* λ)
    @constraint(model, ∂(C_bnd, t) == k_on .* C_in .* C_Rf .- k_off .* C_bnd .- λ .* C_bnd)
    @constraint(model, ∂(C_in, t) == kp .* (C_max .- C_in) .+ k_off .* C_bnd .- k_on .* C_Rf .* C_in .- λ .* C_in)

    @objective(model, Max, ∫(λ, t) - smoothness * ∫(∂(α_R, t)^2, t))

    optimize!(model)
    Ls = value.(λ)
    Rs = value.(φ_R)
    a_Rs = value.(α_R)
    cpcs = value.(c_pc)
    C_bnds = value.(C_bnd)
    C_ins = value.(C_in)

    ts = supports.(α_R)
    # convert vector of [(number), (number)] to [number, number]
    ts = first.(ts)

    return Rs, a_Rs, cpcs, C_bnds, C_ins, Ls, ts
end

function default_ABX_params(;numeric_type=Float64)
    T = numeric_type
    
    γmax = 9.652768467
    νmax = 5
    Km = 0.03
    φ_0 = 0.65

    C_max = 1e-5

    # 800 -> 0.23/s
    kp = 800 # hr^-1 #0.02 * 60^2 
    ke = 35 * 60^2 # hr^-1

    #Kd = 2.7e-6 # M
    # on: 2.25e6, off: 3.78
    k_on = (0.034 + 0.0035) * 60 * 1e6 #1e6 # 2.5e6  # M^-1 hr^-1
    k_off = (0.084 - 0.021) * 60  #k_on*Kd  # M^-1 hr^-1

    k_ind = 1000

    f_prot = 0.55
    ρ_cell = 300e15  # fg/L
    m_R = 4.3e-3  # fg
    N_A = 6.022e23
    n_prot = (ρ_cell * f_prot) / (m_R * N_A)  # 63.7 uM
    m_CAT = 5.11e-4 # fg
    n_CAT = (ρ_cell * f_prot) / (m_CAT * N_A) # 536 uM
    Km_abx = 15e-6  # M

    params = (
        γmax, νmax, Km, φ_0, C_max, kp, ke,
        k_on, k_off, k_ind, n_prot, Km_abx, n_CAT
    )
    params = map(x -> T(x), params)
    abx = PrecursorABXBaseModel(params...)
    return abx
end

function default_Ant_params(;numeric_type=Float64)
    T = numeric_type
    αmax = T(0.0)
    αmin = T(0.0)
    ant = AbxAnticipationParams(αmax, αmin)
    return ant
end

function default_PcABXwAnt_params(;numeric_type=Float64)
    abx = default_ABX_params(numeric_type=numeric_type)
    ant = default_Ant_params(numeric_type=numeric_type)
    return PrecursorABXwAnticipation(abx, ant)
end

function default_PcMultiABXwAnt_params(n; numeric_type=Float64)
    abx = default_ABX_params(numeric_type=numeric_type)
    ants = [default_Ant_params(numeric_type=numeric_type) for i in 1:n]
    return PrecursorMultiABXwAnticipation(abx, ants)
end

function PrecursorABXConstantRiboBase(p::PrecursorABXBaseModel; phi_R=-1)
    if phi_R == -1
        # find the optimal phi_R in ss no ABX, given the current params
        p_noABX = mutate(p, C_max=0.0)
        phi_R = ABXfree_optimal_φ_R(p_noABX)
    end
    # copy params to the new struct
    abx = PrecursorABXConstantRiboBase(
        p.γmax, p.νmax, p.Km, p.φ_0, p.C_max, p.kp, p.ke,
        p.k_on, p.k_off, p.k_ind, p.n_prot, p.Km_abx, p.n_CAT, phi_R
    )
    return abx
end

function PrecursorABXConstantRiboWAnt(p::PrecursorABXwAnticipation; phi_R=-1)
    abx = PrecursorABXConstantRiboBase(p.abx, phi_R=phi_R)
    ant = p.ant
    return PrecursorABXConstantRiboWAnt(abx, ant)
end

function PcABXwAnt_constant_R_params()
    p = default_PcABXwAnt_params()
    p.abx.C_max = 0.0
    return p
end

function PcABXwAnt_params(kind::String)
    """
    	Generates prototypical parameters that represent
    	the maximal and minimal possible values

    	The anticipation parameters are set to zero and not varied
    """

    @assert kind in ("max", "min") "Expected kind:$kind to be 'max' or 'min'"

    is_max = kind == "max" ? true : false
    γmax = 9.652768467

    νmax = is_max ? 12 : 0.1
    Km = 0.03
    φ_0 = 0.65

    C_max = is_max ? 1e-3 : 1e-6

    kp = is_max ? 1 * 60^2 : 0.001 * 60^2
    ke = is_max ? 50 * 60^2 : 20 * 60^2

    Kd = 1e6
    k_on = is_max ? 1e7 : 1e4
    k_off = k_on / Kd

    k_ind = 1000

    f_prot = 0.55
    ρ_cell = 300e15
    m_R = 4.3e-3
    N_A = 6.022e23
    n_prot = (ρ_cell * f_prot) / (m_R * N_A)
    m_CAT = 5.11e-4
    n_CAT = (ρ_cell * f_prot) / (m_CAT * N_A)
    Km_abx = 15e-6

    αmax = 0.0
    αmin = 0.0

    abx = PrecursorABXBaseModel(
        γmax, νmax, Km, φ_0, C_max, kp, ke,
        k_on, k_off, k_ind, n_prot, Km_abx, n_CAT
    )
    ant = AbxAnticipationParams(αmax, αmin)
    return PrecursorABXwAnticipation(abx, ant)
end

function param_sweep_step(
    sym::Symbol, u0, params::PrecursorABXwAnticipation,
    jc::JumpConfig, p_range
)
    Tc = 0.8 * jc.n * jc.Δt
    GRs = zeros(length(p_range), Int(round(Tc * 2 / jc.Δt)))
    mut_params = [Dict(sym => p) for p in p_range]
    for (i, mut_dict) in enumerate(mut_params)
        bM = mutate(params; mut_dict...)
        opt_AR = ABX_optimizer_allOpt(bM)[end]
        respM = mutate(bM, αmax=opt_AR)
        @info "starting $i"
        states, jc_steppy = sim_ABX_step(u0, respM, jc, step_up=true, T_cut_off=Tc)
        global jc_step = jc_steppy

        GRs[i, :] = states[8]
    end

    return GRs, jc_step
end

function convert_dict(x::Dict{Symbol,Vector{T}}) where {T}
    [Dict(k => v[i] for (k, v) in x) for i in 1:length(first(values(x)))]
end

function create_all_combinations(x::Dict{Symbol,Vector{T}}) where {T}
    keys_vec = collect(keys(x))
    values_vec = collect(values(x))

    [Dict(zip(keys_vec, combo)) for combo in Iterators.product(values_vec...)]
end

function param_sweep_step(
    syms_dict::Dict, u0, params::PrecursorABXwAnticipation,
    jc::JumpConfig, synced::Bool=true
)

    Tc = 0.8 * jc.n * jc.Δt
    ref_range = first(values(syms_dict))
    GRs = zeros(length(ref_range), Int(round(Tc * 2 / jc.Δt)))

    if synced
        mut_params = convert_dict(syms_dict)
    else
        mut_params = create_all_combinations(syms_dict)
    end

    for (i, mut_dict) in enumerate(mut_params)
        bM = mutate(params; mut_dict...)
        opt_AR = ABX_optimizer_allOpt(bM)[end]
        respM = mutate(bM, αmax=opt_AR)
        @info "starting $i"
        states, jc_steppy = sim_ABX_step(u0, respM, jc, step_up=true, T_cut_off=Tc)
        global jc_step = jc_steppy

        GRs[i, :] = states[8]
    end

    return GRs, jc_step
end

function permeation(p::PrecursorABXwAnticipation, C_in)
    return p.abx.kp * (p.abx.C_max - C_in)
end

function degradation(p::PrecursorABXwAnticipation, C_in, φ_AR)
    @unpack_PrecursorABXBaseModel p.abx
    return n_CAT * ke * φ_AR * C_in / (C_in + Km_abx)
end

function equilibrated_C_bnd(p::PrecursorABXwAnticipation, C_in, φ_R)
    @unpack_PrecursorABXBaseModel p.abx
    return k_on * C_in * n_prot * φ_R / (k_off + k_on * C_in)
end

function binding(p::PrecursorABXwAnticipation, C_in, C_bnd, φ_R)
    @unpack_PrecursorABXBaseModel p.abx
    return k_on * C_in * (n_prot * φ_R - C_bnd)
end

function unbinding(p::PrecursorABXwAnticipation, C_bnd)
    @unpack_PrecursorABXBaseModel p.abx
    return k_off * C_bnd
end

function sim_inf2(
    u0, p::PrecursorABXwAnticipation, jc::JumpConfig;
    opt_AR=false, smoothness=10, AR_in_phi_F=false
)
    """
    sim_inf2 is a version of sim_inf 

    It additionally models the dynamics (and possible optimization) 
    of α_AR and φ_AR
    """
    if AR_in_phi_F
        sim_inf2_AR_in_phi_F(u0, p, jc; opt_AR=opt_AR, smoothness=smoothness)
    end
    @unpack_PrecursorABXBaseModel p.abx
    @unpack_AbxAnticipationParams p.ant
    @unpack_JumpConfig jc
    model = InfiniteModel(Ipopt.Optimizer)
    set_silent(model)

    T = Δt * n
    @infinite_parameter(model, t ∈ [0, T], num_supports = n)

    # dynamical variables
    vars = JuMP.@variables(model, begin
        0 <= φ_R <= 1 - φ_0, Infinite(t)
        0 <= φ_AR <= 1 - φ_0, Infinite(t)
        0 <= α_AR <= 1 - φ_0, Infinite(t)
        0 <= c_pc, Infinite(t)
        0 <= C_bnd, Infinite(t)
        0 <= C_in, Infinite(t)
        # optimization knob
        0 <= α_R <= 1 - φ_0, Infinite(t)
    end)
    # initial conditions
    @constraint(
        model,
        [i = 1:6],
        vars[i](0) == u0[i]
    )

    @expressions(model, begin
        # concentration of free ribosomes
        C_Rf, n_prot .* φ_R .- C_bnd
        # mass fraction of free ribosomes
        φ_Rf, φ_R .- C_bnd ./ n_prot
        # metabolic protein mass fraction
        φ_M, 1 - φ_0 .- φ_AR .- φ_R
        # translation rate
        γ, γmax .* c_pc ./ (c_pc .+ Km)
        # growth rate
        λ, γ .* φ_Rf
        # α_AR induction
        α_on, k_ind .* (1 .- α_AR ./ αmax)
        # α_AR suppresion
        α_off, -k_ind * (α_AR .- αmin)
    end)
    @constraint(model, φ_Rf .>= 0)

    @constraint(model, ∂(φ_R, t) == (α_R - φ_R) * λ)
    @constraint(model, ∂(c_pc, t) == νmax .* (1 - φ_0 .- φ_AR .- φ_R) .- (1 .+ c_pc) .* λ)
    @constraint(model, ∂(C_bnd, t) == k_on .* C_in .* C_Rf .- k_off .* C_bnd .- λ .* C_bnd)
    @constraint(model, ∂(C_in, t) == kp .* (C_max .- C_in) .+ k_off .* C_bnd .- k_on .* C_Rf .* C_in .- ke * φ_AR * n_CAT * C_in / (C_in + Km_abx) .- λ .* C_in)
    @constraint(model, ∂(φ_AR, t) == (α_AR - φ_AR) * λ)
    if !opt_AR
        if αmax != 0
            ABX_on = Int(C_max > 0)
            @constraint(model, ∂(α_AR, t) == ABX_on * α_on + (1 - ABX_on) * α_off)
        else
            @constraint(model, ∂(α_AR, t) == 0)
            @constraint(model, α_AR == 0)
        end
    end

    @objective(model, Max, ∫(λ, t) - smoothness * ∫(∂(α_R, t)^2, t) - smoothness * ∫(∂(α_AR, t)^2, t))

    optimize!(model)
    Ls = value.(λ)
    Rs = value.(φ_R)
    a_Rs = value.(α_R)
    ARs = value.(φ_AR)
    a_ARs = value.(α_AR)
    cpcs = value.(c_pc)
    C_bnds = value.(C_bnd)
    C_ins = value.(C_in)

    ts = supports.(α_R)
    # convert vector of [(number), (number)] to [number, number]
    ts = first.(ts)

    return Rs, a_Rs, ARs, a_ARs, cpcs, C_bnds, C_ins, Ls, ts
end

function sim_inf2_AR_in_phi_F(
    u0, p::PrecursorABXwAnticipation, jc::JumpConfig;
    opt_AR=false, smoothness=10
)
    """
    models Ribosomal-regulated or True optimal growth dynamics if
    φ_AR = φ_F * f_AR; φ_F = 1 - φ_0 - φ_R
    """
    @unpack_PrecursorABXBaseModel p.abx
    @unpack_AbxAnticipationParams p.ant
    @unpack_JumpConfig jc
    model = InfiniteModel(Ipopt.Optimizer)
    set_silent(model)

    T = Δt * n
    @infinite_parameter(model, t ∈ [0, T], num_supports = n)

    # dynamical variables
    vars = JuMP.@variables(model, begin
        0 <= φ_R <= 1 - φ_0, Infinite(t)
        0 <= φ_AR <= 1 - φ_0, Infinite(t)
        0 <= α_AR <= 1 - φ_0, Infinite(t)
        0 <= c_pc, Infinite(t)
        0 <= C_bnd, Infinite(t)
        0 <= C_in, Infinite(t)
        # optimization knob
        0 <= α_R <= 1 - φ_0, Infinite(t)
    end)
    # initial conditions
    @constraint(
        model,
        [i = 1:6],
        vars[i](0) == u0[i]
    )

    @expressions(model, begin
        # concentration of free ribosomes
        C_Rf, n_prot .* φ_R .- C_bnd
        # mass fraction of free ribosomes
        φ_Rf, φ_R .- C_bnd ./ n_prot
        # flexible protein mass fraction
        φ_F, 1 - φ_0 .- φ_R
        # metabolic protein mass fraction
        φ_M, φ_F - φ_AR
        # translation rate
        γ, γmax .* c_pc ./ (c_pc .+ Km)
        # growth rate
        λ, γ .* φ_Rf
        # α_AR induction
        α_on, k_ind .* (1 .- α_AR ./ αmax)
        # α_AR suppresion
        α_off, -k_ind * (α_AR .- αmin)
    end)
    @constraint(model, φ_Rf .>= 0)

    @constraint(model, ∂(φ_R, t) == (α_R - φ_R) * λ)
    @constraint(model, ∂(c_pc, t) == νmax .* (1 - φ_0 .- φ_AR .- φ_R) .- (1 .+ c_pc) .* λ)
    @constraint(model, ∂(C_bnd, t) == k_on .* C_in .* C_Rf .- k_off .* C_bnd .- λ .* C_bnd)
    @constraint(model, ∂(C_in, t) == kp .* (C_max .- C_in) .+ k_off .* C_bnd .- k_on .* C_Rf .* C_in .- ke * φ_AR * n_CAT * C_in / (C_in + Km_abx) .- λ .* C_in)
    @constraint(model, ∂(φ_AR, t) == (α_AR * φ_F - φ_AR) * λ)
    if !opt_AR
        if αmax != 0
            ABX_on = Int(C_max > 0)
            @constraint(model, ∂(α_AR, t) == ABX_on * α_on + (1 - ABX_on) * α_off)
        else
            @constraint(model, ∂(α_AR, t) == 0)
            @constraint(model, α_AR == 0)
        end
    else
        if C_max == 0
            @constraint(model, α_AR == αmin)
        end
    end

    @objective(model, Max, ∫(λ, t) - smoothness * ∫(∂(α_R, t)^2, t) - smoothness * ∫(∂(α_AR, t)^2, t))

    optimize!(model)
    Ls = value.(λ)
    Rs = value.(φ_R)
    a_Rs = value.(α_R)
    ARs = value.(φ_AR)
    a_ARs = value.(α_AR)
    cpcs = value.(c_pc)
    C_bnds = value.(C_bnd)
    C_ins = value.(C_in)
    #
    ts = supports.(α_R)
    # convert vector of [(number), (number)] to [number, number]
    ts = first.(ts)

    return Rs, a_Rs, ARs, a_ARs, cpcs, C_bnds, C_ins, Ls, ts
end

function simulate(
    u0, p::PrecursorABXwAnticipation, jc::JumpConfig;
    opt_AR=false, smoothness=10, degradation=false,
    AR_in_phi_F=true
)
    """
    Simulates the precursor model with anticipation in the presence of ABX
    with or without optimization α_AR and φ_AR (antibiotic resistance)

    Always returns the same number of outputs, but some may be zeros
    """
    sim_AR = false

    if length(u0) == 6
        if p.ant.αmax == 0
            sim_AR = false
            idxs = [1, 4, 5, 6]
            u0 = u0[idxs]
        else
            sim_AR = true
        end
    elseif length(u0) == 4
        sim_AR = false
    else
        throw(ArgumentError("Initial conditions must be of length 4 or 6"))
    end

    if degradation
        # simulate with degradation
        @warn "Overriding AR_in_phi_F to false, and sim_AR to false"
        out = sim_degrad(u0, p, jc; opt_AR=opt_AR, smoothness=smoothness)
    end
    if sim_AR
        out = sim_inf2(
            u0, p, jc; opt_AR=opt_AR,
            smoothness=smoothness, AR_in_phi_F=AR_in_phi_F
        )
    else
        # simulate without AR
        out_partial = sim_inf(u0, p, jc; smoothness=smoothness)
        # add zeros for AR
        filler = zeros(length(out_partial[1]))
        out = Vector{Vector{Float64}}(undef, 9)
        j = 1
        for i in 1:9
            # i ∈ {3,4} is α_AR and φ_AR,
            # which are set to zero in this case
            if i == 3 || i == 4
                out[i] = filler
            else
                out[i] = out_partial[j]
                j += 1
            end
        end
    end
    return out
end

function sim_ABX_step(
    u0, p::PrecursorABXwAnticipation, jc::JumpConfig;
    opt_AR=false, step_up=true, T_cut_off=4.0,
    smoothness=10, degradation=false, AR_in_phi_F=true
)
    """
    Simulate a step up (or down) in ABX concentration
    by using the global optimal control for α_R
    and truncating at T_cut_off to prevent the
    "prescient" response of the global optimal control
    """
    if T_cut_off > jc.n * jc.Δt
        throw(ArgumentError("T_cut_off must be less than the total simulation time"))
    end

    if step_up
        p0 = mutate(p.abx, C_max=0)
        p1 = mutate(p.abx, C_max=p.abx.C_max)
    else
        p0 = mutate(p.abx, C_max=p.abx.C_max)
        p1 = mutate(p.abx, C_max=0)
    end
    pM_0 = PrecursorABXwAnticipation(p0, p.ant)
    pM_1 = PrecursorABXwAnticipation(p1, p.ant)

    out0 = simulate(
        u0, pM_0, jc; opt_AR=opt_AR,
        smoothness=smoothness, degradation=degradation,
        AR_in_phi_F=AR_in_phi_F
    )
    ts = out0[end]
    in_T_range = ts .<= T_cut_off
    out0_trunc = collect(out0[i][in_T_range] for i in 1:length(out0))
    IC_idxs = [1, 3, 4, 5, 6, 7]
    u1 = [out0_trunc[i][end] for i in IC_idxs]
    out1 = simulate(
        u1, pM_1, jc; opt_AR=opt_AR,
        smoothness=smoothness, degradation=degradation,
        AR_in_phi_F=AR_in_phi_F
    )
    out1_trunc = collect(out1[i][in_T_range] for i in 1:length(out1))
    # combine the two simulations, while leaving out the time variable
    out_step = [vcat(out0_trunc[i], out1_trunc[i]) for i in 1:8]
    jc_step = mutate(jc, n=length(out_step[1]))
    return out_step, jc_step
end

# ~5s runtime
function simulate(u0, p::PrecursorABXBaseModel, jc::JumpConfig)
    @unpack_PrecursorABXBaseModel p
    @unpack_JumpConfig jc

    model = Model(Ipopt.Optimizer)
    set_silent(model)

    u = JuMP.@variables(model, begin
        0 <= φ_R[1:n] <= 1 - φ_0
        0 <= c_pc[1:n]
        0 <= C_bnd[1:n]
        0 <= C_in[1:n]
        0 <= M[1:n]
    end)
    for i in 1:length(u)
        fix(u[i][1], u0[i]; force=true)
    end
    @variable(model, 0 <= α_R[1:n] <= 1 - φ_0)
    @expressions(model, begin
        C_Rf, n_prot .* φ_R .- C_bnd
        φ_Rf, φ_R .- C_bnd ./ n_prot
        φ_M, 1 - φ_0 .- φ_R
        γ, γmax .* c_pc ./ (c_pc .+ Km)
        λ, γ .* φ_Rf
    end)
    @constraint(model, φ_Rf .>= 0)
    du = [@variable(model, [1:n]) for _ in 1:length(u0)]
    @constraints(model, begin
        du[1] .== (α_R .- φ_R) .* λ
        du[2] .== νmax .* (1 - φ_0 .- φ_R) .- (1 .+ c_pc) .* λ
        du[3] .== k_on .* C_in .* C_Rf .- k_off .* C_bnd .- λ .* C_bnd
        du[4] .== kp .* (C_max .- C_in) .+ k_off .* C_bnd .- k_on .* C_Rf .* C_in .- λ .* C_in
        du[5] .== M .* λ
    end)
    integration_JuMP!(model, u, du, jc, TrapezoidalIntegrationRule())
    @objective(model, Max, M[n])
    optimize!(model)
    @assert is_solved_and_feasible(model)
    Rs = value.(φ_R)
    a_Rs = value.(α_R)
    cpcs = value.(c_pc)
    C_bnds = value.(C_bnd)
    C_ins = value.(C_in)
    Ls = value.(λ)
    return Rs, a_Rs, cpcs, C_bnds, C_ins, Ls
end

function integration_JuMP!(model, u, du, jc, ::TrapezoidalIntegrationRule)
    @constraint(
        model,
        [t in 2:jc.n, k in 1:length(u)],
        u[k][t] == u[k][t-1] + 0.5 * jc.Δt * (du[k][t-1] + du[k][t]),
    )
    return
end

function φ_R_heuristic(
    cpc, p::PrecursorABXBaseModel;
    V=2, K=0.8, φ_R0=0.05
)
    @unpack_PrecursorABXBaseModel p
    φ_R = φ_R0 + V * (1 - φ_0 - φ_R0) * cpc / (cpc + K)
    return φ_R
end

function growth_dynx_param(V, K)
    f(du, u, p::PrecursorABXwAnticipation, t) = growth_dynamics(du, u, p, t; V=V, K=K)
    return f
end

function calculate_total_C_bnd(
    φ_R, C_bnd_vec, n_prot
)
    n_ARs = length(C_bnd_vec)
        
    # Step 1: Calculate the probability of each ribosome being bound 
    # by each individual antibiotic
    p_bound_by_abx = [C_bnd_vec[i] / (n_prot * φ_R) for i in 1:n_ARs]

    # Step 2: Calculate probability of a ribosome being completely free
    # P(free) = P(not bound by ABX 1) * P(not bound by ABX 2) * 
    # ... * P(not bound by ABX n)
    p_completely_free = prod(1.0 .- p_bound_by_abx)

    # Step 3: Probability of being bound by at least one antibiotic
    p_bound_by_any = 1.0 - p_completely_free

    # Step 4: Convert back to mass fraction of bound ribosomes
    total_C_bnd = p_bound_by_any * φ_R * n_prot

    return total_C_bnd
end

function growth_dynamics_ss(du, u, p::PrecursorMultiABXwAnticipation, t)
    c_pc = u[1]

    # Extract vectors for each antibiotic
    n_ARs = (length(u) - 1) ÷ 2
    C_bnd_vec = u[2:1+n_ARs]
    C_in_vec = u[2+n_ARs:1+2*n_ARs]

    @unpack_PrecursorABXBaseModel p.abx
    α_maxs = [p.ant[i].αmax for i in 1:n_ARs]
    α_mins = [p.ant[i].αmin for i in 1:n_ARs]

    # Heuristic for allocation
    φ_R = φ_R_heuristic(c_pc, p.abx)
    φ_F = 1 - φ_0 - φ_R
    γ = monod(γmax, Km)
    total_C_bnd = calculate_total_C_bnd(φ_R, C_bnd_vec, n_prot)
    φ_Rf = φ_R - total_C_bnd / n_prot
    C_Rf_vec = n_prot * φ_R .- C_bnd_vec

    φ_AR_vec = α_maxs
    φ_M = φ_F - sum(φ_AR_vec)

    λ = γ(c_pc) * φ_Rf
    du[1] = dc_pc = νmax * φ_M - γ(c_pc) * φ_Rf - c_pc * λ
    # C_bnd dynamics
    du[2:1+n_ARs] = k_on .* C_in_vec .* C_Rf_vec .- k_off .* C_bnd_vec .- λ .* C_bnd_vec
    # C_in dynamics
    du[2+n_ARs:1+2*n_ARs] = (
        kp .* (C_max .- C_in_vec)
        .- ke .* φ_AR_vec .* C_in_vec .* n_CAT ./ (C_in_vec .+ Km_abx)
        .+ k_off .* C_bnd_vec
        .- k_on .* C_Rf_vec .* C_in_vec
        .- λ .* C_in_vec
    )
end

function growth_dynamics(
    du, u, p::PrecursorMultiABXwAnticipation, t;
    V=2, K=0.8
)
    # Extract state variables
    φ_R = u[1]
    c_pc = u[2]

    # Extract vectors for each antibiotic
    n_ARs = (length(u) - 2) ÷ 4
    C_bnd_vec = u[3:2+n_ARs]
    C_in_vec = u[3+n_ARs:2+2*n_ARs]
    α_AR_vec = u[3+2*n_ARs:2+3*n_ARs]
    φ_AR_vec = u[3+3*n_ARs:2+4*n_ARs]

    @unpack_PrecursorABXBaseModel p.abx
    α_maxs = [p.ant[i].αmax for i in 1:n_ARs]
    α_mins = [p.ant[i].αmin for i in 1:n_ARs]

    # Heuristic for allocation
    α_R = φ_R_heuristic(c_pc, p.abx; V=V, K=K)

    # Calculate total allocation to antibiotic resistance
    total_φ_AR = sum(φ_AR_vec)

    φ_F = 1 - φ_0 - φ_R
    γ = monod(γmax, Km)
    
    # 
    total_C_bnd = calculate_total_C_bnd(φ_R, C_bnd_vec, n_prot)
    φ_Rf = φ_R - total_C_bnd / n_prot
    C_Rf_vec = n_prot * φ_R .- C_bnd_vec

    φ_M = φ_F - total_φ_AR
    λ = γ(c_pc) * φ_Rf

    # First two state variables
    du[1] = (α_R - φ_R) * λ
    du[2] = νmax * φ_M - γ(c_pc) * φ_Rf - c_pc * λ

    # Get all C_max values at once
    C_max_values = [C_max(i, t) for i in 1:n_ARs]

    # C_bnd dynamics
    du[3:2+n_ARs] = k_on .* C_in_vec .* C_Rf_vec .- k_off .* C_bnd_vec .- λ .* C_bnd_vec

    # C_in dynamics
    du[3+n_ARs:2+2*n_ARs] = (
        kp .* (C_max_values .- C_in_vec)
        .- ke .* φ_AR_vec .* C_in_vec .* n_CAT ./ (C_in_vec .+ Km_abx)
        .+ k_off .* C_bnd_vec
        .- k_on .* C_Rf_vec .* C_in_vec
        .- λ .* C_in_vec
    )

    # α_AR dynamics - requires conditional logic, so use a loop
    for i in 1:n_ARs
        if α_maxs[i] != 0
            reserve = α_mins[i]
            α_on = (1 - α_AR_vec[i] / α_maxs[i]) * k_ind
            α_off = -k_ind * (α_AR_vec[i] - reserve)
            indicator = Int(C_max_values[i] > 0)
            du[2+2*n_ARs+i] = α_on * indicator + α_off * (1 - indicator)
        else
            du[2+2*n_ARs+i] = 0
        end
    end

    # φ_AR dynamics
    du[3+3*n_ARs:2+4*n_ARs] = (α_AR_vec .- φ_AR_vec) .* λ
end

function growth_dynamics(
    du, u, p::PrecursorABXwAnticipation, t;
    steady_state=false
)
    # for C_max as a function of time
    if steady_state
        return growth_dynamics_ss(du, u, p, t)
    end

    φ_R, c_pc, C_bnd, C_in, α_AR, φ_AR = u

    @unpack_PrecursorABXBaseModel p.abx
    @unpack αmax, αmin = p.ant

    # the heuristic should set the allocation,
    # NOT the instantaneous, realized φ_R
    α_R = φ_R_heuristic(c_pc, p.abx)
    φ_F = 1 - φ_0 - φ_R
    γ = monod(γmax, Km)
    φ_Rf = φ_R - C_bnd / n_prot
    C_Rf = n_prot * φ_R - C_bnd

    # α_AR is fn of αmin/max
    if αmax != 0
        reserve = αmin
        α_on = (1 - α_AR / αmax) * k_ind
        α_off = -k_ind * (α_AR .- reserve)
        indicator = Int(C_max(t) > 0)
        dα_AR = α_on * indicator .+ α_off * (1 - indicator)
    else
        α_AR = 0
        dα_AR = 0
    end

    φ_M = φ_F - φ_AR
    λ = γ(c_pc) * φ_Rf
    du[1] = dφ_R = (α_R - φ_R) * λ
    du[2] = dc_pc = νmax * φ_M - γ(c_pc) * φ_Rf - c_pc * λ
    du[3] = dC_bnd = k_on * C_in * C_Rf - k_off * C_bnd - λ * C_bnd
    du[4] = dC_in = (
        kp * (C_max(t) - C_in)
        - ke * φ_AR * C_in * n_CAT / (C_in + Km_abx)
        + k_off * C_bnd
        - k_on * C_Rf * C_in
        - λ * C_in
    )
    du[5] = dα_AR
    du[6] = dφ_AR = (α_AR - φ_AR) * λ
end

function growth_dynamics(
    du, u, p::PcABXAntWPerturbation, t;
)
    # for C_max controlled by a PeriodicCallback
    φ_R, c_pc, C_bnd, C_in, α_AR, φ_AR = u

    @unpack_PrecursorABXBaseModel p.abx
    @unpack αmax, αmin = p.ant
    # override C_max from p.abx
    C_max = p.perturbation.current_value

    # the heuristic should set the allocation,
    # NOT the instantaneous, realized φ_R
    α_R = φ_R_heuristic(c_pc, p.abx)
    φ_F = 1 - φ_0 - φ_R
    γ = monod(γmax, Km)
    φ_Rf = φ_R - C_bnd / n_prot
    C_Rf = n_prot * φ_R - C_bnd

    # α_AR is fn of αmin/max
    if αmax != 0
        reserve = αmin
        α_on = (1 - α_AR / αmax) * k_ind
        α_off = -k_ind * (α_AR .- reserve)
        indicator = Int(C_max > 0)
        dα_AR = α_on * indicator .+ α_off * (1 - indicator)
    else
        α_AR = 0
        dα_AR = 0
    end

    φ_M = φ_F - φ_AR
    λ = γ(c_pc) * φ_Rf
    du[1] = dφ_R = (α_R - φ_R) * λ
    du[2] = dc_pc = νmax * φ_M - γ(c_pc) * φ_Rf - c_pc * λ
    du[3] = dC_bnd = k_on * C_in * C_Rf - k_off * C_bnd - λ * C_bnd
    du[4] = dC_in = (
        kp * (C_max - C_in)
        - ke * φ_AR * C_in * n_CAT / (C_in + Km_abx)
        + k_off * C_bnd
        - k_on * C_Rf * C_in
        - λ * C_in
    )
    du[5] = dα_AR
    du[6] = dφ_AR = (α_AR - φ_AR) * λ
end

function growth_step(
    du, u, p::PrecursorABXwAnticipation, t;
)
    # for when C_max is a number
    φ_R, c_pc, C_bnd, C_in, α_AR, φ_AR = u

    @unpack_PrecursorABXBaseModel p.abx
    @unpack αmax, αmin = p.ant

    # the heuristic should set the allocation,
    # NOT the instantaneous, realized φ_R
    α_R = φ_R_heuristic(c_pc, p.abx)
    φ_F = 1 - φ_0 - φ_R
    γ = monod(γmax, Km)
    φ_Rf = φ_R - C_bnd / n_prot
    C_Rf = n_prot * φ_R - C_bnd

    # α_AR is fn of αmin/max
    if αmax != 0
        reserve = αmin
        α_on = (1 - α_AR / αmax) * k_ind
        α_off = -k_ind * (α_AR .- reserve)
        indicator = Int(C_max > 0)
        dα_AR = α_on * indicator .+ α_off * (1 - indicator)
    else
        α_AR = 0
        dα_AR = 0
    end

    φ_M = φ_F - φ_AR
    λ = γ(c_pc) * φ_Rf
    du[1] = dφ_R = (α_R - φ_R) * λ
    du[2] = dc_pc = νmax * φ_M - γ(c_pc) * φ_Rf - c_pc * λ
    du[3] = dC_bnd = k_on * C_in * C_Rf - k_off * C_bnd - λ * C_bnd
    du[4] = dC_in = (
        kp * (C_max - C_in)
        - ke * φ_AR * C_in * n_CAT / (C_in + Km_abx)
        + k_off * C_bnd
        - k_on * C_Rf * C_in
        - λ * C_in
    )
    du[5] = dα_AR
    du[6] = dφ_AR = (α_AR - φ_AR) * λ
end

function growth_dynamics_ss(du, u, p::PrecursorABXwAnticipation, t)
    c_pc, C_bnd, C_in = u

    @unpack_PrecursorABXBaseModel p.abx
    @unpack αmax, αmin = p.ant

    φ_R = φ_R_heuristic(c_pc, p.abx)
    φ_F = 1 - φ_0 - φ_R
    γ = monod(γmax, Km)
    φ_Rf = φ_R - C_bnd / n_prot
    C_Rf = n_prot * φ_R - C_bnd

    φ_AR = αmax
    # fAR is fn of αmin/max -> dα/dt gives α gives fAR
    φ_M = φ_F - αmax

    λ = γ(c_pc) * φ_Rf
    du[1] = dc_pc = νmax * φ_M - γ(c_pc) * φ_Rf - c_pc * λ
    du[2] = dC_bnd = k_on * C_in * C_Rf - k_off * C_bnd - λ * C_bnd
    du[3] = dC_in = (
        kp * (C_max - C_in)
        -
        ke * φ_AR * C_in * n_CAT / (C_in + Km_abx)
        +
        k_off * C_bnd
        -
        k_on * C_Rf * C_in
        -
        λ * C_in
    )
end

function growth_dynamics_multiABX(
    du, u, p::PrecursorABXwAnticipation, t;
    steady_state=false, n=2
)
    φ_R, c_pc, C_bnd, C_in, AR_variables = u
    α_ARs = AR_variables[1:n]
    φ_ARs = AR_variables[n+1:2*n]

    @unpack_PrecursorABXBaseModel p.abx
    @unpack αmax, αmin = p.ant

    # the heuristic should set the allocation,
    # NOT the instantaneous, realized φ_R
    α_R = φ_R_heuristic(c_pc, p.abx)
    φ_F = 1 - φ_0 - φ_R
    γ = monod(γmax, Km)
    φ_Rf = φ_R - C_bnd / n_prot
    C_Rf = n_prot * φ_R - C_bnd

    # α_AR is fn of αmin/max
    if αmax != 0
        reserve = αmin
        α_on = (1 - α_AR / αmax) * k_ind
        α_off = -k_ind * (α_AR - reserve)
        indicator = Int(C_max(t) > 0)
        dα_AR = α_on * indicator + α_off * (1 - indicator)
    else
        α_AR = 0
        dα_AR = 0
    end

    φ_M = φ_F - φ_AR
    λ = γ(c_pc) * φ_Rf
    du[1] = dφ_R = (α_R - φ_R) * λ
    du[2] = dc_pc = νmax * φ_M - γ(c_pc) * φ_Rf - c_pc * λ

    for i in 0:n-1
        du[3+i*4] = dC_bnd = k_on * C_in * C_Rf - k_off * C_bnd - λ * C_bnd
        du[4+i*4] = dC_in = (
            # C_max must be a function of time
            kp * (C_max(t) - C_in)
            - ke * φ_AR * C_in * n_CAT / (C_in + Km_abx)
            + k_off * C_bnd
            - k_on * C_Rf * C_in
            - λ * C_in
        )
        du[5+i*4] = dα_AR
        du[6+i*4] = dφ_AR = (α_AR - φ_AR) * λ
    end
end

function toggle_perturbation_on!(integrator)
    p = integrator.p.perturbation
    p.current_value = p.amplitude
end

function toggle_perturbation_off!(integrator)
    p = integrator.p.perturbation
    p.current_value = 0.0
end

function maclaurin(f, x, n)
    ddx = Symbolics.Differential(x)
    terms = Vector{typeof(x)}(undef, n + 1)
    terms[1] = f
    for i in 1:n
        terms[i+1] = expand_derivatives((ddx^i)(f)) * x^i / factorial(i)
    end
    return sum(terms)
end

function find_nu(φ_R0, λ, m::PrecursorABXwAnticipation)
    @unpack_PrecursorABXBaseModel m.abx
    """ from eqn 20 in flux parity methods, solving for νmax"""
    # best value ~2.42 for Deris. Mismatch bc φR = 0.11 and λ = 0.583
    # is on the outer edge of the "barrel" of data
    φ_M = 1 - φ_R0 - φ_0
    return (λ + Km * λ^2 / (γmax * φ_R0 - λ)) / φ_M
end

function find_nu_heur(φ_R, λ, m::PrecursorABXwAnticipation)
    @unpack_PrecursorABXBaseModel m.abx
    """ from equation for cpc in heuristic model"""
    φ_M = 1 - φ_R - φ_0
    V = 2
    K = 0.8
    φ_R0 = 0.05
    cpc = K * (φ_R - φ_R0) / (V * (1 - φ_0 - φ_R0) - (φ_R - φ_R0))
    return λ * (1 + cpc) / φ_M
end

function compute_GR_vs_ABX_Dai_curves(
    φR_series, λ_series, abx_series, p::PrecursorABXwAnticipation; n=51,
    opt=ABX_optimizer_v2, opt_kwargs=Dict()
)
    @unpack_PrecursorABXBaseModel p.abx

    # find appropriate ν for each growth media
    nu = find_nu(φR_series[1], λ_series[1], p)

    sim_abx_range = collect(10.0 .^ range(-7, -4.5, length=n))
    # add a zero to the beginning of the range
    sim_abx_range = vcat(0, sim_abx_range)
    # this lengthens the range by one, so we adjust
    n += 1
    simulated_states = Matrix(undef, n, 2)
    for i in 1:n
        C_max = sim_abx_range[i]
        base_model = mutate(
            p.abx, νmax=nu,
            C_max=C_max,
        )
        m = PrecursorABXwAnticipation(base_model, p.ant)
        state = opt(m; opt_kwargs...)
        simulated_states[i, :] = [state[1], state[2]]
    end
    return simulated_states, sim_abx_range
end

function lazy_copy_GR_vs_ABX_Dai_curves(
    φR_series, λ_series, abx_series, p::PrecursorABXwAnticipation
)
    @unpack_PrecursorABXBaseModel p.abx

    # find appropriate ν for each growth media
    nu = find_nu(φR_series[1], λ_series[1], p)

    n = length(abx_series)
    simulated_states = Matrix(undef, n, 2)
    for i in 1:n
        C_max = abx_series[i]
        base_model = mutate(
            p.abx, νmax=nu,
            C_max=C_max,
        )
        m = PrecursorABXwAnticipation(base_model, p.ant)
        state = ABX_optimizer_v2(m)
        simulated_states[i, :] = [state[1], state[2]]
    end
    return simulated_states
end

function sweep_ss(
    ::SingleSolver, φR0, λ0, abx_series,
    p::PrecursorABXwAnticipation;
    opt=ABX_optimizer_v2,
)
    @unpack_PrecursorABXBaseModel p.abx

    # find appropriate ν for each growth media
    nu = find_nu(φR0, λ0, p)

    n = length(abx_series)
    seven_soln_solvers = [
        homotopy_continuation_SS_solver,
        heuristic_solver, dynamicSS_solver_gen(),
        heuristic_solver_coR, abxOpt_solver, allOpt_solver,
        abxOpt_coR_solver,
    ]
    if opt == ABX_optimizer_v2
        solver_soln_size = 5
        # check if opt is in the list of seven_soln_solvers
    elseif opt in seven_soln_solvers
        solver_soln_size = 7
    else
        throw(ArgumentError("Solver '$opt' not recognized"))
    end
    simulated_states = Matrix(undef, n, solver_soln_size)
    for i in 1:n
        C_max = abx_series[i]
        base_model = mutate(
            p.abx, νmax=nu,
            C_max=C_max,
        )
        m = PrecursorABXwAnticipation(base_model, p.ant)
        state = opt(m)
        simulated_states[i, :] .= state
    end
    return simulated_states
end

function param_sweep_ss(
    sym_dict::Dict, params::PrecursorABXwAnticipation;
    opt=ABX_optimizer_v2, opt_kwargs=Dict()
)
    mut_params = convert_dict(sym_dict)
    all_states = []
    for mut_dict in mut_params
        bM = mutate(params; mut_dict...)
        states = opt(bM; opt_kwargs...)
        push!(all_states, states)
    end
    return all_states
end

function dSS_solver(p::PrecursorABXwAnticipation, solver=Rodas4P(); T=Float64)
    u0 = T.([0.5, 0.0, 0.0])
    prob = SteadyStateProblem(growth_dynamics_ss, u0, p)
    sol = solve(prob, DynamicSS(solver))
    cpc = sol[1]
    C_bnd = sol[2]
    C_in = sol[3]
    φR = φ_R_heuristic(cpc, p.abx)
    λ = monod(p.abx.γmax, p.abx.Km)(cpc) * (φR - C_bnd / p.abx.n_prot)
    φ_F = 1 - p.abx.φ_0 - φR
    φ_AR = φ_F * p.ant.αmax
    φ_M = φ_F * (1 - p.ant.αmax)
    out = [φR, λ, cpc, C_bnd, C_in, φ_AR, φ_M]
    return out
end

function dSS_solver(p::PrecursorMultiABXwAnticipation, solver=Rodas4P(); n = 2)
    u0 = zeros(1 + 2 * n)
    u0[1] = 0.5
    prob = SteadyStateProblem(growth_dynamics_ss, u0, p)
    sol = solve(prob, DynamicSS(solver))
    cpc = sol[1]
    C_bnd_vec = sol[n:3+n]
    φR = φ_R_heuristic(cpc, p.abx)
    total_C_bnd = calculate_total_C_bnd(φR, C_bnd_vec, p.abx.n_prot)
    λ = monod(p.abx.γmax, p.abx.Km)(cpc) * (φR - total_C_bnd / p.abx.n_prot)
    φ_F = 1 - p.abx.φ_0 - φR
    α_maxes = [p.ant[i].αmax for i in 1:n]
    φ_AR_total = φ_F * sum([p.ant[i].αmax for i in 1:n])
    φ_AR_vec = φ_F .* [p.ant[i].αmax for i in 1:n]
    φ_M = φ_F - φ_AR_total
    C_in_vec = sol[2+n:1+2*n]
    out = [φR, λ, cpc, C_bnd_vec..., C_in_vec..., φ_AR_vec..., φ_M]
    return out
end

function full_state_to_u0(state; α_AR=0)
    return [state[1], state[3], state[4], state[5], α_AR, state[6]]
end

function full_state_to_u0_multiABX(state; α_ARs=nothing)
    n_ARs = (length(state) - 4) ÷ 3
    φ_ARs = state[4+2*n_ARs:3+3*n_ARs]
    C_bnd_vec = state[4:3+n_ARs]
    C_in_vec = state[4+n_ARs:3+2*n_ARs]
    if α_ARs == nothing
        α_ARs = zeros(n_ARs)
    end
    return [state[1], state[3], C_bnd_vec..., C_in_vec..., α_ARs..., φ_ARs...]
end

function dynamicSS_solver_gen(solver=Rodas4P())
    return dSS_solver
end

function homotopy_continuation_SS_solver(p::PrecursorABXwAnticipation)
    @unpack_PrecursorABXBaseModel p.abx
    fAR = p.ant.αmax
    HC.@var cpc C_bnd C_in

    # parameters from the empirically fit φ_R vs λ curve
    k = 0.8
    V = 2
    φR_0 = 0.05

    # polynomials derived from the steady-state equations
    # for cpc, C_bnd, and C_in
    # computed in Mathematica and pasted here
    f1 = (
        (-1 + fAR) * k * Km * νmax * (-1 + φ_0 + φR_0) +
        cpc^2 * (-2 * γmax + (C_bnd * (1 + k) * γmax) / n_prot - νmax +
                 fAR * νmax + 2 * γmax * φ_0 + νmax * φ_0 -
                 fAR * νmax * φ_0 + γmax * φR_0 -
                 k * γmax * φR_0 + νmax * φR_0 - fAR * νmax * φR_0) +
        cpc * ((C_bnd * k * γmax) / n_prot + k * νmax - fAR * k * νmax - Km * νmax +
               fAR * Km * νmax - k * νmax * φ_0 + fAR * k * νmax * φ_0 +
               Km * νmax * φ_0 - fAR * Km * νmax * φ_0 - k * γmax * φR_0 -
               k * νmax * φR_0 + fAR * k * νmax * φR_0 + Km * νmax * φR_0 -
               fAR * Km * νmax * φR_0) +
        cpc^3 * ((C_bnd * γmax) / n_prot + γmax * (-2 + 2 * φ_0 + φR_0))
    )
    f2 = (
        -C_bnd * k * Km * k_off + C_in * (-C_bnd * k * Km * k_on + k * Km * k_on * n_prot * φR_0) +
        cpc^2 * ((C_bnd^2 * γmax) / n_prot +
                 C_bnd * (-k_off - 2 * γmax + 2 * γmax * φ_0 + γmax * φR_0) +
                 C_in * (-C_bnd * k_on - k_on * n_prot * (-2 + 2 * φ_0 + φR_0))) +
        cpc * ((C_bnd^2 * k * γmax) / n_prot +
               C_bnd * (-k * k_off - Km * k_off - k * γmax * φR_0) +
               C_in * (-C_bnd * (k + Km) * k_on +
                       k_on * n_prot * (2 * Km - 2 * Km * φ_0 + k * φR_0 - Km * φR_0)))
    )
    f3 = (
        C_bnd * k * Km * Km_abx * k_off + C_max * k * Km * Km_abx * kp +
        C_in^2 * (C_bnd * k * Km * k_on - k * Km * (kp + k_on * n_prot * φR_0)) +
        C_in * (C_bnd * k * Km * (k_off + Km_abx * k_on) +
                k * Km * (C_max * kp - Km_abx * kp - fAR * ke * n_CAT + fAR * ke * n_CAT * φ_0 +
                          fAR * ke * n_CAT * φR_0 - Km_abx * k_on * n_prot * φR_0)) +
        cpc^2 * (C_bnd * Km_abx * k_off + C_max * Km_abx * kp +
                 C_in^2 * (-kp - 2 * k_on * n_prot - 2 * γmax + (
                               C_bnd * (k_on * n_prot + γmax)) / n_prot + 2 * k_on * n_prot * φ_0 +
                           2 * γmax * φ_0 + k_on * n_prot * φR_0 + γmax * φR_0) +
                 C_in * (C_max * kp - Km_abx * kp + fAR * ke * n_CAT - 2 * Km_abx * k_on * n_prot -
                         2 * Km_abx * γmax + (
                             C_bnd * (k_off * n_prot + Km_abx * k_on * n_prot + Km_abx * γmax)) / n_prot -
                         fAR * ke * n_CAT * φ_0 + 2 * Km_abx * k_on * n_prot * φ_0 +
                         2 * Km_abx * γmax * φ_0 - fAR * ke * n_CAT * φR_0 +
                         Km_abx * k_on * n_prot * φR_0 + Km_abx * γmax * φR_0)) +
        cpc * (C_bnd * (k + Km) * Km_abx * k_off + C_max * (k + Km) * Km_abx * kp +
               C_in^2 * (-k * kp - Km * kp - 2 * Km * k_on * n_prot + (
                             C_bnd * (k * k_on * n_prot + Km * k_on * n_prot + k * γmax)) / n_prot +
                         2 * Km * k_on * n_prot * φ_0 - k * k_on * n_prot * φR_0 +
                         Km * k_on * n_prot * φR_0 - k * γmax * φR_0) +
               C_in * (C_max * k * kp + C_max * Km * kp - k * Km_abx * kp - Km * Km_abx * kp -
                       fAR * k * ke * n_CAT + fAR * ke * Km * n_CAT - 2 * Km * Km_abx * k_on * n_prot + (
                           C_bnd * (k * k_off * n_prot + Km * k_off * n_prot + k * Km_abx * k_on * n_prot +
                                    Km * Km_abx * k_on * n_prot + k * Km_abx * γmax)) / n_prot +
                       fAR * k * ke * n_CAT * φ_0 - fAR * ke * Km * n_CAT * φ_0 +
                       2 * Km * Km_abx * k_on * n_prot * φ_0 + fAR * k * ke * n_CAT * φR_0 -
                       fAR * ke * Km * n_CAT * φR_0 - k * Km_abx * k_on * n_prot * φR_0 +
                       Km * Km_abx * k_on * n_prot * φR_0 - k * Km_abx * γmax * φR_0))
    )
    F = HC.System([f1, f2, f3])
    HC_results = HC.solve(F)
    real_results = HC.real_solutions(HC_results)
    # reformat to matrix
    real_rezs = Matrix(hcat(real_results...)')
    positive = vec(all(real_rezs .> 0, dims=2))
    valid_results = real_rezs[positive, :]

    # for each valid result, compute GR and φR
    num_solns = size(valid_results)[1]
    ss_states = zeros(size(valid_results)[1], 7)
    for i in 1:num_solns
        cpc, C_bnd, C_in = valid_results[i, :]
        φR = φ_R_heuristic(cpc, p.abx)
        φ_F = 1 - p.abx.φ_0 - φR
        φ_AR = φ_F * fAR
        φ_M = φ_F * (1 - fAR)
        λ = monod(p.abx.γmax, p.abx.Km)(cpc) * (φR - C_bnd / p.abx.n_prot)
        ss_states[i, :] = [φR, λ, cpc, C_bnd, C_in, φ_AR, φ_M]
    end
    return ss_states
end

function bifurcation_SS_solver(p0::PrecursorABXwAnticipation)
    @unpack_PrecursorABXBaseModel p0.abx
    x0 = [0.0, 0.0, 0.0]
    p0 = mutate(p0, C_max=0.0)

    f(x, par) = core_ss_eqns(x, par)
    growth_rate = (x, par; k...) -> (φ_R_heuristic(x[1], p0.abx) - x[2] / n_prot) * monod(γmax, Km)(x[1])
    record = (x, p; k...) -> (λ=growth_rate(x, p; k), cpc=x[1])
    prob = BK.BifurcationProblem(f, x0, p0, (@optic _.abx.C_max); record_from_solution=record)
    br = BK.continuation(prob, BK.PALC(), BK.ContinuationPar(
        n_inversion=2, detect_bifurcation=3, max_steps=10000, dsmin=1e-6,
        dsmax=1e-4, ds=1e-6, p_max=1.5e-3, p_min=0.0
    ))
    return br, prob
end

function alt_bifurcation_SS_solver(p0::PrecursorABXwAnticipation)
    """ φ_AR co-regulated with φ_R """
    @unpack_PrecursorABXBaseModel p0.abx
    x0 = [0.0, 0.0, 0.0]
    p0 = mutate(p0, C_max=0.0)
    f(x, par) = alt_core_ss_eqns(x, par)
    growth_rate = (x, par; k...) -> (φ_R_heuristic(x[1], p0.abx) - x[2] / n_prot) * monod(γmax, Km)(x[1])
    record = (x, p; k...) -> (λ=growth_rate(x, p; k), cpc=x[1])
    prob = BK.BifurcationProblem(f, x0, p0, (@optic _.abx.C_max); record_from_solution=record)
    br = BK.continuation(prob, BK.PALC(), BK.ContinuationPar(
        n_inversion=2, detect_bifurcation=3, max_steps=10000, dsmin=1e-6,
        dsmax=1e-4, ds=1e-6, p_max=1.5e-3, p_min=0.0
    ))
    return br, prob
end

function bifurc_soln_to_full_state(
    br, p::PrecursorABXwAnticipation;
    only_phi=true, both_branches=true
)
    point_types = getfield.(br.specialpoint, :type)
    n_branches = -1
    if :bp in point_types
        # assume we have two breakpoints
        # which corresponds to 3 regions in the
        # following order:
        # stable, bp1, unstable, bp2, stable
        @assert(sum(point_types .== :bp) == 2)
        n_branches = 2
    else
        # there is no bifurcation
        @assert(length(point_types) == 1)
        @assert(point_types[1] == :endpoint)
        n_branches = 1
    end

    if n_branches == 2
        bp1_idx = br.specialpoint[1].idx
        branch1 = getfield.(br.sol[1:bp1_idx-1], :x)
        branch1 = Matrix(hcat(branch1...)')
        abx_series1 = getfield.(br.sol[1:bp1_idx-1], :p)

        bp2_idx = br.specialpoint[2].idx
        @assert(bp2_idx > bp1_idx)
        branch2 = getfield.(br.sol[bp2_idx:end], :x)
        branch2 = Matrix(hcat(branch2...)')
        abx_series2 = getfield.(br.sol[bp2_idx:end], :p)
        abx_series = [abx_series1, abx_series2]
        branches = [branch1, branch2]
        if !both_branches
            # return the max growth from either branch
            overlap = (abx_series2[1], abx_series1[end])
            b1_overlap_idxs = abx_series1 .>= overlap[1]
            branch1_overlap = branch1[b1_overlap_idxs, :]
            b2_overlap_idxs = abx_series2 .<= overlap[2]
            branch2_overlap = branch2[b2_overlap_idxs, :]
            # assume the first branch has a higher growth rate
            b2_nonOverlap_idxs = abx_series2 .> overlap[2]
            b2_non_overlap = branch2[b2_nonOverlap_idxs, :]
            branch = vcat(branch1, b2_non_overlap)
            abx_series = vcat(abx_series1, abx_series2[b2_nonOverlap_idxs])
            branches = [branch]
        end
    else
        branch = getfield.(br.sol, :x)
        branch = Matrix(hcat(branch...)')
        abx_series = getfield.(br.sol, :p)
        branches = [branch]
    end

    full_states = []
    for branch in branches
        cpcs = branch[:, 1]
        C_bnds = branch[:, 2]
        φRs = φ_R_heuristic.(cpcs, p.abx)
        φ_Fs = 1 - p.abx.φ_0 .- φRs
        φ_ARs = φ_Fs .* p.ant.αmax
        φ_Ms = φ_Fs .* (1 - p.ant.αmax)
        λs = monod(p.abx.γmax, p.abx.Km).(cpcs) .* (φRs .- C_bnds ./ p.abx.n_prot)
        if only_phi
            push!(full_states, hcat(φRs, φ_ARs, φ_Ms))
        else
            push!(full_states, hcat(φRs, λs, cpcs, C_bnds, φ_ARs, φ_Ms))
        end
    end
    return full_states, abx_series
end

function compute_GR_vs_pR_Dai_curves(
    φR_series, λ_series, abx_series, p::PrecursorABXwAnticipation; num_sim_pts=51
)
    @unpack_PrecursorABXBaseModel p.abx

    # find appropriate ν for each growth media
    nu = find_nu(φR_series[1], λ_series[1], p)

    n = length(abx_series)
    simulated_states = Matrix(undef, n, 2)
    for i in 1:n
        C_max = abx_series[i]
        pcm = PrecursorABXConstantRiboBase(
            γmax,
            nu,    #
            Km,
            φ_0,
            C_max, #
            kp,
            ke,
            k_on,
            k_off,
            k_ind,
            n_prot,
            -1
        )
        m = PrecursorABXConstantRiboWAnt(pcm, p.ant)
        simulated_states[i, :] = compute_steadystate(m, n=num_sim_pts, duration=300.0)
    end
    return simulated_states
end

function compute_GR_vs_AR_curve(
    p::PrecursorABXwAnticipation, num_sim_pts=51
)
    λs = Vector(undef, num_sim_pts)
    φ_ARs = 10 .^ range(-5, -1, length=num_sim_pts)
    for i in 1:num_sim_pts
        m = PrecursorABXwAnticipation(p.abx, mutate(p.ant, αmax=φ_ARs[i]))
        λs[i] = ABX_optimizer(m)[2]
    end
    return λs, φ_ARs
end

function compute_dα(α, αmax, αmin, k_ind, C_max, t)::Numba
    reserve = αmin
    if αmax == 0
        dα = 0.0
    else
        α_on = (1 - α / αmax) * k_ind
        α_off = -k_ind * (α - reserve)
        indicator = Int(C_max > 0)
        dα = α_on * indicator + α_off * (1 - indicator)
    end
    return dα
end

function growth_dynamics(du, u, p::PrecursorABXConstantRiboWAnt, t)
    φ_R, c_pc, C_bnd, C_in, α_AR, φ_AR = u

    @unpack_PrecursorABXConstantRiboBase p.abx
    @unpack αmax, αmin = p.ant

    φ_F = 1 - φ_0 - φ_R
    γ = monod(γmax, Km)
    φ_Rf = φ_R - C_bnd / n_prot
    C_Rf = n_prot * φ_R - C_bnd

    # α_AR is fn of αmin/max
    if αmax != 0
        reserve = αmin
        α_on = (1 - α_AR / αmax) * k_ind
        α_off = -k_ind * (α_AR - reserve)
        indicator = Int(C_max(t) > 0)
        dα_AR = α_on * indicator + α_off * (1 - indicator)
    else
        α_AR = 0
        dα_AR = 0
    end

    φ_M = φ_F - φ_AR
    λ = γ(c_pc) * φ_Rf
    du[1] = dφ_R = (α_R - φ_R) * λ
    du[2] = dc_pc = νmax * φ_M - γ(c_pc) * φ_Rf - c_pc * λ
    du[3] = dC_bnd = k_on * C_in * C_Rf - k_off * C_bnd - λ * C_bnd
    du[4] = dC_in = (
        # C_max must be a function of time
        kp * (C_max(t) - C_in)
        - ke * φ_AR * C_in * n_CAT / (C_in + Km_abx)
        + k_off * C_bnd
        - k_on * C_Rf * C_in
        - λ * C_in
    )
    du[5] = dα_AR
    du[6] = dφ_AR = (α_AR - φ_AR) * λ
end

function growth_dynamics(du, u, p::ABXwAnticipation, t)
    φ_R, r_ch, r_uc, φ_AR, α, C_bnd, C_in = u
    @unpack_AbxBaseModel p.abx
    @unpack αmax, αmin = p.ant
    reserve = αmin

    α_R = monod(1 - φ_0 - φ_AR, τ)
    γ = monod(γmax, Km)
    ν = monod(νmax, Km)
    κ = monod(κmax, τ)
    ppgpp = r_uc ./ r_ch
    ppgpp_inv = ppgpp^-1

    φ_Rb = C_bnd / n_prot
    φ_Rf = φ_R - φ_Rb
    φ_M = 1.0 - φ_R - φ_0

    C_R = φ_R * n_prot
    C_Rf = C_R - C_bnd

    if αmax == 0
        dα = 0
    else
        α_on = (1 - α / αmax) * k_ind
        α_off = -k_ind * (α - reserve)
        indicator = Int(C_max > 0)
        dα = α_on * indicator + α_off * (1 - indicator)
    end

    λ = γ(r_ch) * φ_Rf
    du[1] = dφ_R = (α_R(ppgpp_inv) - φ_R) * λ
    du[2] = dr_ch = ν(r_uc) * φ_M - γ(r_ch) * φ_Rf - r_ch * λ
    du[3] = dr_uc = -ν(r_uc) * φ_M + γ(r_ch) * φ_Rf - r_uc * λ + κ(ppgpp_inv)
    du[4] = dφ_AR = (α - φ_AR) * λ
    du[5] = dα
    du[6] = dC_bnd = k_on * C_in * C_Rf - k_off * C_bnd - λ * C_bnd
    du[7] = dC_in = (
        kp * (C_max - C_in)
        -
        ke * φ_AR * C_in
        +
        k_off * C_bnd
        -
        k_on * C_Rf * C_in
        -
        λ * C_in
    )
end

function growth_rate(soln, p)
    γ = monod(p.γmax, p.Km)
    φ_R = soln[1, :]
    r_ch = soln[2, :]
    gr = γ.(r_ch) .* φ_R
    return gr
end

function growth_rate(soln, p::AbxModels)
    γ = monod(p.abx.γmax, p.abx.Km)
    φ_R = soln[1, :]
    C_bnd = soln[6, :]
    φ_R_free = φ_R - C_bnd / p.abx.n_prot
    r_ch = soln[2, :]
    gr = γ.(r_ch) .* φ_R_free
    return gr
end

SingleABXModels = Union{
    PrecursorABXConstantRiboWAnt, PrecursorABXwAnticipation,
    PcABXAntWPerturbation,
}
function growth_rate(soln, p::T) where T<:SingleABXModels
    γ = monod(p.abx.γmax, p.abx.Km)
    φ_R = soln[1, :]
    C_bnd = soln[3, :]
    φ_R_free = φ_R - C_bnd / p.abx.n_prot
    cpc = soln[2, :]
    gr = γ.(cpc) .* φ_R_free
    return gr
end

function growth_rate(soln, p::PrecursorMultiABXwAnticipation)
    # for use with non-ss solution
    γ = monod(p.abx.γmax, p.abx.Km)
    n_abx = length(p.ant)
    φ_R = soln[1, :]
    C_bnd_vecs = [soln[2+i, :] for i in 1:n_abx]
    # Calculate free ribosome fraction
    total_C_bnds = zeros(length(φ_R))
    C_bnds_mtx = hcat(C_bnd_vecs...)
    for i in 1:length(φ_R)
        total_C_bnds[i] = calculate_total_C_bnd(φ_R[i], C_bnds_mtx[i,:], p.abx.n_prot)
    end
    φ_Rf = φ_R .- total_C_bnds ./ p.abx.n_prot
    cpc = soln[2, :]
    gr = γ.(cpc) .* φ_Rf
    return gr
end

function ss_growth_rate_w_phiR_heuristic(soln, p::PrecursorABXwAnticipation)
    γ = monod(p.abx.γmax, p.abx.Km)
    cpc = soln[1, :]
    φ_R = φ_R_heuristic.(cpc, p.abx)
    C_bnd = soln[2, :]
    φ_R_free = φ_R - C_bnd / p.abx.n_prot
    gr = γ.(cpc) .* φ_R_free
    return gr
end

function max_growth_rate_for_PD(duty, period)
    C_max = Cm_ug_per_mL_to_M(5)
    template_pm = mutate(default_PcABXwAnt_params(), νmax=5.0)
    pm_ABX = mutate(template_pm, C_max=C_max)
    αmax_opt = ABX_optimizer_allOpt(pm_ABX)[end]

    # optimal growth rate in each environment is like
    # responsive strategy at steady state
    pm_noABX = mutate(template_pm, C_max=0.0, αmax=0.0, αmin=0.0)
    max_gr_noABX = dSS_solver(pm_noABX)[2]
    pm_ABX = mutate(template_pm, C_max=C_max, αmax=αmax_opt)
    max_gr_ABX = dSS_solver(pm_ABX)[2]

    # duty is fraction of period that ABX is present
    average_gr = duty * max_gr_ABX + (1 - duty) * max_gr_noABX
    return average_gr
end

function ABX_exposure_for_PD(duty, period; C_max=Cm_ug_per_mL_to_M(5))
    # duty is fraction of period that ABX is present
    return duty * C_max
end

function compute_opt_amin_matrix(
    ;duties=0:0.01:1, periods=1:1:100,
    abx_conc=Cm_ug_per_mL_to_M(15)
)
    opt_amins = zeros(length(duties), length(periods))
    for (i, d) in enumerate(duties)
        for (j, p) in enumerate(periods)
            duration = 10 * p
            opt_out = opt_amin(abx_conc, p, d)
            opt_amins[i,j] = first(opt_out[1])
        end
    end
    return opt_amins
end

function opt_amin(
    abx_conc, period, duty; log=false, 
    DE_solver=nothing, DE_precision=nothing, high_precision = false
)
    if high_precision
        # Set BigFloat precision
	    setprecision(BigFloat, 128)
        T = BigFloat
    else
        T = Float64
    end
    abx_conc = T(abx_conc)
    period = T(period)
    duty = T(duty)

    all_kwargs = (; log=log, solver=DE_solver, precision=DE_precision, 
        numeric_type=T)
    kwargs = Dict(kw=>all_kwargs[kw] for kw in keys(all_kwargs) if !isnothing(all_kwargs[kw]))
    if period * duty < 6.0 && period * (1 - duty) < 6.0
        opt_fun =  GR_given_amin
    else
        opt_fun = GR_given_amin
    end

    function h(x, grad)
        if length(x) == 1
            # For a single value, pass it directly
            return opt_fun(x[1], abx_conc, period, duty; kwargs...)
        else
            # For multiple values, map over each one individually
            return [opt_fun(x_i, abx_conc, period, duty; kwargs...) for x_i in x]
        end
    end
    base_pm = default_PcABXwAnt_params(numeric_type=T)
    pm_ABX = mutate(base_pm, C_max=abx_conc)
	αmax_opt = ABX_optimizer_allOpt(pm_ABX)[end]

    x0 = [αmax_opt/4]
	opt = NLopt.Opt(:LN_BOBYQA, 1)
	NLopt.lower_bounds!(opt, T(0.0))
	NLopt.upper_bounds!(opt, αmax_opt)
	NLopt.xtol_abs!(opt, T(1e-5))
	NLopt.max_objective!(opt, h)
	NLopt.maxeval!(opt, 1000)
    max_f, max_x, ret = NLopt.optimize(opt, x0)
    num_evals = NLopt.numevals(opt)
    if log
        @info (
            """
            objective value       : $max_f
            solution              : $max_x
            solution status       : $ret
            # function evaluation : $num_evals
            """
        )
    end
    return max_x, max_f, ret
end

function opt_amin_multi(
    n::Int, abx_conc, period, duty, duration; flucts=nothing, log=false,
    n_responsive=0
)
    function h(x, grad)
        if length(x) == 1
            # For a single value, pass it directly
            return GR_given_amin_multi(
                x[1], n, abx_conc, period, duty, duration, 
                env=flucts,
                n_responsive=n_responsive,
            )
        else
            # For multiple values, map over each one individually
            return [GR_given_amin_multi(
                x_i, n, abx_conc, period, duty, duration, 
                env=flucts,
                n_responsive=n_responsive,
            ) for x_i in x]
        end
    end
    base_pm = default_PcABXwAnt_params()
    pm_ABX = mutate(base_pm, C_max=abx_conc*n)
	αmax_opt_tot = ABX_optimizer_allOpt(pm_ABX)[end]
    αmax_opt = αmax_opt_tot / n

    x0 = [αmax_opt/4]
	opt = NLopt.Opt(:LN_BOBYQA, 1)
	NLopt.lower_bounds!(opt, 0.0)
	NLopt.upper_bounds!(opt, αmax_opt)
	NLopt.xtol_abs!(opt, 1e-4)
	NLopt.max_objective!(opt, h)
	NLopt.maxeval!(opt, 100)
    max_f, max_x, ret = NLopt.optimize(opt, x0)
    num_evals = NLopt.numevals(opt)
    if log
        @info (
            """
            objective value       : $max_f
            solution              : $max_x
            solution status       : $ret
            # function evaluation : $num_evals
            """
        )
    end
    return max_x, max_f, ret
end

function GR_given_amin_multi(
    amin, n::Int, abx_conc, period, duty, duration; env=nothing, n_responsive=0,
    precision=1e-7, ret2=false
)
    base_pm = default_PcMultiABXwAnt_params(n)
    u0 = full_state_to_u0_multiABX(dSS_solver(base_pm, n=n))
    pm_ABX = mutate(default_PcABXwAnt_params(), C_max=abx_conc*n)
    # doesnt work w multi
    αmax_opt_tot = ABX_optimizer_allOpt(pm_ABX)[end]
    if env == nothing
        starts = vcat(0, zeros(n-1)) .* period
        flucts(i, t) = [
            square_wave(
                period, duty, duration, amplitude=abx_conc, x0 = starts[i]
            ) for i in 1:n
        ][i](t)
    else
        flucts = env
    end
    amins = vcat(amin * ones(n - n_responsive), zeros(n_responsive))
    pm = mutate(base_pm, αmax = αmax_opt_tot/n, C_max = flucts, αmin=amins)
    tspan = (0.0, 10 * period)
    prob = ODEProblem(growth_dynamics, u0, tspan, pm)
    sol = solve(prob, Rodas4(), abstol=precision, reltol=precision)
    # integrate over the terminal period
    ts = range(duration - period, stop=duration, length=5000)
    gr_series = growth_rate(sol(ts), pm)
    gr_cumul = integrate(ts, gr_series)
    gr = gr_cumul / period
    if ret2
        return gr, sol
    end
    return gr
end

function GR_given_amin(
    amin, abx_conc, period, duty;
    precision=1e-7, solver=Rodas4(), return_last_state=false,
    log=nothing, numeric_type = Float64
)
    T = numeric_type
    amin = T(amin)
    abx_conc = T(abx_conc)
    period = T(period)
    duty = T(duty)

    if duty > 0.95 || duty < 0.05 || period > 20
        duration = 3*period
    elseif period > 5
        duration = 6 * period
    else
        duration = 10 * period
    end

    base_pm = default_PcABXwAnt_params(numeric_type=numeric_type)
    u0 = full_state_to_u0(dSS_solver(base_pm, T=numeric_type))
    pm_ABX = mutate(base_pm, C_max=abx_conc)
    αmax_opt = T(ABX_optimizer_allOpt(pm_ABX)[end])
    "flucts = square_wave(
        period, duty, duration, amplitude=abx_conc, numeric_type=numeric_type
    )"
    C_ext_0 = abx_conc
    perturbation = Perturbation(abx_conc, duty, period, C_ext_0)
    bpm = mutate(base_pm, αmax = αmax_opt, αmin=amin)
    pm = PcABXAntWPerturbation(bpm.abx, bpm.ant, perturbation)

    tspan = (T(0.0), duration)
    prob = ODEProblem(growth_dynamics, u0, tspan, pm)

    cb_on = PeriodicCallback(toggle_perturbation_on!, period)
    cb_off = PeriodicCallback(
        toggle_perturbation_off!, period, phase= period * duty
    )
    cbs = CallbackSet(cb_on, cb_off)

    sol = solve(prob, solver, abstol=T(precision), reltol=T(precision), callback=cbs)
    # integrate over the terminal period
    ts = range(duration - period, stop=duration, length=5000)
    gr_series = growth_rate(sol(ts), pm)
    gr_cumul = integrate(ts, gr_series)
    gr = gr_cumul / period
    if return_last_state
        return gr, sol[end]
    end
    return gr
end

function sweep_amin_vs_kcat(
    amin_range, kcat_range, base_pm, abx_conc, period, duty, duration;
    precis=1e-7
)
    u0 = full_state_to_u0(dSS_solver(base_pm))
	pm_ABX = mutate(base_pm, C_max=abx_conc)
	αmax_opt = ABX_optimizer_allOpt(pm_ABX)[end]
    flucts = square_wave(period, duty, duration, amplitude=abx_conc)
	template_pm = mutate(base_pm, αmax = αmax_opt, C_max = flucts)

    grs_amin = zeros(m,n)
    states_amin = Matrix{Any}(undef, m, n)
    
    m = length(kcat_range)
    n = length(amin_range)
    for i in 1:m
        for j in 1:n
            pm_ = mutate(template_pm, ke=kcat_range[i], αmin=amin_range[j])
            tspan = (0.0, 10 * period)
            prob = ODEProblem(growth_dynamics, u0, tspan, pm_)
            sol = solve(prob, Rodas4(), abstol=precis, reltol=precis)
            # integrate over the terminal period
            ts = range(duration - period, stop=duration, length=5000)
            gr_series = growth_rate(sol(ts), pm_)
            gr_cumul = integrate(ts, gr_series)
            grs_amin[i,j] = gr_cumul / period
            states_amin[i,j] = sol
        end
    end
    return grs_amin, states_amin
end

function growth_rate_phase_diagram_mtx(
    duties, periods; AR_strategy="none",
    normalize_by_max_gr=false, C_max=Cm_ug_per_mL_to_M(5),
    by_freqs=false, ant_αmin=0.001, const_αmin=0.001,
)
    precis = 1e-7
    template_pm = mutate(default_PcABXwAnt_params(), νmax=5.0)
    pm_ABX = mutate(template_pm, C_max=C_max)
    pm_noABX = mutate(template_pm, C_max=0.0)
    # optimal ϕ_AR for given [ABX] (and other params)
    αmax_opt = ABX_optimizer_allOpt(pm_ABX)[end]

    if AR_strategy == "none"
        base_pm = mutate(template_pm, αmax=0)
    elseif AR_strategy == "responsive"
        base_pm = mutate(template_pm, αmax=αmax_opt)
    elseif AR_strategy == "anticipatory"
        base_pm = mutate(template_pm, αmax=αmax_opt, αmin=ant_αmin)
    elseif AR_strategy == "constant"
        base_pm = mutate(template_pm, αmax=const_αmin, αmin=const_αmin)
    else
        error("Invalid AR_strategy")
    end
    #  u0 set by ss in no ABX 
    u0 = full_state_to_u0(dSS_solver(pm_noABX))

    if by_freqs
        periods = 1 ./ periods
    end
    grs = zeros(length(duties), length(periods))
    for (i, d) in enumerate(duties)
        for (j, p) in enumerate(periods)
            # simplify calculation
            # longer periods means more time at steady state
            if p > 10
                # NOTE: assumes square wave fluctuations
                # get steady state growth soln
                pm_ABX = mutate(base_pm, C_max=C_max)
                gr_ABX_ss = ABX_optimal_λ(pm_ABX, opt=dSS_solver)
                # solvers use αmax to set φ_AR in ss
                # so set αmax to αmin that would occur in the absence of ABX
                pm_noABX = mutate(base_pm, αmax=base_pm.ant.αmin, C_max=0.0)
                gr_noABX_ss = dSS_solver(pm_noABX)[2]
                average_gr = d * gr_ABX_ss + (1 - d) * gr_noABX_ss
                gr_correction = average_gr * (p - 10)
            end

            if p > 10
                period = 10
            else
                period = p
            end
            # p <= 10 
            duration = 10 * period
            PD_flucts = square_wave(period, d, duration, amplitude=C_max)
            pm = mutate(base_pm, C_max=PD_flucts)
            tspan_ = (0.0, 10 * period)
            prob = ODEProblem(growth_dynamics, u0, tspan_, pm)
            sol = solve(prob, Rodas4(), abstol=precis, reltol=precis)
            # integrate over the terminal period
            ts = range(duration - period, stop=duration, length=5000)
            gr_series = growth_rate(sol(ts), pm)
            gr_cumul = integrate(ts, gr_series)

            if p > 10
                grs[i, j] = (gr_cumul + gr_correction)/p
            else
                grs[i, j] = gr_cumul / period
            end
        end
    end

    if normalize_by_max_gr
        for (i, d) in enumerate(duties)
            for (j, p) in enumerate(periods)
                grs[i, j] /= max_growth_rate_for_PD(d, p)
            end
        end
    end

    return grs
end

# WARN - period is not well defined if there are multiple periods
function extract_average_growth_rate(soln, pm, period, duration)
    # integrate over the terminal period
    ts = range(duration - period, stop=duration, length=5000)
    gr_series = growth_rate(soln(ts), pm)
    gr_cumul = integrate(ts, gr_series)
    return gr_cumul / period
end

function compute_optimal_steadystate(n, p::PrecursorModelConstantRiboFraction)
    nus = range(1e-3, 20, length=n)
    out = zeros(2, n)

    g_pc = monod(p.γmax, p.Km)
    growth_rate_pc(pc, ϕ_R) = ϕ_R * g_pc(pc)

    for i in 1:n
        νmax = nus[i]
        v = νmax
        model = Model(Ipopt.Optimizer)

        @variable(model, pc >= 0.0, start = 1e-3)
        @variable(model, 0.0 <= ϕ_R <= 1 - p.φ_0, start = 0.1)

        register(model, :g_pc, 1, g_pc; autodiff=true)
        register(model, :growth_rate_pc, 2, growth_rate_pc; autodiff=true)

        @NLconstraint(model, 0.0 == v * (1.0 - ϕ_R - p.φ_0) - g_pc(pc) * ϕ_R - pc * g_pc(pc) * ϕ_R)

        @NLobjective(model, Max, growth_rate_pc(pc, ϕ_R))

        JuMP.optimize!(model)
        out[1, i] = value(pc)
        out[2, i] = value(ϕ_R)
    end
    return out
end

function compute_steadystate(p::PrecursorABXConstantRiboWAnt; n::Int64=1, params_out::Int64=1, duration::Numba=20.0, ABX=true)
    # given a fully specified set of parameters, compute the growth rate at steady state
    @unpack_PrecursorABXConstantRiboBase p.abx
    @unpack_AbxAnticipationParams p.ant

    if ABX == true
        φ_AR0 = αmax
    else
        φ_AR0 = αmin
    end
    tspan = (0.0, duration)

    cpc0 = Km
    α_AR0 = φ_AR0
    C_in0 = C_max / 10
    C_bnd0 = C_in0 / 5
    γ = monod(γmax, Km)

    λs = zeros(n)
    if n > 1
        α_Rs = range(0, 1 - φ_0, length=n)
    else
        α_Rs = [α_R]
    end
    for i in 1:n
        φ_R0 = α_Rs[i]
        u0 = [φ_R0, cpc0, φ_AR0, α_AR0, C_bnd0, C_in0]
        p_base = PrecursorABXConstantRiboBase(
            γmax, νmax, Km, φ_0, C_max,
            kp, ke, k_on, k_off, k_ind, n_prot,
            α_Rs[i]
        )
        p = PrecursorABXConstantRiboWAnt(p_base, p.ant)
        prob = ODEProblem(growth_dynamics, u0, tspan, p)
        soln = solve(prob, reltol=1e-9, abstol=1e-9)

        φ_R = soln[1, end]
        C_bnd = soln[5, end]
        φ_R_free = φ_R .- C_bnd ./ n_prot
        c_pc = soln[2, end]
        λ = γ.(c_pc) .* φ_R_free
        λs[i] = λ
    end

    if params_out == 1
        return λs[1]
    end

    return [α_Rs, λs]
end

function analytic_optimal_growth_ABXfree(p::PrecursorABXwAnticipation)
    return analytic_optimal_growth(p, abx=false)
end

function analytic_optimal_growth(
    p::PrecursorABXwAnticipation; abx=false, opt=ABX_optimizer_v2
)
    """
    Only true in steady-state
    """
    @unpack_PrecursorABXBaseModel p.abx
    @unpack αmax, αmin = p.ant
    reserve = αmin

    if abx == false
        φ_R = ABXfree_optimal_φ_R(p)
        φ_M = 1 .- reserve .- φ_R .- φ_0
        Γ = γmax .* φ_R
        prefactor = 1 / (2 * (1 - Km))
        N = νmax .* φ_M
        return prefactor * (Γ .+ N .- sqrt.((N .+ Γ) .^ 2 - 4 * (1 - Km) .* N .* Γ))
    else
        @warn "Returning numerical solution for ABX optimal growth"
        return ABX_optimal_λ(p, opt=opt)
    end
end

function optimal_φ_R(p::PrecursorModelFamily)
    prefactor = 1 - p.φ_0
    add = p.νmax + p.γmax
    mul = p.νmax * p.γmax
    #       Gamma - Nu
    sub = p.γmax - p.νmax
    numer = mul * (1 - 2 * p.Km) + p.νmax^2 + sqrt(p.Km * mul) * sub
    denom = add^2 - 4 * p.Km * mul
    return prefactor * numer / denom
end

function optimal_precursor_concentration(p::PrecursorModelFamily)
    νmax = p.νmax
    γmax = p.γmax
    Km = p.Km
    φ_0 = p.φ_0

    φ_R_opt = optimal_φ_R(p)
    N = νmax * (1 - φ_0 - φ_R_opt)
    prefactor = N - γmax * φ_R_opt
    radical = (N - γmax * φ_R_opt)^2 + 4 * γmax * φ_R_opt * N * Km
    denom = 2 * γmax * φ_R_opt
    return (prefactor + sqrt(radical)) / denom
end

function simple_optimizer(p::PrecursorModelFamily)
    """
    Optimal cell state for the given PrecursorModelFamily params
    """
    φ_R = optimal_φ_R(p)
    c_pc = optimal_precursor_concentration(p)
    λ = φ_R * p.γmax * c_pc / (c_pc + p.Km)
    return (φ_R, c_pc, λ)
end

function optimal_φ_R(p::PrecursorABXwAnticipation; ABX=false)
    """
    steady-state optimal φ_R given a particular φ_AR expression strategy
    """
    if ABX
        return ABX_optimal_φ_R(p)
    else
        return ABXfree_optimal_φ_R(p)
    end
end

function abxOpt_solver(p::PrecursorABXwAnticipation)
    @unpack_PrecursorABXBaseModel p.abx
    state = ABX_optimizer_v2(p)
    φ_R, λ, c_pc, C_bnd, C_in = state
    fAR = p.ant.αmax
    φ_F = 1 - φ_0 - φ_R
    φ_M = φ_F * (1 - fAR)
    φ_AR = φ_F * fAR
    full_state = [φ_R, λ, c_pc, C_bnd, C_in, φ_AR, φ_M]
    return full_state
end

function abxOpt_coR_solver(p::PrecursorABXwAnticipation)
    """ sets φ_AR to be co-regulated with φ_R"""
    @unpack_PrecursorABXBaseModel p.abx
    state = ABX_optimizer_v2_coR(p)
    φ_R, λ, c_pc, C_bnd, C_in = state
    fAR = p.ant.αmax
    φ_AR = φ_R * fAR
    φ_M = 1 - φ_0 - φ_R - φ_AR
    full_state = [φ_R, λ, c_pc, C_bnd, C_in, φ_AR, φ_M]
    return full_state
end

function ABX_optimizer_v2(p::PrecursorABXwAnticipation)
    """
    Computes optimal steady state cell state for a given maximal ABX resistance fraction

    IMPORTANT: 
    	computes the R vs M balance out of 1 - φ_0 and not 1-φ_AR-φ_0
    	bc the φ_AR is considered an unregulated protein whose fraction is diminished
    	wrt to the 'command' hard requirement for the ribosomal fraction.
    	
    	Since taking φ_AR out of φ_M after the optimal balance btw R vs M would 
    	not be optimal, we define a new ν_eff that accounts for the fact that
    	φ_M total = φ_M^true + φ_AR. we rename φ_M^total to φ_F for 'Flexible'
    	and φ_M^true to φ_M. Further, to create the distinction between the
    	fraction of φ_F devoted to AR and the total mass fraction devoted to AR,
    	we denote φ_AR as the total mass fraction and f_AR as the fraction of φ_F.

    returns φ_R, λ, c_pc, C_bnd, and C_in

    Formulated to solve directly from the steady-state equations,
    so that grokking the code is easier

    and assumes that φ_AR===αmax
    """
    @unpack_PrecursorABXBaseModel p.abx
    model = Model(Ipopt.Optimizer)
    set_silent(model)

    f_AR = p.ant.αmax

    # dynamical variables
    u = JuMP.@variables(model, begin
        0 <= φ_R <= 1 - φ_0
        0 <= c_pc <= 20
        0 <= C_bnd <= (1 - φ_0) * n_prot
        0 <= C_in <= C_max
    end)

    @expressions(model, begin
        # concentration of free ribosomes
        C_Rf, n_prot .* φ_R .- C_bnd
        # mass fraction of free ribosomes
        φ_Rf, φ_R .- C_bnd ./ n_prot
        # 'flexible' protein mass fraction
        φ_F, 1 - φ_0 .- φ_R
        # metabolic protein mass fraction
        φ_M, φ_F .* (1 .- f_AR)
        # AR protein mass fraction
        φ_AR, φ_F .* f_AR
        # translation rate
        γ, γmax .* c_pc ./ (c_pc .+ Km)
        # growth rate
        λ, γ * φ_Rf
    end)
    @constraint(model, φ_Rf >= 0)

    @constraints(model, begin
        0 == νmax * φ_M - γ * φ_Rf - c_pc * λ
        0 == k_on * C_in * C_Rf - k_off * C_bnd - λ * C_bnd
        0 == kp * (C_max - C_in) - ke * φ_AR * n_CAT * C_in / (C_in + Km_abx) + k_off * C_bnd - k_on * C_Rf * C_in - λ * C_in
    end)

    @objective(model, Max, γmax * c_pc / (c_pc + Km) * φ_Rf)
    JuMP.optimize!(model)
    out = (value.(φ_R), value(λ), value.(c_pc), value.(C_bnd), value.(C_in))

    return out
end

function ABX_optimizer_v2_coR(p::PrecursorABXwAnticipation)
    """
    like ABX_optimizer_v2, but with φ_AR co-regulated with φ_R
    """
    @unpack_PrecursorABXBaseModel p.abx
    model = Model(Ipopt.Optimizer)
    set_silent(model)

    f_AR = p.ant.αmax

    # dynamical variables
    u = JuMP.@variables(model, begin
        0 <= φ_R <= 1 - φ_0
        0 <= c_pc <= 20
        0 <= C_bnd <= (1 - φ_0) * n_prot
        0 <= C_in <= C_max
    end)

    @expressions(model, begin
        # concentration of free ribosomes
        C_Rf, n_prot .* φ_R .- C_bnd
        # mass fraction of free ribosomes
        φ_Rf, φ_R .- C_bnd ./ n_prot
        # AR protein mass fraction
        φ_AR, φ_R .* f_AR
        # metabolic protein mass fraction
        φ_M, 1 - φ_0 .- φ_R .- φ_AR
        # translation rate
        γ, γmax .* c_pc ./ (c_pc .+ Km)
        # growth rate
        λ, γ * φ_Rf
    end)
    @constraint(model, φ_Rf >= 0)

    @constraints(model, begin
        0 == νmax * φ_M - γ * φ_Rf - c_pc * λ
        0 == k_on * C_in * C_Rf - k_off * C_bnd - λ * C_bnd
        0 == kp * (C_max - C_in) - ke * φ_AR * n_CAT * C_in / (C_in + Km_abx) + k_off * C_bnd - k_on * C_Rf * C_in - λ * C_in
    end)

    @objective(model, Max, γmax * c_pc / (c_pc + Km) * φ_Rf)
    JuMP.optimize!(model)
    out = (value.(φ_R), value(λ), value.(c_pc), value.(C_bnd), value.(C_in))

    return out
end

function core_ss_eqns(x, p::PrecursorABXwAnticipation)
    @unpack_PrecursorABXBaseModel p.abx
    c_pc, C_bnd, C_in = x
    fAR = p.ant.αmax

    φ_R = 2 * (1 - φ_0 - 0.05) * c_pc / (c_pc + 0.8) + 0.05
    φ_F = 1 - φ_0 - φ_R
    φ_M = φ_F * (1 - fAR)
    φ_AR = φ_F * fAR
    C_Rf = n_prot * φ_R - C_bnd
    φ_Rf = φ_R - C_bnd / n_prot
    λ = γmax * c_pc / (c_pc + Km) * φ_Rf

    # steady-state equations
    cpc_ss = νmax * φ_M - (1 + c_pc) * λ
    C_bnd_ss = k_on * C_in * C_Rf - k_off * C_bnd - λ * C_bnd
    C_in_ss = kp * (C_max - C_in) - ke * φ_AR * n_CAT * C_in / (C_in + Km_abx) + k_off * C_bnd - k_on * C_Rf * C_in - λ * C_in

    return [cpc_ss, C_bnd_ss, C_in_ss]
end

function core_ss_eqns_for_Bifurc(x, abx_conc, p::PrecursorABXwAnticipation)
    @unpack_PrecursorABXBaseModel p.abx
    c_pc, C_bnd, C_in = x
    fAR = p.ant.αmax

    φ_R = 2 * (1 - φ_0 - 0.05) * c_pc / (c_pc + 0.8) + 0.05
    φ_F = 1 - φ_0 - φ_R
    φ_M = φ_F * (1 - fAR)
    φ_AR = φ_F * fAR
    C_Rf = n_prot * φ_R - C_bnd
    φ_Rf = φ_R - C_bnd / n_prot
    λ = γmax * c_pc / (c_pc + Km) * φ_Rf

    # steady-state equations
    cpc_ss = νmax * φ_M - (1 + c_pc) * λ
    C_bnd_ss = k_on * C_in * C_Rf - k_off * C_bnd - λ * C_bnd
    C_in_ss = kp * (abx_conc[1] - C_in) - ke * φ_AR * n_CAT * C_in / (C_in + Km_abx) + k_off * C_bnd - k_on * C_Rf * C_in - λ * C_in

    return [cpc_ss, C_bnd_ss, C_in_ss]
end

function alt_core_ss_eqns(x, p::PrecursorABXwAnticipation)
    """ φ_AR co-regulated with φ_R """
    @unpack_PrecursorABXBaseModel p.abx
    c_pc, C_bnd, C_in = x
    fAR = p.ant.αmax

    φ_R = 2 * (1 - φ_0 - 0.05) * c_pc / (c_pc + 0.8) + 0.05
    φ_AR = fAR * φ_R
    φ_M = 1 - φ_0 - φ_R - φ_AR
    C_Rf = n_prot * φ_R - C_bnd
    φ_Rf = φ_R - C_bnd / n_prot
    λ = γmax * c_pc / (c_pc + Km) * φ_Rf

    # steady-state equations
    cpc_ss = νmax * φ_M - (1 + c_pc) * λ
    C_bnd_ss = k_on * C_in * C_Rf - k_off * C_bnd - λ * C_bnd
    C_in_ss = kp * (C_max - C_in) - ke * φ_AR * n_CAT * C_in / (C_in + Km_abx) + k_off * C_bnd - k_on * C_Rf * C_in - λ * C_in

    return [cpc_ss, C_bnd_ss, C_in_ss]
end

function alt_eqns_for_Bifurc(x, abx_conc, p::PrecursorABXwAnticipation)
    """ φ_AR co-regulated with φ_R """
    @unpack_PrecursorABXBaseModel p.abx
    c_pc, C_bnd, C_in = x
    fAR = p.ant.αmax

    φ_R = 2 * (1 - φ_0 - 0.05) * c_pc / (c_pc + 0.8) + 0.05
    φ_AR = fAR * φ_R
    φ_M = 1 - φ_0 - φ_R - φ_AR
    C_Rf = n_prot * φ_R - C_bnd
    φ_Rf = φ_R - C_bnd / n_prot
    λ = γmax * c_pc / (c_pc + Km) * φ_Rf

    # steady-state equations
    cpc_ss = νmax * φ_M - (1 + c_pc) * λ
    C_bnd_ss = k_on * C_in * C_Rf - k_off * C_bnd - λ * C_bnd
    C_in_ss = kp * (abx_conc[1] - C_in) - ke * φ_AR * n_CAT * C_in / (C_in + Km_abx) + k_off * C_bnd - k_on * C_Rf * C_in - λ * C_in

    return [cpc_ss, C_bnd_ss, C_in_ss]
end

function heuristic_solver(p::PrecursorABXwAnticipation; kwargs...)
    """ φ_AR co-regulated with φ_M """
    @unpack_PrecursorABXBaseModel p.abx
    state = SS_optimizer_v4(p; kwargs...)
    φ_R, λ, c_pc, C_bnd, C_in = state
    fAR = p.ant.αmax
    φ_F = 1 - φ_0 - φ_R
    φ_M = φ_F * (1 - fAR)
    φ_AR = φ_F * fAR
    full_state = [φ_R, λ, c_pc, C_bnd, C_in, φ_AR, φ_M]
    return full_state
end

function heuristic_solver_coR(p::PrecursorABXwAnticipation)
    """ φ_AR co-regulated with φ_R """
    @unpack_PrecursorABXBaseModel p.abx
    state = SS_optimizer_v4_coR(p)
    φ_R, λ, c_pc, C_bnd, C_in = state
    fAR = p.ant.αmax
    φ_AR = φ_R * fAR
    φ_M = 1 - φ_0 - φ_R - φ_AR
    full_state = [φ_R, λ, c_pc, C_bnd, C_in, φ_AR, φ_M]
    return full_state
end

function SS_optimizer_v4(
    p::PrecursorABXwAnticipation;
    V=2, K=0.8, n=1,
)
    """
    Computes steady state cell state

    *** φ_AR co-regulated with φ_M ***

    Assumes φ_R = (1 - φ_0) * c_pc / (c_pc + K)

    returns φ_R, λ, c_pc, C_bnd, and C_in
    """
    @unpack_PrecursorABXBaseModel p.abx

    # Ipopt specific settings
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "tol", 1e-6)
    set_optimizer_attribute(model, "constr_viol_tol", 1e-6)
    set_silent(model)


    f_AR = p.ant.αmax

    # dynamical variables
    # for Ipopt
    u = JuMP.@variables(model, begin
        0 <= c_pc <= 15
        0 <= C_bnd <= (1 - φ_0) * n_prot
        0 <= C_in <= C_max
    end)

    @expressions(model, begin
        # ribosome MF set by c_pc
        φ_R, V * (1 - φ_0 - 0.05) * c_pc^n / (c_pc^n + K) + 0.05
        # concentration of free ribosomes
        C_Rf, n_prot * φ_R - C_bnd
        # mass fraction of free ribosomes
        φ_Rf, φ_R - C_bnd / n_prot
        # 'flexible' protein mass fraction
        φ_F, 1 - φ_0 - φ_R
        # metabolic protein mass fraction
        φ_M, φ_F * (1 - f_AR)
        # AR protein mass fraction
        φ_AR, φ_F * f_AR
        # translation rate
        γ, γmax * c_pc / (c_pc + Km)
        # growth rate
        λ, γ * φ_Rf
    end)
    @constraint(model, φ_Rf >= 0)

    @constraints(model, begin
        0 == νmax * φ_M - γ * φ_Rf - c_pc * λ
        0 == k_on * C_in * C_Rf - k_off * C_bnd - λ * C_bnd
        0 == kp * (C_max - C_in) - ke * φ_AR * n_CAT * C_in / (C_in + Km_abx) + k_off * C_bnd - k_on * C_Rf * C_in - λ * C_in
    end)

    @objective(model, Max, γmax * c_pc / (c_pc + Km) * φ_Rf)
    JuMP.optimize!(model)
    out = (value(φ_R), value(λ), value(c_pc), value(C_bnd), value(C_in))

    return out
end

function SS_optimizer_v4_coR(
    p::PrecursorABXwAnticipation;
    V=2, K=0.8, n=1,
)
    """
    Computes steady state cell state

    *** φ_AR co-regulated with φ_R ***

    otherwise same as SS_optimizer_v4
    """
    @unpack_PrecursorABXBaseModel p.abx

    # Ipopt specific settings
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "tol", 1e-6)
    set_optimizer_attribute(model, "constr_viol_tol", 1e-6)
    set_silent(model)


    f_AR = p.ant.αmax

    # dynamical variables
    # for Ipopt
    u = JuMP.@variables(model, begin
        0 <= c_pc <= 15
        0 <= C_bnd <= (1 - φ_0) * n_prot
        0 <= C_in <= C_max
    end)

    @expressions(model, begin
        # ribosome MF set by c_pc
        φ_R, V * (1 - φ_0 - 0.05) * c_pc^n / (c_pc^n + K) + 0.05
        # concentration of free ribosomes
        C_Rf, n_prot * φ_R - C_bnd
        # mass fraction of free ribosomes
        φ_Rf, φ_R - C_bnd / n_prot
        # AR protein mass fraction
        φ_AR, φ_R * f_AR
        # 'flexible' protein mass fraction
        φ_M, 1 - φ_0 - φ_R - φ_AR
        # translation rate
        γ, γmax * c_pc / (c_pc + Km)
        # growth rate
        λ, γ * φ_Rf
    end)
    @constraint(model, φ_Rf >= 0)

    @constraints(model, begin
        0 == νmax * φ_M - γ * φ_Rf - c_pc * λ
        0 == k_on * C_in * C_Rf - k_off * C_bnd - λ * C_bnd
        0 == kp * (C_max - C_in) - ke * φ_AR * n_CAT * C_in / (C_in + Km_abx) + k_off * C_bnd - k_on * C_Rf * C_in - λ * C_in
    end)

    @objective(model, Max, γmax * c_pc / (c_pc + Km) * φ_Rf)
    JuMP.optimize!(model)
    out = (value(φ_R), value(λ), value(c_pc), value(C_bnd), value(C_in))

    return out
end

function ABX_optimizer(p::PrecursorABXwAnticipation; counter=0)
    """
    Computes optimal steady state cell state for a given maximal ABX resistance fraction

    returns φ_R, λ, c_pc, C_bnd, and C_in

    Formulated to solve directly from the steady-state equations,
    so that grokking the code is easier

    and assumes that φ_AR===αmax
    """
    @unpack_PrecursorABXBaseModel p.abx
    model = Model(Ipopt.Optimizer)
    set_silent(model)

    φ_AR = p.ant.αmax

    # dynamical variables
    u = JuMP.@variables(model, begin
        0 <= φ_R <= 1 - φ_0
        0 <= c_pc
        0 <= C_bnd
        0 <= C_in
    end)

    @expressions(model, begin
        # concentration of free ribosomes
        C_Rf, n_prot .* φ_R .- C_bnd
        # mass fraction of free ribosomes
        φ_Rf, φ_R .- C_bnd ./ n_prot
        # metabolic protein mass fraction
        φ_M, 1 - φ_0 .- φ_AR .- φ_R
        # translation rate
        γ, γmax .* c_pc ./ (c_pc .+ Km)
        # growth rate
        λ, γ * φ_Rf
    end)
    @constraint(model, φ_Rf >= 0)

    @constraints(model, begin
        0 == νmax * φ_M - γ * φ_Rf - c_pc * λ
        0 == k_on * C_in * C_Rf - k_off * C_bnd - λ * C_bnd
        0 == kp * (C_max - C_in) - ke * φ_AR * n_CAT * C_in / (C_in + Km_abx) + k_off * C_bnd - k_on * C_Rf * C_in - λ * C_in
    end)

    @objective(model, Max, γmax * c_pc / (c_pc + Km) * φ_Rf)
    JuMP.optimize!(model)
    out = (value.(φ_R), value(λ), value.(c_pc), value.(C_bnd), value.(C_in))

    # sometimes the solver fails
    if out[3] > 10 && counter < 10
        # re-run the optimizer with an infinitesimal perturbation to the parameters
        δ = 1e-10
        # randomly select a parameter to perturb, exclude index 6
        index = rand(1:12)
        while index == 6
            index = rand(1:12)
        end
        arg_names = propertynames(p.abx)
        args = [getproperty(p.abx, arg) for arg in arg_names]
        args[index] += δ
        p2_base = PrecursorABXBaseModel(args...)
        p2 = PrecursorABXwAnticipation(p2_base, p.ant)
        @debug "Re-running optimizer with $(arg_names[index]) perturbed by: $(δ)"
        return ABX_optimizer(p2, counter=counter + 1)
    elseif counter >= 10
        @warn "Optimization failed to converge after 10 attempts"
        return (NaN, NaN, NaN, NaN, NaN)
    end

    return out
end

function allOpt_solver(p::PrecursorABXwAnticipation; counter=0)
    state = ABX_optimizer_allOpt(p, counter=counter)
    φ_R, λ, c_pc, C_bnd, C_in, φ_AR = state
    φ_M = 1 - p.abx.φ_0 - φ_AR - φ_R
    return [φ_R, λ, c_pc, C_bnd, C_in, φ_AR, φ_M]
end

function ABX_optimizer_allOpt(p::PrecursorABXwAnticipation; counter=0)
    """
    Computes optimal steady state cell state; also optimizes the ABX resistance fraction

    returns φ_R, λ, c_pc, C_bnd, C_in, and φ_AR

    Formulated to solve directly from the steady-state equations,
    so that grokking the code is easier

    maximizes φ_AR as well
    """
    @unpack_PrecursorABXBaseModel p.abx
    model = Model(Ipopt.Optimizer)
    set_silent(model)

    # dynamical variables
    # runs faster with the constraints.
    # but gives a little singularity for one of the kcats
    # at ABX = 0.0
    u = JuMP.@variables(model, begin
        0 <= φ_R <= 1 - φ_0
        0 <= φ_AR <= 1 - φ_0
        0 <= c_pc # <= 20
        0 <= C_bnd # <= (1 - φ_0) * n_prot
        0 <= C_in # <= C_max
    end)

    @expressions(model, begin
        # concentration of free ribosomes
        C_Rf, n_prot .* φ_R .- C_bnd
        # mass fraction of free ribosomes
        φ_Rf, φ_R .- C_bnd ./ n_prot
        # metabolic protein mass fraction
        φ_M, 1 - φ_0 .- φ_AR .- φ_R
        # translation rate
        γ, γmax .* c_pc ./ (c_pc .+ Km)
        # growth rate
        λ, γ * φ_Rf
    end)
    @constraint(model, φ_Rf >= 0)

    @constraints(model, begin
        0 .== νmax .* (1 - φ_0 .- φ_R - φ_AR) .- (1 .+ c_pc) .* λ
        0 .== k_on .* C_in .* C_Rf .- k_off .* C_bnd .- λ .* C_bnd
        0 .== kp .* (C_max .- C_in) .- ke .* φ_AR .* n_CAT .* C_in ./ (C_in .+ Km_abx) .+ k_off .* C_bnd .- k_on .* C_Rf .* C_in .- λ .* C_in
    end)

    @objective(model, Max, γmax * c_pc / (c_pc + Km) * φ_Rf)
    JuMP.optimize!(model)
    out = (value.(φ_R), value(λ), value.(c_pc), value.(C_bnd), value.(C_in), value.(φ_AR))

    if out[3] > 10 && counter < 10
        # re-run the optimizer with an infinitesimal perturbation to the parameters
        δ = 1e-10
        # randomly select a parameter to perturb, exclude index 6
        index = rand(1:12)
        while index == 6
            index = rand(1:12)
        end
        arg_names = propertynames(p.abx)
        args = [getproperty(p.abx, arg) for arg in arg_names]
        args[index] += δ
        p2_base = PrecursorABXBaseModel(args...)
        p2 = PrecursorABXwAnticipation(p2_base, p.ant)
        @debug "Re-running optimizer with $(arg_names[index]) perturbed by: $(δ)"
        return ABX_optimizer_allOpt(p2, counter=counter + 1)
    elseif counter >= 10
        @warn "Optimization failed to converge after 10 attempts"
        return (NaN, NaN, NaN, NaN, NaN, NaN)
    end

    return out
end

function ABX_optimal_φ_R(p::PrecursorABXwAnticipation)
    return ABX_optimizer(p)[1]
end

function ABX_optimal_λ(p::PrecursorABXwAnticipation; opt=ABX_optimizer_v2)
    return opt(p)[2]
end

function _ABXfree_optimal_φ_R(u, p::PrecursorABXwAnticipation)
    """
    computed symbolically
    using stateful values
    """
    _, _, φ_AR, α, _, _ = u
    @unpack_PrecursorABXBaseModel p.abx
    @unpack αmax, αmin = p.ant
    reserve = αmin

    prefactor = 1 - φ_0 - φ_AR
    add = νmax + γmax
    mul = νmax * γmax
    sub = γmax - νmax
    numer = mul * (1 - 2 * Km) + νmax^2 + sqrt(Km * mul) * sub
    denom = add^2 - 4 * Km * mul
    return prefactor * numer / denom
end

function analytic_optimal_growth(p::PrecursorModelFamily)
    φ_R = optimal_φ_R(p)
    φ_M = 1 .- φ_R .- p.φ_0
    Γ = p.γmax .* φ_R
    prefactor = 1 / (2 * (1 - p.Km))
    N = p.νmax .* φ_M
    return prefactor * (Γ .+ N .- sqrt.((N .+ Γ) .^ 2 - 4 * (1 - p.Km) .* N .* Γ))
end

function ABXfree_optimal_φ_R(p::PrecursorABXwAnticipation)
    """
    computed symbolically
    assuming steady-state/equilibrated values
    """
    @unpack_PrecursorABXBaseModel p.abx
    @unpack αmax, αmin = p.ant
    reserve = αmin

    prefactor = 1 - φ_0 - reserve
    add = νmax + γmax
    mul = νmax * γmax
    sub = γmax - νmax
    numer = mul * (1 - 2 * Km) + νmax^2 + sqrt(Km * mul) * sub
    denom = add^2 - 4 * Km * mul
    return prefactor * numer / denom
end

function ABXfree_optimal_φ_R(p::ExpandedPrecursorModels)
    """
    computed symbolically
    assuming steady-state/equilibrated values
    """

    prefactor = 1 - p.φ_0
    add = p.νmax + p.γmax
    mul = p.νmax * p.γmax
    sub = p.γmax - p.νmax
    numer = mul * (1 - 2 * p.Km) + p.νmax^2 + sqrt(p.Km * mul) * sub
    denom = add^2 - 4 * p.Km * mul
    return prefactor * numer / denom
end

function get_phi_AR(soln, n_abx)
    """
    Extracts the φ_AR values from the solution vector from DE soln
    of PrecursorMultiABXwAnticipation problem
    """
    # Extract φ_AR values from the solution
    φ_ARs = soln[3+3*n_abx:2+4*n_abx, :]
    return φ_ARs
end