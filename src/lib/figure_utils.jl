using Printf
using Plots
using ColorSchemes
using PerceptualColourMaps
using Colors



const green = colorant"#04C13C"
const brown = colorant"#8B4513"
const orange = colorant"#FFA500"
const purp = colorant"#BA79BE"
const tangerine = colorant"#EA8407"
const magenta = colorant"#D84691"
const sand = colorant"#EABD47"
const sky = colorant"#3685D9"
const lago = colorant"#23B7F1"
const teel = colorant"#2BE3CE"
const deepbrown = colorant"#80350E"
const plum = colorant"#D088DD"

const purp = colorant"#BA79BE"
const pink = colorant"#FA007B"
const blue = colorant"#05A8C9"

precursor_color = plum
C_bnd_color = orange
C_in_color = brown
φR_color = sky
αR_color = lago
φR_free_color = teel
φM_color = magenta
φAR_color = tangerine
αAR_color = sand
φ0_color = deepbrown
growth_color = green


const thick_line = 3.5
const thin_line = 2
const extra_thin_line = 1
const opt_color = :gray

ValidModels = Union{PrecursorModel,PrecursorABXwAnticipation}

function cache_plot_data(filename, f, args...; overwrite=false, kwargs...)
    local_dir = pwd()
    cache = joinpath(local_dir, "cache")
    if filename in readdir(cache) && overwrite != true
        println("Loading data from saved .jld2")
        stored_data = load(joinpath(cache, filename))
        p = stored_data["p"]
        data = stored_data["d"]
    else
        println("Simulating...")
        p, data = f(args...; kwargs...)
        x = Dict("p" => p, "d" => data)
        println("Saving data to $cache/$filename")
        save(joinpath(cache, filename), x)
    end
    return p, data
end

function plot_alternating_conditions(plot_handle, period, duty, t_final; env_labels=["", ""], colors=[:purple, :lightgray])
    p = plot_handle
    T = period
    t_switch = duty * period
    N = t_final ÷ period
    last_period = t_final % period

    vspan!(p, [0, t_switch], alpha=0.2, color=colors[1], label=env_labels[1])
    vspan!(p, [t_switch, T], alpha=0.2, color=colors[2], label=env_labels[2])
    if N > 1
        for i = 2:N
            j = i - 1
            vspan!(p, [T * j, t_switch + T * j], alpha=0.2, color=colors[1], label="")
            vspan!(p, [t_switch + T * j, T * i], alpha=0.2, color=colors[2], label="")
        end
    end
    if last_period > t_switch
        j = N
        vspan!(p, [T * j, t_switch + T * j], alpha=0.2, color=colors[1], label="")
        vspan!(p, [t_switch + T * j, t_final], alpha=0.2, color=colors[2], label="")
    else
        j = N
        vspan!(p, [T * j, t_final], alpha=0.2, color=colors[1], label="")
    end
    p
end

function plot_overexpression_curves(
    df, m::PrecursorABXwAnticipation
)
    style_guide = Dict("RDM_gluc" => "red", "cAA_gluc" => "blue", "M63_gluc" => "green", "M9_glycerol" => "pink", "M9_glycerol_EF-Tu" => "brown")
    used_ids = Set()
    df[:, :id] .= ""
    plt = plot(xlabel="ϕ_z", ylabel="λ (1/hr)")
    for row in eachrow(df)
        id = row.media
        if row.protein != "beta-galactosidase"
            id *= "_$(row.protein)"
        end
        if !(id in used_ids)
            label = id
            push!(used_ids, id)
        else
            label = ""
        end
        scatter!(
            plt, [row.unnecessary_protein_frac] ./ 100, [row.growth_rate],
            label=label, color=style_guide[id]
        )
        row.id = id
    end

    φ_R0_guide = Dict(
        "RDM_gluc" => 0.215, "cAA_gluc" => 0.14, "M63_gluc" => 0.1,
        "M9_glycerol" => 0.15, "M9_glycerol_EF-Tu" => 0.16
    )
    # select the data point with minimal overexpression for each idxs
    phi_Z_range = range(0.0, 0.35, length=100) .+ m.abx.φ_0
    for id in used_ids
        sub_df = filter(row -> row.id == id, df)
        min_row_idx = argmin(sub_df.unnecessary_protein_frac)
        min_row = sub_df[min_row_idx, :]
        nu = find_nu(φ_R0_guide[id], min_row.growth_rate, m)
        abx_models = mutate(m.abx, φ_0=phi_Z_range, νmax=nu)
        models = PrecursorABXwAnticipation.(abx_models, Ref(m.ant))
        λs = analytic_optimal_growth.(models)
        plot!(plt, phi_Z_range .- m.abx.φ_0, λs, color=style_guide[id], label="")
    end
    return plt
end

Dai_colors = [
    colorant"seagreen", colorant"gold2", colorant"darkorange1",
    colorant"violetred2", colorant"cornflowerblue"
]
Dai6_colors = [
    colorant"seagreen", colorant"gold2", colorant"darkorange1",
    colorant"violetred2", colorant"turquoise", colorant"cornflowerblue"
]
Dai_shapes = [
    :circle, :diamond, :square, :star, :utriangle,
]

function plot_GR_vs_ABX_Dai_curves(
    φR_series, λ_series, abx_series, label, params::PrecursorABXwAnticipation;
    n=51, color=colorant"cornflowerblue", plot_handle=nothing,
    opt=ABX_optimizer_v2, opt_kwargs=Dict(), plot_kwargs=Dict()
)
    states, abxs = compute_GR_vs_ABX_Dai_curves(
        φR_series, λ_series, abx_series, params, n=n,
        opt=opt, opt_kwargs=opt_kwargs
    )
    if plot_handle == nothing
        plt = plot(xlabel="Antibiotics ([M])", ylabel="Growth Rate (1/hr)")
    else
        plt = plot_handle
    end
    λs = states[:, 2]
    plot!(plt, abxs, λs, color=color, label="") # label="No resistance (fit kp)"
    scatter!(plt, abx_series, λ_series, label=label, color=color; plot_kwargs...)
    return plt, abxs, λs
end

function plot_GR_vs_ABX_Dai_curves(
    data_dict, params;
    n=51, opt=ABX_optimizer_v2, opt_kwargs=Dict(), plot_kwargs=Dict()
)
    plt = plot(xlabel="Antibiotics ([M])", ylabel="Growth Rate (1/hr)")
    out_dict = Dict()
    # set order and color based on number of series
    # NB: strings in order must match the keys in data_dict
    if length(data_dict) == 6
        colors = range(magenta, colorant"yellow3",length=6)
        order = ["Rich Media", "This study", "Glucose 0.2%", "Fructose", "Acetate", "Aspartate"]
    elseif length(data_dict) == 5
        colors = Dai_colors
        order = ["Rich Media", "Glucose 0.2%", "Fructose", "Acetate", "Aspartate"]
    else
        # raise error
        error("Invalid number of series in data_dict, must be 5 or 6: $(length(data_dict))")
    end

    for (i, key) in enumerate(order)
        value = data_dict[key]
        abx_series = value[1]
        λ_series = value[2]
        φR_series = value[3]
        plt, abxs, λs = plot_GR_vs_ABX_Dai_curves(
            φR_series, λ_series, abx_series, key, params, n=n, color=colors[i], plot_handle=plt,
            opt=opt, opt_kwargs=opt_kwargs, plot_kwargs=plot_kwargs
        )
        out_dict[key] = [abxs, λs]
    end
    plot!(plt, legend=:topright; plot_kwargs...)
    return plt, out_dict
end

function plot_GR_vs_pR_Dai_curves(
    φR_series, λ_series, abx_series, label, params::PrecursorABXwAnticipation;
    num_sim_pts=51, color=colorant"cornflowerblue", plot_handle=nothing,
    truncation_size=2, shape=:circle
)
    simulated_traces = compute_GR_vs_pR_Dai_curves(φR_series, λ_series, abx_series, params, num_sim_pts=num_sim_pts)
    # truncate bc the extreme values sometimes diverge
    φ_Rs_simulated = simulated_traces[:, 1]
    λs_simulated = simulated_traces[:, 2]

    num_data_pts = length(abx_series)
    colorscale = gen_color_range(num_data_pts, color)

    if plot_handle == nothing
        plt = plot(xlabel="ϕ_R", ylabel="Growth Rate (1/hr)")
    else
        plt = plot_handle
    end

    # truncate the simulated traces
    for i in 1:num_data_pts
        φ_Rs_simulated[i] = φ_Rs_simulated[i][1:end-truncation_size]
        λs_simulated[i] = λs_simulated[i][1:end-truncation_size]
    end

    for i in 1:num_data_pts
        plot!(plt, φ_Rs_simulated[i], λs_simulated[i], color=colorscale[i], label="")
        scatter!(
            plt, [φR_series[i]], [λ_series[i]], color=colorscale[i], marker=shape,
            label="$label with $( round(10^6 * abx_series[i], sigdigits=1) ) μM Cm"
        )
    end
    plot!(plt, label=label)
    return plt, φ_Rs_simulated, λs_simulated
end

function plot_GR_vs_pR_Dai_curves(
    φR_series, λ_series, abx_series, label, cached_data::Dict;
    color=colorant"cornflowerblue", plot_handle=nothing,
    shape=:circle
)
    simulated_traces = cached_data[label]
    φ_Rs_simulated = simulated_traces[1]
    λs_simulated = simulated_traces[2]

    num_data_pts = length(abx_series)
    colorscale = gen_color_range(num_data_pts, color)

    if plot_handle == nothing
        plt = plot(xlabel="ϕ_R", ylabel="Growth Rate (1/hr)")
    else
        plt = plot_handle
    end

    for i in 1:num_data_pts
        plot!(plt, φ_Rs_simulated[i], λs_simulated[i], color=colorscale[i], label="")
        scatter!(
            plt, [φR_series[i]], [λ_series[i]], color=colorscale[i], marker=shape,
            label="$label with $( round(10^6 * abx_series[i], sigdigits=1) ) μM Cm"
        )
    end
    plot!(plt, label=label)
    return plt, φ_Rs_simulated, λs_simulated
end

function plot_GR_vs_pR_Dai_curves(
    data_dict, params::PrecursorABXwAnticipation; num_sim_pts=51, truncation_size=2,
    select=nothing
)
    plt = plot(xlabel="ϕ_R", ylabel="Growth Rate (1/hr)")
    out_dict = Dict()
    for (i, (key, value)) in enumerate(data_dict)
        if select != nothing && !(select == key)
            continue
        end
        abx_series = value[1]
        λ_series = value[2]
        φR_series = value[3]
        plt, φ_Rs, λs = plot_GR_vs_pR_Dai_curves(
            φR_series, λ_series, abx_series, key, params, num_sim_pts=num_sim_pts,
            plot_handle=plt, truncation_size=truncation_size, color=Dai_colors[i],
            shape=Dai_shapes[i]
        )
        out_dict[key] = [φ_Rs, λs]
    end
    plot!(plt, legend=:outerright)
    return plt, out_dict
end

function plot_GR_vs_pR_Dai_curves(
    data_dict, sim_data::Dict;
    select=nothing
)
    plt = plot(xlabel="ϕ_R", ylabel="Growth Rate (1/hr)")
    out_dict = Dict()
    for (i, (key, value)) in enumerate(data_dict)
        if select != nothing && !(select == key)
            continue
        end
        abx_series = value[1]
        λ_series = value[2]
        φR_series = value[3]
        plt, φ_Rs, λs = plot_GR_vs_pR_Dai_curves(
            φR_series, λ_series, abx_series, key, sim_data,
            plot_handle=plt, color=Dai_colors[i],
            shape=Dai_shapes[i]
        )
        out_dict[key] = [φ_Rs, λs]
    end
    plot!(plt, legend=:outerright)
    return plt, out_dict
end

function plot_param_sweep_step(
    sym::Symbol, u0, params::PrecursorABXwAnticipation,
    jc::JumpConfig, p_range
)
    GRs, jc_step = param_sweep_step(sym, u0, params, jc, p_range)
    ts = gen_ts(jc_step)
    labs = ["$sym = $(Printf.@sprintf("%.2e", x))" for x in p_range]
    p = λ_plot(ts, Matrix(GRs'), labels=labs)
    plot!(p, legend=:outerright, title="Growth Rate for Responsive Strategies")
    return p, GRs
end

function plot_param_sweep_step(
    syms_dict::Dict, u0, params::PrecursorABXwAnticipation,
    jc::JumpConfig, synced::Bool=true
)
    GRs, jc_step = param_sweep_step(syms_dict, u0, params, jc, synced)
    ts = gen_ts(jc_step)
    ref_key = first(keys(syms_dict))
    labs = ["$ref_key = $(Printf.@sprintf("%.2e", x))" for x in syms_dict[ref_key]]
    p = λ_plot(ts, Matrix(GRs'), labels=labs)
    plot!(p, legend=:outerright, title="Growth Rate for Responsive Strategies")
    return p, GRs
end

function plot_DE_soln(soln, p::tRNAModelFamily)
    # variables are φ_R, r_ch, r_uc, φ_M
    # plot φ's, r's, and growth rate

    t = soln.t
    p1 = plot(t, soln[1, :], label="φ_R", size=(600, 600))
    plot!(t, soln[4, :], label="φ_M", ylabel="Mass Fraction")
    p2 = plot(t, soln[2, :], label="Charged tRNA")
    plot!(t, soln[3, :], label="Uncharged tRNA", ylabel="Concentration [M]", size=(600, 600))

    γ = monod(p.γmax, p.Km)
    φ_R = soln[1, :]
    r_ch = soln[2, :]
    gr = γ.(r_ch) .* φ_R
    p3 = plot(t, gr, xlabel="Time (hr)", ylabel="Growth Rate (1/hr)", label="", size=(600, 600))
    p = plot(p1, p2, p3, layout=(3, 1), suptitle="tRNA Model")
    return p
end

function plot_DE_soln(soln, p::PrecursorModelFamily)
    t = soln.t
    p1 = plot(t, soln[1, :], label="φ_R", ylabel="Mass Fraction", size=(600, 600))
    p2 = plot(t, soln[2, :], label="Precursors", ylabel="Concentration ([M])", size=(600, 600))

    γ = monod(p.γmax, p.Km)
    φ_R = soln[1, :]
    pc = soln[2, :]
    gr = γ.(pc) .* φ_R
    p3 = plot(t, gr, xlabel="Time (hr)", ylabel="Growth Rate (1/hr)", label="", size=(600, 600))
    p = plot(p1, p2, p3, layout=(3, 1), suptitle="Precursor Model")
    return p
end

function plot_DE_soln(soln, p::AbxModels, plot_intracellular_concs=false)
    t = soln.t

    filler = ones(length(t))
    φ_R = soln[1, :]
    C_bnd = soln[6, :]
    φ_R_free = φ_R - C_bnd / p.abx.n_prot
    φ_AR = soln[4, :]
    φ_M = 1 .- φ_R - φ_AR .- p.abx.φ_0
    p1 = plot(t, φ_R, label="Total φ_R", ylabel="Mass Fraction", size=(600, 600))
    plot!(t, φ_R_free, label="Free φ_R")
    plot!(t, φ_AR, label="φ_AR")
    plot!(t, p.abx.φ_0 * filler, label="φ_0")
    plot!(t, φ_M, label="φ_M")

    p2 = plot(t, soln[2, :], label="Charged tRNA", ylabel="Concentration ([M])", size=(600, 600))
    plot!(t, soln[3, :], label="Uncharged tRNA")

    colors = ColorSchemes.Dark2_6
    p3 = plot(
        t, soln[5, :], label="α", color=colors[1],
        y_guidefontcolor=colors[1], ytickfontcolor=colors[1],
        ylabel="Fraction of Biosynthesis", size=(600, 600),
        xlabel="Time (hr)",
    )
    if p == AbxAntANDRiboRegulation
        ppGpp = p.ppGpp(soln, p)
    else
        # r_uc / r_ch
        ppGpp = soln[3, :] ./ soln[2, :]
    end
    plot!(
        twinx(p3), t, ppGpp, label="", color=colors[2],
        y_guidefontcolor=colors[2], ytickfontcolor=colors[2],
        ylabel="ppGpp (A.U.)", size=(600, 600)
    )
    plot!(t, t .* NaN, label="ppGpp", color=colors[2])

    γ = monod(p.abx.γmax, p.abx.Km)
    r_ch = soln[2, :]
    gr = γ.(r_ch) .* φ_R_free
    p4 = plot(t, gr, xlabel="Time (hr)", ylabel="Growth Rate (1/hr)", label="", size=(600, 600))

    if plot_intracellular_concs
        p5 = plot(t, soln[7, :], label="C_in", ylabel="Concentration ([M])", size=(600, 600))
        plot!(t, soln[6, :], label="C_bnd")
        C_R_total = φ_R .* p.abx.n_prot
        plot!(t, C_R_total, label="Total C_R")
        plt = plot(p1, p2, p3, p4, p5, layout=(5, 1), suptitle="Abx Model")
    else
        plt = plot(p1, p2, p3, p4, layout=(2, 2), suptitle="Abx Model", size=(900, 750))
    end
    return plt
end

function plot_DE_soln_v2(soln, p::PrecursorABXwAnticipation)
    t = soln.t
    φ_R = soln[1, :]
    cpc = soln[2, :]
    α_R = φ_R_heuristic.(cpc, p.abx)
    φ_F = 1 - p.abx.φ_0 .- φ_R
    α_AR = soln[5, :]
    φ_AR = soln[6, :]
    C_bnd = soln[3, :]
    p1 = plot(t, φ_R, label="φ_R", ylabel="Mass Fraction", size=(600, 600))
    plot!(t, φ_AR, label="φ_AR")
    φ_M = 1 .- φ_R .- φ_AR .- p.abx.φ_0
    plot!(t, φ_M, label="φ_M")
    # plot!(t, p.abx.φ_0 .* ones(length(t)), label="φ_0")
    φ_Rf = φ_R - C_bnd ./ p.abx.n_prot
    plot!(t, φ_Rf, label="φ_R free", legend=:outerright)
    φ_total = φ_R + φ_AR + φ_M .+ p.abx.φ_0
    # plot!(t, φ_total, label="Total φ")

    p2 = plot(t, cpc, label="Precursors", ylabel="Concentration ([M])", size=(600, 600))

    γ = monod(p.abx.γmax, p.abx.Km)
    gr = γ.(cpc) .* φ_Rf
    p3 = plot(t, gr, xlabel="Time (hr)", ylabel="Growth Rate (1/hr)", label="", size=(600, 600))
    # optimal_gr = analytic_optimal_growth(p)
    # doesn't work nicely because of changing C_max -- ambiguity in which
    # C_max to use for optimal calculation
    # probably ought to use the max C_max(t) for a useful comparison
    # abx_opt_gr = analytic_optimal_growth(p, abx=true)
    #plot!(t, optimal_gr .* ones(length(t)), label="Optimal ABX-free", linestyle=:dash)
    #plot!(t, abx_opt_gr .* ones(length(t)), label="Optimal ABX", linestyle=:dash, legend=:topright)
    p = plot(p1, p2, p3, layout=(3, 1), suptitle="Precursor Model w ABX")
    return p
end

function plot_DE_soln_deprecated(soln, p::PrecursorABXwAnticipation)
    t = soln.t
    φ_R = soln[1, :]
    C_bnd = soln[5, :]
    φ_AR = soln[3, :]
    p1 = plot(t, φ_R, label="φ_R", ylabel="Mass Fraction", size=(600, 600))
    plot!(t, φ_AR, label="φ_AR")
    φ_M = 1 .- φ_R .- φ_AR .- p.abx.φ_0
    plot!(t, φ_M, label="φ_M")
    plot!(t, p.abx.φ_0 .* ones(length(t)), label="φ_0")
    φ_Rf = φ_R - C_bnd ./ p.abx.n_prot
    plot!(t, φ_Rf, label="φ_R free", legend=:topright)
    φ_total = φ_R + φ_AR + φ_M .+ p.abx.φ_0
    # plot!(t, φ_total, label="Total φ")

    p2 = plot(t, soln[2, :], label="Precursors", ylabel="Concentration ([M])", size=(600, 600))

    γ = monod(p.abx.γmax, p.abx.Km)
    pc = soln[2, :]
    gr = γ.(pc) .* φ_Rf
    p3 = plot(t, gr, xlabel="Time (hr)", ylabel="Growth Rate (1/hr)", label="", size=(600, 600))
    optimal_gr = analytic_optimal_growth(p)
    abx_opt_gr = analytic_optimal_growth(p, abx=true)
    plot!(t, optimal_gr .* ones(length(t)), label="Optimal ABX-free", linestyle=:dash)
    plot!(t, abx_opt_gr .* ones(length(t)), label="Optimal ABX", linestyle=:dash, legend=:topright)
    p = plot(p1, p2, p3, layout=(3, 1), suptitle="Precursor Model w ABX")
    return p
end

function plot_DE_soln(soln, p::PrecursorABXConstantRiboWAnt)
    t = soln.t
    φ_R = soln[1, :]
    C_bnd = soln[5, :]
    φ_AR = soln[3, :]
    p1 = plot(t, φ_R, label="φ_R", ylabel="Mass Fraction", size=(600, 600))
    plot!(t, φ_AR, label="φ_AR")
    φ_M = 1 .- φ_R .- φ_AR .- p.abx.φ_0
    plot!(t, φ_M, label="φ_M")
    plot!(t, p.abx.φ_0 .* ones(length(t)), label="φ_0")
    φ_Rf = φ_R - C_bnd ./ p.abx.n_prot
    plot!(t, φ_Rf, label="φ_R free", legend=:topright)
    φ_total = φ_R + φ_AR + φ_M .+ p.abx.φ_0
    # plot!(t, φ_total, label="Total φ")

    p2 = plot(t, soln[2, :], label="Precursors", ylabel="Concentration ([M])", size=(600, 600))

    γ = monod(p.abx.γmax, p.abx.Km)
    pc = soln[2, :]
    gr = γ.(pc) .* φ_Rf
    p3 = plot(t, gr, xlabel="Time (hr)", ylabel="Growth Rate (1/hr)", label="", size=(600, 600))
    p = plot(p1, p2, p3, layout=(3, 1), suptitle="Precursor Model w ABX")
    return p
end

function plot_many_αR(solns, ps::Vector{PrecursorABXConstantRiboWAnt}; colors)
    # TODO - colors not working for some reason??
    oneDsize = 700
    plotsize = (oneDsize, oneDsize)
    p1 = plot(ylabel="φ_R", size=plotsize, linecolor=colors)
    p2 = plot(ylabel="Free Ribosomes (%)", size=plotsize, linecolor=colors)
    p3 = plot(ylabel="φ_AR", size=plotsize, linecolor=colors)
    p4 = plot(ylabel="φ_M", size=plotsize, linecolor=colors)
    p5 = plot(
        ylabel="Precursors ([M])", xlabel="Time (hr)",
        size=plotsize, linecolor=colors
    )
    p6 = plot(ylabel="Growth Rate (1/hr)", xlabel="Time (hr)", size=plotsize, linecolor=colors)



    for i in 1:length(solns)
        soln = solns[i]
        p = ps[i]
        ar = p.abx.α_R

        t = soln.t
        φ_R = soln[1, :]
        C_bnd = soln[5, :]
        φ_AR = soln[3, :]
        φ_M = 1 .- φ_R .- φ_AR .- p.abx.φ_0
        φ_Rf = φ_R - C_bnd ./ p.abx.n_prot
        pc = soln[2, :]
        γ = monod(p.abx.γmax, p.abx.Km)
        gr = γ.(pc) .* φ_Rf
        p1 = plot!(p1, t, φ_R, label="", line_z=ar)
        p2 = plot!(p2, t, φ_Rf ./ φ_R .* 100, label="", line_z=ar)
        p3 = plot!(p3, t, φ_AR, label="", line_z=ar)
        p4 = plot!(p4, t, φ_M, label="", line_z=ar)
        p5 = plot!(p5, t, pc, label="", line_z=ar)
        p6 = plot!(p6, t, gr, label="", line_z=ar)
    end

    p = plot(p1, p4, p3, p2, p5, p6, layout=(3, 2), suptitle="Precursor Model w ABX")
    return p
end

function plot_JuMP_soln(soln, p::PrecursorModel, jc::JumpConfig; subset_idxs=nothing)
    Rs, cpcs, aRs, Ls = parse_soln(soln, p)
    normed_cpcs = cpcs ./ p.Km

    ts = cumsum([0; repeat([jc.Δt], jc.n)])[1:end-1]

    plt_growth = λ_plot(ts, Ls, p; subset_idxs=subset_idxs)
    plt_pfractions = φR_plot(ts, Rs, aRs, p; subset_idxs=subset_idxs)
    plot_pcursors = precursor_plot(ts, normed_cpcs, p; subset_idxs=subset_idxs)

    p = plot(plt_pfractions, plt_growth, plot_pcursors; layout=grid(3, 1), size=(700, 700), legend=:outerright)
    return [plt_pfractions, plt_growth, plot_pcursors]
end

function parse_soln(soln, ::PrecursorModelFamily)
    φ_Rs = soln[1]
    cpcs = soln[2]
    aRs = soln[3]
    Ls = soln[4]
    return φ_Rs, cpcs, aRs, Ls
end

function parse_soln(soln, jc, ::PrecursorABXBaseModel)
    Rs = soln[1]
    aRs = soln[2]
    cpcs = soln[3]
    C_bnds = soln[4]
    C_ins = soln[5]
    Ls = soln[6]
    if length(soln) == 6
        ts = cumsum([0; repeat([jc.Δt], jc.n)])[1:end-1]
    elseif length(soln) == 7
        ts = soln[7]
    else
        error("soln must have 6 or 7 elements")
    end

    return Rs, aRs, cpcs, C_bnds, C_ins, Ls, ts
end

function parse_soln(soln, jc, p::PrecursorABXwAnticipation)
    if length(soln) > 20
        @warn "soln has more than 20 elements, assuming it's a DE solution"
        return parse_soln(soln, p)
    end
    """ Parses a JuMP solution"""
    Rs = soln[1]
    aRs = soln[2]
    ARs = soln[3]
    a_ARs = soln[4]
    cpcs = soln[5]
    C_bnds = soln[6]
    C_ins = soln[7]
    Ls = soln[8]
    if length(soln) == 8
        ts = cumsum([0; repeat([jc.Δt], jc.n)])[1:end-1]
    elseif length(soln) == 9
        ts = soln[9]
    else
        error("soln must have 8 or 9 elements")
    end

    return Rs, aRs, ARs, a_ARs, cpcs, C_bnds, C_ins, Ls, ts
end

function parse_soln(soln, p::PrecursorABXwAnticipation)
    """ Parses a DifferentialEquations solution"""
    ts = soln.t
    φ_R = soln[1, :]
    cpc = soln[2, :]
    α_R = φ_R_heuristic.(cpc, p.abx)
    φ_AR = soln[6, :]
    α_AR = soln[5, :]
    C_bnds = soln[3, :]
    C_ins = soln[4, :]
    φ_Rf = φ_R - C_bnds ./ p.abx.n_prot

    γ = monod(p.abx.γmax, p.abx.Km)
    Ls = γ.(cpc) .* φ_Rf
    return φ_R, α_R, φ_AR, α_AR, cpc, C_bnds, C_ins, Ls, ts
end

function parse_soln(soln, p::PrecursorMultiABXwAnticipation)
    """ Parses a DifferentialEquations solution"""
    ts = soln.t
    φ_R = soln[1, :]
    cpc = soln[2, :]
    α_R = φ_R_heuristic.(cpc, p.abx)
    
    n_abx = length(p.ant)
    # Extract antibiotic-specific variables
    C_bnd_vecs = [soln[2+i, :] for i in 1:n_abx]
    C_in_vecs = [soln[2+n_abx+i, :] for i in 1:n_abx]
    α_AR_vecs = [soln[2+2*n_abx+i, :] for i in 1:n_abx]
    φ_AR_vecs = [soln[2+3*n_abx+i, :] for i in 1:n_abx]

    # Calculate free ribosome fraction
    total_C_bnds = zeros(length(ts))
    C_bnds_mtx = hcat(C_bnd_vecs...)
    for i in 1:length(ts)
        total_C_bnds[i] = calculate_total_C_bnd(φ_R[i], C_bnds_mtx[i,:], p.abx.n_prot)
    end
    φ_Rf = φ_R .- total_C_bnds ./ p.abx.n_prot

    # Calculate growth rate
    γ = monod(p.abx.γmax, p.abx.Km)
    Ls = γ.(cpc) .* φ_Rf

    return [
        φ_R, α_R, cpc, Ls, ts, φ_AR_vecs..., α_AR_vecs..., 
        C_bnd_vecs..., C_in_vecs...
    ]
end

function gen_ts(jc)
    ts = cumsum([0; repeat([jc.Δt], jc.n)])[1:end-1]
    return ts
end

function parse_solns(solns, jc, ::Vector{PrecursorABXwAnticipation})
    soln_array = convert_nested_to_3D_array(solns)
    ts = gen_ts(jc)

    return soln_array, ts
end

function convert_nested_to_3D_array(nested_vec)
    arr = [
        nested_vec[i][j][k] for i in 1:length(nested_vec),
        j in 1:length(nested_vec[1]),
        k in 1:length(nested_vec[1][1])
    ]
    return arr
end

function plot_many_solns(
    soln_set,
    param_set::Vector{PrecursorABXwAnticipation}, jc,
    labels=nothing
)
    solns, ts = parse_solns(soln_set, jc, param_set)
    # norm the cpcs
    Kms = [p.abx.Km for p in param_set]
    normed_cpcs = Matrix((solns[:, 5, :] ./ Kms)')

    Ls = Matrix(solns[:, 8, :]')
    C_bnds = Matrix(solns[:, 6, :]')
    C_ins = Matrix(solns[:, 7, :]')
    plt_growth = λ_plot(ts, Ls, labels=labels)
    plt_pcursors = precursor_plot(ts, normed_cpcs, xlab=true, labels=labels)
    plt_ABX = ABX_plot(ts, C_ins, C_bnds, xlab=true, labels=labels)
    plt_pfractions = massfraction_plot(solns, jc, param_set[1], labels=labels)

    p = plot(
        plt_pfractions, plt_growth, plt_pcursors, plt_ABX;
        layout=grid(4, 1), size=(700, 700), legend=:outerright
    )
    return [plt_pfractions, plt_growth, plt_pcursors, plt_ABX]
end

function plot_JuMP_soln(
    soln, p::PrecursorABXBaseModel, jc::JumpConfig
)
    Rs, aRs, cpcs, C_bnds, C_ins, Ls, ts = parse_soln(soln, jc, p)
    normed_cpcs = cpcs ./ p.Km

    plt_growth = λ_plot(ts, Ls)
    plt_pfractions = φR_plot(ts, Rs, aRs, plot_φM=true, φ0=p.φ_0)
    plt_pcursors = precursor_plot(ts, normed_cpcs, xlab=true)
    plt_ABX = ABX_plot(ts, C_ins, C_bnds, xlab=true)

    p = plot(
        plt_pfractions, plt_growth, plt_pcursors, plt_ABX;
        layout=grid(4, 1), size=(700, 700), legend=:outerright
    )
    return [plt_pfractions, plt_growth, plt_pcursors, plt_ABX]
end


function plot_phiR(soln, p::PrecursorABXwAnticipation)
    Rs, aRs, ARs, a_ARs, cpcs, C_bnds, C_ins, Ls, ts = parse_soln(soln, p)
    normed_cpcs = cpcs ./ p.abx.Km

    plt = plot_series(
        ts, Rs, φR_color;
        xlab="Elapsed Minutes Since Antibiotic Shock", ylab="Ribosomal Protein Mass Fraction",
        title="",
        label="Simulated φ_R"
    )
    return plt
end

function plot_DE_soln(
    soln, p::PrecursorABXwAnticipation; plot_AR=true, integrate_GR=false,
    fluct_params = nothing
)
    # TODO - implement GR integration to get total biomass
    Rs, aRs, ARs, a_ARs, cpcs, C_bnds, C_ins, Ls, ts = parse_soln(soln, p)
    normed_cpcs = cpcs ./ p.abx.Km

    null_jc = JumpConfig(0, 0, RectangularIntegrationRule())
    plt_growth = λ_plot(
        ts, Ls, integrate=integrate_GR, fluct_params=fluct_params
    )
    plt_pfractions = massfraction_plot(
        soln, null_jc, p, plot_aR=plot_AR, fluct_params=fluct_params
    )
    plt_pcursors = precursor_plot(ts, normed_cpcs, xlab=true, fluct_params=fluct_params)
    plt_ABX = ABX_plot(ts, C_ins, C_bnds, xlab=true, fluct_params=fluct_params)
    all_plots = [plt_pfractions, plt_growth, plt_pcursors, plt_ABX]

    p = plot(
        plt_pfractions, plt_growth, plt_pcursors, plt_ABX;
        layout=grid(4, 1), size=(700, 700), legend=:outerright
    )
    return all_plots
end

function plot_DE_soln(
    soln, p::PrecursorMultiABXwAnticipation; plot_aR=true, integrate_GR=false,
    fluct_params_vec = nothing, v_lines=false
)
    u = parse_soln(soln, p)
    n_abx = (length(u) - 5) ÷ 4
    Rs = u[1]
    aRs = u[2]
    cpcs = u[3]
    Ls = u[4]
    ts = u[5]
    ARs = u[6:5+n_abx]
    a_ARs = u[6+n_abx:5+2*n_abx]
    C_bnds = u[6+2*n_abx:5+3*n_abx]
    C_ins = u[6+3*n_abx:5+4*n_abx]


    normed_cpcs = cpcs ./ p.abx.Km

    plt_growth = λ_plot(ts, Ls, integrate=integrate_GR)
    plt_pfractions = massfraction_plot_multiABX(soln, p, plot_aR=plot_aR)
    plt_pcursors = precursor_plot(ts, normed_cpcs, xlab=true)
    plt_ABX = multiABX_plot(ts, C_ins, C_bnds, xlab=true)
    all_plots = [plt_pfractions, plt_growth, plt_pcursors, plt_ABX]
    if !isnothing(fluct_params_vec)
        for plt in all_plots
            add_multi_ABX_annotations!(plt, fluct_params_vec, v_lines=v_lines)
        end
    end

    p = plot(
        plt_pfractions, plt_growth, plt_pcursors, plt_ABX;
        layout=grid(4, 1), size=(700, 700), legend=:outerright
    )
    return all_plots
end

function add_ABX_annotations!(plt, fluct_params; env_labels=["",""], colors=[:red,:gray])
    # assume t=0 starts with ABX
    # assume duty species fraction of time ABX is on
    T, duty, duration = fluct_params
    t_switch = duty * T
    N = Int(round(duration / T))

    vspan!(plt, [0,t_switch], alpha=0.2, color = colors[1], label=env_labels[1])
    vspan!(plt, [t_switch, T], alpha=0.2, color = colors[2], label=env_labels[2])
    if N > 1
        for i = 2:N 
            j = i - 1
            vspan!(plt, [T*j, t_switch + T*j], alpha=0.2, color = colors[1], label="")
            vspan!(plt, [t_switch + T*j, T*i], alpha=0.2, color = colors[2], label="")
        end
    end
    return plt
end

function add_multi_ABX_annotations!(
    plt, fluct_params_vec; colors=nothing, labels=nothing, v_lines=false,
)
    shock_times = get_shock_start_times(fluct_params_vec)
    n_abx = length(fluct_params_vec)
    widths = zeros(n_abx)
    for (i, fp) in enumerate(fluct_params_vec)
        period, duty, _, _ = fp
        widths[i] = period * duty
    end
    if isnothing(labels)
        labels = ["ABX $i" for i in 1:n_abx]
    elseif !labels
        labels = nothing
    end
    if isnothing(colors)
        colors = gen_color_range(n_abx, colorant"brown2", colorant"gray34")
    end
    if v_lines
        for i in 1:n_abx
            vline!(plt, shock_times[i], color=colors[i], ls=:dashdot, label="", alpha=0.5)
            vline!(plt, shock_times[i] .+ widths[i], color=colors[i], ls=:dashdot, label="", alpha=0.5)
        end
    end

    p = add_rectangles!(
        plt, shock_times, colors=colors, rect_width=widths, labels=labels,
    )
    return p
end

function get_shock_start_times(fluct_params_vec)
	n_abx = length(fluct_params_vec)
	events = Vector{Vector{Float64}}(undef, n_abx)
	for (i, fp) in enumerate(fluct_params_vec)
		period, duty, duration, t0 = fp
		x = t0
		n_events = 1
		while x < duration
			n_events += 1
			x += period
		end
		events[i] = [t0 + (j - 1)*period for j in 1:n_events]
	end
	return events
end

function add_rectangles!(
    plt, 
    x_starts;
    position=:top, 
    colors=nothing, 
    rect_width=1.0,        # Can be a single value or a vector (one per row)
    row_height=0.03,       # Height per row as fraction of y-range
    opacity=0.8,
    row_padding=0.01,      # Padding between rows
    plot_margin=0.04,      # ← NEW: Extra space between plot and first rectangle row
    labels=nothing         # Optional labels for rows
)
    """
    Add multiple rows of non-overlapping, fixed-width rectangles outside a plot.
    
    Parameters:
    - plt: The plot to modify
    - x_starts: Vector of vectors, where each inner vector contains starting x-positions
                for rectangles in a single row
    - position: :top or :bottom - where to place the rectangles
    - colors: Vector of colors (one per row) or single color for all
    - rect_width: Width of each rectangle (same for all)
    - row_height: Height of each row as a fraction of the y-range
    - opacity: Opacity of the rectangles
    - row_padding: Padding between rows as a fraction of the y-range
    - plot_margin: Extra space between the plot and the first row of rectangles
    - labels: Optional labels for each row (Vector of strings)
    
    Returns:
    - Modified plot with rectangles added
    """
    # Get the current y-limits
    y_min, y_max = Plots.ylims(plt)
    y_range = y_max - y_min
    
    # Number of rows
    n_rows = length(x_starts)
    
    # Set default colors if not provided
    if isnothing(colors)
        colors = gen_color_range(n_rows,  colorant"brown2", colorant"gray34")
    elseif length(colors) == 1 || !(colors isa Vector)
        # Single color provided, expand to all rows
        colors = fill(colors, n_rows)
    end

    # Process rect_width - ensure it's a vector with one width per row
    if rect_width isa Number
        rect_widths = fill(rect_width, n_rows)
    elseif length(rect_width) == n_rows
        rect_widths = rect_width
    else
        error("rect_width must be a single value or a vector with one width per row")
    end
    
    # Make sure colors has correct length
    if length(colors) < n_rows
        colors = vcat(colors, distinguishable_colors(n_rows - length(colors), colors))
    end
    
    # Total height needed for all rows, including the extra margin
    total_rect_height = n_rows * row_height + (n_rows - 1) * row_padding + plot_margin
    
    # Extend the y-limits to accommodate rectangles
    if position == :top
        # Rectangles above the plot
        rect_base = y_max + plot_margin * y_range  # ← Add margin before first row
        rect_top = y_max + total_rect_height * y_range
        ylims!(plt, (y_min, rect_top))
    else  # :bottom
        # Rectangles below the plot
        rect_base = y_min - total_rect_height * y_range
        rect_top = y_min - plot_margin * y_range  # ← Add margin before first row
        ylims!(plt, (rect_base, y_max))
    end
    
    # Draw each row of rectangles
    for i in 1:n_rows
        # Calculate y-position for this row, accounting for the extra margin
        if position == :top
            # Rows go from bottom to top, with extra margin before first row
            row_base = y_max + plot_margin * y_range + (i-1) * (row_height + row_padding) * y_range
        else  # :bottom
            # Rows go from top to bottom, with extra margin before first row
            row_base = y_min - plot_margin * y_range - i * row_height * y_range - (i-1) * row_padding * y_range
        end
        
        row_top = row_base + row_height * y_range
        
        # Add label if provided
        if !isnothing(labels) && i <= length(labels) && !isempty(labels[i])
            # Position label at the left side of the row
            x_min, x_max = Plots.xlims(plt)
            label_x = x_min - 0.02 * (x_max - x_min)  # Slightly left of x_min
            label_y = (row_base + row_top) / 2
            
            annotate!(plt, label_x, label_y, text(labels[i], 8, colors[i], :right))
        end
        
        # Draw each rectangle in this row
        for start_x in x_starts[i]
            end_x = start_x + rect_width[i]
            
            # Add the rectangle
            plot!(
                plt,
                [start_x, end_x, end_x, start_x, start_x],
                [row_base, row_base, row_top, row_top, row_base],
                seriestype = :shape,
                fillcolor = colors[i],
                fillalpha = opacity,
                linecolor = colors[i],
                label = nothing
            )
        end
    end

	# Remove the last y-tick
	current_yticks, current_ylabels = Plots.yticks(plt)[1]
		
	# Remove the last tick and its label
	new_yticks = current_yticks[1:end-1]
	new_ylabels = current_ylabels[1:end-1]
	
	# Set the modified ticks with labels
	yticks!(plt, (new_yticks, new_ylabels))
    
    return plt
end

function plot_JuMP_soln(
    soln, p::PrecursorABXwAnticipation, jc::JumpConfig; plot_aR=false
)
    Rs, aRs, ARs, a_ARs, cpcs, C_bnds, C_ins, Ls, ts = parse_soln(soln, jc, p)
    normed_cpcs = cpcs ./ p.abx.Km

    plt_growth = λ_plot(ts, Ls)
    plt_pfractions = massfraction_plot(soln, jc, p, plot_aR=plot_aR)
    plt_pcursors = precursor_plot(ts, normed_cpcs, xlab=true)
    plt_ABX = ABX_plot(ts, C_ins, C_bnds, xlab=true)

    p = plot(
        plt_pfractions, plt_growth, plt_pcursors, plt_ABX;
        layout=grid(4, 1), size=(700, 700), legend=:outerright
    )
    return [plt_pfractions, plt_growth, plt_pcursors, plt_ABX]
end

function plot_GR_comparison(
    solns, p::PrecursorABXwAnticipation, jc::JumpConfig, labels
)
    plt = plot()
    colors = distinguishable_colors(length(solns))
    for i in 1:length(solns)
        _, _, _, _, _, _, _, Ls, ts = parse_soln(solns[i], jc, p)
        plt_growth = λ_plot(
            ts, Ls, labels=labels[i], growth_color=colors[i],
            plot=plt
        )
    end
    return plt
end

function plot_test_soln(
    soln, p::PrecursorABXwAnticipation, jc::JumpConfig
)
    Rs, aRs, cpcs, C_bnds, C_ins, Ls, ts = parse_soln(soln, jc, p.abx)
    normed_cpcs = cpcs ./ p.abx.Km

    plt_growth = λ_plot(ts, Ls)
    plt_pfractions = massfraction_plot(soln, jc, p.abx)
    plt_pcursors = precursor_plot(ts, normed_cpcs, xlab=true)
    plt_ABX = ABX_plot(ts, C_ins, C_bnds, xlab=true)

    p = plot(
        plt_pfractions, plt_growth, plt_pcursors, plt_ABX;
        layout=grid(4, 1), size=(700, 700), legend=:outerright
    )
    return [plt_pfractions, plt_growth, plt_pcursors, plt_ABX]
end

function plot_series(
    ts, xs, color::RGB;
    title, xlab, ylab,
    subset_idxs=nothing, lw=thick_line,
    ls=:solid, label="", sub_labels=nothing,
    distinguish_colors=false, fluct_params=nothing,
    # pass in the rest as kwargs
    kwargs...
)
    xs, n_plots = subset_series(xs, subset_idxs)
    colors = set_of_colors(n_plots, color, distinguish_colors)
    sub_labels = isnothing(sub_labels) ? repeat([""], n_plots) : sub_labels
    if sub_labels isa String
        sub_labels = [sub_labels]
    end
    if length(sub_labels) != n_plots
        @info "sub_labels = $sub_labels"
        @info "n_plots = $n_plots"
        error("sub_labels must be same length as n_plots")
    end

    plt = plot(title=title, xlabel=xlab, ylabel=ylab; kwargs...)
    if !isnothing(fluct_params)
        add_ABX_annotations!(plt, fluct_params)
    end
    for i in 1:n_plots
        if i == n_plots
            lab = label * sub_labels[i]
        else
            lab = sub_labels[i]
        end
        plot!(
            ts,
            xs[:, i];
            label=lab,
            linecolor=colors[i],
            linewidth=lw, linestyle=ls,
        )
    end
    return plt
end

function plot_series!(
    plt, ts, xs, color::RGB;
    subset_idxs=nothing, lw=thick_line,
    ls=:solid, label="", sub_labels=nothing,
    distinguish_colors=false,
    # pass in the rest as kwargs
    kwargs...
)
    xs, n_plots = subset_series(xs, subset_idxs)
    colors = set_of_colors(n_plots, color, distinguish_colors)
    sub_labels = isnothing(sub_labels) ? repeat([""], n_plots) : sub_labels
    if length(sub_labels) != n_plots
        error("sub_labels must be same length as n_plots")
    end

    for i in 1:n_plots
        if i == n_plots
            lab = label * sub_labels[i]
        else
            lab = sub_labels[i]
        end
        plot!(
            plt,
            ts,
            xs[:, i];
            label=lab,
            linecolor=colors[i],
            linewidth=lw, linestyle=ls,
            kwargs...
        )
    end
    return plt
end

function precursor_plot(
    ts, normed_c_pcs;
    subset_idxs=nothing, xlab=false, precursor_color=precursor_color,
    labels=nothing, fluct_params=nothing
)
    if xlab
        xlab = "Time (hrs)"
    else
        xlab = ""
    end
    plt = plot_series(
        ts, normed_c_pcs, precursor_color;
        title="Precursors",
        xlab=xlab,
        ylab="Concentration (Km)",
        ylims=(0, 55),
        subset_idxs=subset_idxs,
        sub_labels=labels,
        fluct_params=fluct_params,
    )
    return plt
end

function precursor_plot(
    ts, normed_c_pcs, p::PrecursorModelFamily;
    subset_idxs=nothing, xlab=false
)
    plt = precursor_plot(
        ts, normed_c_pcs;
        subset_idxs=subset_idxs, xlab=xlab
    )

    opt_cpc = optimal_precursor_concentration(p)
    hline!(
        [opt_cpc / p.Km], linestyle=:dash, label="optimal",
        linewidth=thin_line, linecolor=opt_color,
    )
    return plt
end

function massfraction_plot_multiABX(
    soln, p::PrecursorMultiABXwAnticipation;
    subset_idxs=nothing, xlab=false, plot_aR=false,
    labels=nothing
)
    if xlab
        xlab = "Time (hrs)"
    else
        xlab = ""
    end
    
    u = parse_soln(soln, p)
    n_abx = (length(u) - 5) ÷ 4
    Rs = u[1]
    aRs = u[2]
    ts = u[5]
    ARs = u[6:5+n_abx]
    a_ARs = u[6+n_abx:5+2*n_abx]
    C_bnd_vecs = u[6+2*n_abx:5+3*n_abx]

    C_bnds_mtx = hcat(C_bnd_vecs...)
    total_C_bnds = zeros(length(ts))
    for i in 1:length(ts)
        total_C_bnds[i] = calculate_total_C_bnd(Rs[i], C_bnds_mtx[i,:], p.abx.n_prot)
    end
    R_frees = Rs .- total_C_bnds ./ p.abx.n_prot

    ARs_mtx = hcat(ARs...)
    total_ARs = zeros(length(ts))
    for i in 1:length(ts)
        total_ARs[i] = sum(ARs_mtx[i,:])
    end
    φ_M = 1 .- Rs .- total_ARs .- p.abx.φ_0

    plt = plot_series(
        ts, Rs, φR_color;
        title="Mass Fractions",
        xlab=xlab,
        ylab="Mass Fraction",
        subset_idxs=subset_idxs,
        label="φ_R",
        ylims=(0, 0.5), #1 - p.abx.φ_0),
        lw=thin_line,
        sub_labels=labels
    )
    if plot_aR
        plot_series!(
            plt, ts, aRs, αR_color;
            subset_idxs=subset_idxs, label="α_R",
            lw=extra_thin_line, ls=:dash,
            sub_labels=labels
        )
    end
    plot_series!(
        plt, ts, R_frees, φR_free_color;
        subset_idxs=subset_idxs, label="φ_Rf",
        lw=thin_line,
        sub_labels=labels
    )
    plot_series!(
        plt, ts, φ_M, φM_color;
        subset_idxs=subset_idxs, label="φ_M",
        lw=thin_line,
        sub_labels=labels
    )
    # generate colors for ARs
    φAR_color_range = gen_color_range(n_abx, φAR_color)
    αAR_color_range = gen_color_range(n_abx, αAR_color)
    for i in 1:n_abx
        plot_series!(
            plt, ts, ARs[i], φAR_color_range[i];
            subset_idxs=subset_idxs, label="φ_AR",
            lw=thin_line,
            sub_labels=labels
        )
        plot_series!(
            plt, ts, a_ARs[i], αAR_color_range[i];
            subset_idxs=subset_idxs, label="α_AR",
            lw=extra_thin_line, ls=:dashdot,
            sub_labels=labels
        )
    end

    return plt
end

function massfraction_plot(
    soln, jc::JumpConfig, p::PrecursorABXwAnticipation;
    subset_idxs=nothing, xlab=false, plot_aR=false,
    labels=nothing, fluct_params=nothing
)
    if xlab
        xlab = "Time (hrs)"
    else
        xlab = ""
    end
    if soln isa NTuple && length(soln) == 9
        dim = 1
    else
        dim = length(size(soln))
    end

    if dim == 3
        # multi solution set
        ts = gen_ts(jc)
        Rs = Matrix(soln[:, 1, :]')
        aRs = Matrix(soln[:, 2, :]')
        ARs = Matrix(soln[:, 3, :]')
        a_ARs = Matrix(soln[:, 4, :]')
        C_bnds = Matrix(soln[:, 6, :]')
    elseif dim == 1
        Rs, aRs, ARs, a_ARs, _, C_bnds, _, _, ts = parse_soln(soln, jc, p)
    elseif dim == 2
        # solns from DE solver
        @assert size(soln, 1) == 6 "soln is expected to be 6xN from DE solver"
        Rs, aRs, ARs, a_ARs, cpcs, C_bnds, _, _, ts = parse_soln(soln, p)
    else
        error("soln must be 1D or 3D")
    end
    R_frees = Rs .- C_bnds ./ p.abx.n_prot
    φ_M = 1 .- Rs .- ARs .- p.abx.φ_0

    plt = plot_series(
        ts, Rs, φR_color;
        title="Mass Fractions",
        xlab=xlab,
        ylab="Mass Fraction",
        subset_idxs=subset_idxs,
        label="φ_R",
        ylims=(0, 0.5), #1 - p.abx.φ_0),
        lw=thin_line,
        sub_labels=labels,
        fluct_params=fluct_params,
    )
    if plot_aR
        plot_series!(
            plt, ts, aRs, αR_color;
            subset_idxs=subset_idxs, label="α_R",
            lw=extra_thin_line, ls=:dashdot,
            sub_labels=labels
        )
    end
    plot_series!(
        plt, ts, R_frees, φR_free_color;
        subset_idxs=subset_idxs, label="φ_Rf",
        lw=thin_line,
        sub_labels=labels
    )
    plot_series!(
        plt, ts, φ_M, φM_color;
        subset_idxs=subset_idxs, label="φ_M",
        lw=thin_line,
        sub_labels=labels
    )
    plot_series!(
        plt, ts, ARs, φAR_color;
        subset_idxs=subset_idxs, label="φ_AR",
        lw=thin_line,
        sub_labels=labels
    )
    plot_series!(
        plt, ts, a_ARs, αAR_color;
        subset_idxs=subset_idxs, label="α_AR",
        lw=extra_thin_line, ls=:dashdot,
        sub_labels=labels
    )

    return plt
end

function massfraction_plot(soln, jc::JumpConfig, p::PrecursorABXBaseModel; subset_idxs=nothing, xlab=false)
    if xlab
        xlab = "Time (hrs)"
    else
        xlab = ""
    end
    Rs, aRs, cpcs, C_bnds, C_ins, Ls, ts = parse_soln(soln, jc, p)
    R_frees = Rs - C_bnds ./ p.n_prot
    φ_M = 1 .- Rs .- p.φ_0

    plt = plot_series(
        ts, Rs, φR_color;
        title="Mass Fractions",
        xlab=xlab,
        ylab="Mass Fraction",
        subset_idxs=subset_idxs,
        label="φ_R",
        lw=thin_line,
    )
    plot_series!(
        plt, ts, aRs, αR_color;
        subset_idxs=subset_idxs, label="α_R",
        lw=extra_thin_line, ls=:dashdot,
    )
    plot_series!(
        plt, ts, R_frees, φR_free_color;
        subset_idxs=subset_idxs, label="φ_Rf",
        lw=thin_line
    )
    plot_series!(
        plt, ts, φ_M, φM_color;
        subset_idxs=subset_idxs, label="φ_M",
        lw=thin_line
    )

    return plt
end

function φR_plot(
    ts, φ_Rs, α_Rs; subset_idxs=nothing,
    xlab=false, plot_φM=false, φ0=nothing
)
    if xlab
        xlab = "Time (hrs)"
    else
        xlab = ""
    end
    plt = plot_series(
        ts, φ_Rs, φR_color;
        title="Protein Fractions",
        xlab=xlab,
        ylab="Mass Fraction",
        subset_idxs=subset_idxs,
        label="φ_R",
    )
    plot_series!(
        plt, ts, α_Rs, αR_color;
        subset_idxs=subset_idxs, lw=extra_thin_line, ls=:dashdot,
        label="α_R",
    )
    if plot_φM
        if !isnothing(φ0)
            φ_Ms = 1 .- φ_Rs .- φ0
        else
            error("φ0 must be provided to plot φM")
        end
        plot_series!(
            plt, ts, φ_Ms, φM_color;
            subset_idxs=subset_idxs, lw=thin_line,
            label="φ_M"
        )
    end
    return plt
end

function φR_plot(
    ts, φ_Rs, α_Rs, p::PrecursorModelFamily;
    subset_idxs=nothing, xlab=false
)
    plt = φR_plot(
        ts, φ_Rs, α_Rs; subset_idxs=subset_idxs, xlab=xlab
    )

    opt_R = optimal_φ_R(p)
    hline!(
        [opt_R], linestyle=:dash, label="optimal",
        linewidth=thin_line, linecolor=opt_color,
    )

    return plt
end

function λ_plot(
    ts, λs; subset_idxs=nothing, xlab=false, growth_color=growth_color,
    labels=nothing, distinguish_colors=false, plot=nothing,
    integrate=false, fluct_params=nothing
)
    if xlab
        xlab = "Time (hrs)"
    else
        xlab = ""
    end
    if integrate
        title = "Biomass Abundance"
        ylab = "Simulated Optical Density"
        # cumsum trapz; check the actual syntax
        λs = trapz(λs)
    else
        title = "Growth Rate"
        ylab = "Rate (1/hr)"
    end

    if isnothing(plot)
        plt = plot_series(
            ts, λs, growth_color;
            title=title,
            xlab=xlab,
            ylab=ylab,
            ylims=(0, 1.1),
            subset_idxs=subset_idxs,
            sub_labels=labels,
            distinguish_colors=distinguish_colors,
            fluct_params=fluct_params,
        )
    else
        plt = plot_series!(
            plot, ts, λs, growth_color;
            subset_idxs=subset_idxs,
            sub_labels=labels,
            distinguish_colors=distinguish_colors
        )
    end
    return plt
end

function λ_plot(
    ts, λs, p::PrecursorModelFamily; subset_idxs=nothing, xlab=false
)
    plt = λ_plot(ts, λs; subset_idxs=subset_idxs, xlab=xlab)

    opt_gr = analytic_optimal_growth(p)
    hline!(
        [opt_gr], linestyle=:dash, label="optimal",
        linewidth=thin_line, linecolor=opt_color,
    )

    return plt
end

function ABX_plot(
    ts, C_ins, C_bnds; subset_idxs=nothing, xlab=false, labels=nothing,
    fluct_params=nothing
)
    if xlab
        xlab = "Time (hrs)"
    else
        xlab = ""
    end
    plt = plot_series(
        ts, C_bnds, C_bnd_color;
        title="ABX",
        xlab=xlab,
        ylab="Concentration ([M])",
        ylims=(0, 2e-5),
        subset_idxs=subset_idxs,
        label="Bound to Ribosome",
        sub_labels=labels,
        fluct_params=fluct_params,
    )
    plot_series!(
        plt, ts, C_ins, C_in_color;
        subset_idxs=subset_idxs,
        label="Intracellular",
        sub_labels=labels
    )

    return plt
end

function multiABX_plot(
    ts, C_ins, C_bnds; subset_idxs=nothing, xlab=false, labels=nothing,
)
    if xlab
        xlab = "Time (hrs)"
    else
        xlab = ""
    end
    n_abx = length(C_bnds)
    C_bnd_color_range = gen_color_range(n_abx, C_bnd_color)
    C_in_color_range = gen_color_range(n_abx, C_in_color)
    plt = plot(title="ABX", xlab=xlab, ylab="Concentration ([M])")
    for i in 1:n_abx
        plot_series!(
            plt, ts, C_bnds[i], C_bnd_color_range[i];
            ylims=(0, 2e-5),
            subset_idxs=subset_idxs,
            label="Bound to Ribosome",
            sub_labels=labels
        )
        plot_series!(
            plt, ts, C_ins[i], C_in_color_range[i];
            subset_idxs=subset_idxs,
            label="Intracellular",
            sub_labels=labels
        )
    end
    return plt
end

function set_of_colors(n_plots, base, distinguishable_colors=false)
    if n_plots == 1
        return [base]
    else
        if distinguishable_colors
            return distinguishable_colors(n_plots)
        else
            return gen_color_range(n_plots, base)
        end
    end
end

function subset_series(series, subset_idxs)
    if series isa Matrix
        if subset_idxs != nothing
            series = series[:, subset_idxs]
        end
        n_plots = size(series)[2]
    elseif series isa Vector
        n_plots = 1
    else
        error("series must be a matrix or vector, not a $(typeof(series))")
    end
    return series, n_plots
end

function gen_color_range(n, end_color::RGB, start_color::RGB=colorant"cornsilk3")
    """ Generates a discrete color range with n colors"""
    if n == 1
        return [end_color]
    end
    # rescale 
    xs = (0:n-1) / (5 / 4 * (n - 1)) .+ 0.2
    cs = ColorScheme(range(start_color, end_color))
    return get(cs, xs)
end

function gen_color_range(end_color::RGB, start_color::RGB=colorant"cornsilk3")
    """ Generates a continuous color range"""
    cs = ColorScheme(range(start_color, end_color))
    return cs
end

function color_displacement(color::RGB, displacement)
    """ 
    Displace RGB color by angular displacement in HSV space
    and return new RGB color
    """
    base = convert(HSV, color)
    new_hue = base.h + displacement
    new_hue = new_hue % 360
    new_color = convert(typeof(color), HSV(new_hue, base.s, base.v))
    return new_color
end