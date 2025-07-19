using Random, Distributions

Cm_ug_per_mL_to_M(x) = x * 3.095e-6
Cm_M_to_ug_per_mL(x) = x / 3.095e-6

heaviside(A, c; T=Float64) = x -> heaviside_base(x, c, A; T=T)
heaviside_base(x, c, A; T=Float64) = (x - c) < 0 ? T(0.0) : T(A)

function square_pulse(pulse_length; amplitude=1, x0=0, T=Float64)
    h1 = heaviside(amplitude, x0, T=T)
    h2 = heaviside(amplitude, pulse_length + x0, T=T)
    h(x) = h1.(x) .- h2.(x)
    return h
end

function square_wave(period, duty, duration; x0=0, amplitude=1, numeric_type=Float64)
    n_waves = duration ÷ period
    pulses = [
        square_pulse(
            period * duty, amplitude=amplitude,
            x0=x0 + i * period, T=numeric_type
        ) for i in 0:n_waves
    ]
    h(x) = sum(f -> f(x), pulses)
    return h
end

function is_approximately_equal(a, b; tol=1e-4)
    return abs(a - b) < tol
end

function random_wave(
    avg_period, wave_length, n_waves, duration; x0=0, amplitude=1,
    distribution="uniform", stdev=1
)
    if distribution == "uniform"
        d = Uniform(0, 2 * avg_period)
    elseif distribution == "normal"
        d = Normal(avg_period, stdev)
    else
        error("'distribution' must be 'normal' or 'uniform'")
    end
    pulse_pts = rand(d, n_waves)
    pulse_timepts = cumsum(pulse_pts)


    pulses = [
        square_pulse(
            wave_length, amplitude=amplitude,
            x0=x0 + t
        ) for t in pulse_timepts
    ]
    h(x) = sum(f -> f(x), pulses)
    return h
end

function cache_function_output(filename, f, args...; overwrite=false, kwargs...)
    local_dir = pwd()
    cache = joinpath(local_dir, "cache")
    if filename in readdir(cache) && overwrite != true
        println("Loading data from saved .jld2")
        stored_data = load(joinpath(cache, filename))
        data = stored_data["d"]
    else
        println("Simulating...")
        data = f(args...; kwargs...)
        x = Dict("d" => data)
        println("Saving data to $cache/$filename")
        save(joinpath(cache, filename), x)
    end
    return data
end
