using LinearAlgebra
using Random, Statistics
using Test
using ProgressMeter
using GLMakie

using ForwardDiff
using ForwardDiff: value
using Optimisers
using Optimisers: fmap


using GuiSDE
using GuiSDE: forward, backward, logU, kallianpur_str_is, guided_is, guided_cr
import GuiSDE: b, β, B, A, σ, a, H, ς, R, backward

REMOTE = false

function backward(::Type{T}, B, β, ts, Ys, νT, PT, cT=0.0) where {T}
    cs = Vector{T}(eachindex(ts))
    d = length(νT)
    νs = Vector{Vector{T}}(undef, length(ts))
    Ps = Vector{Matrix{T}}(undef, length(ts))
    νs[end] = νT
    Ps[end] = PT
    cs[end] = cT

    i = reverse(eachindex(ts))[2]
    for i in reverse(eachindex(ts))[2:end]
        dt = ts[i+1] - ts[i]
        @assert dt > 0
        K = Ps[i+1]*(H(ts[i+1])'/R(ts[i+1]))
        Ps[i] = Ps[i+1] + (-B*Ps[i+1] - Ps[i+1]*B' + A(ts[i+1]) - K*R(ts[i+1])*K')*dt # ok
        dy = Ys[i+1] - Ys[i] # careful
        νs[i] = νs[i+1] -(β + B*νs[i+1])*dt  + K*(dy - H(ts[i+1])*νs[i+1]*dt) # dy surprisingly large? vs[i] > vs[i+1]
        Δc = (-tr(B))*dt + ((R(ts[i+1])\H(ts[i+1])*νs[i+1])'*dy - 0.5norm(ς(ts[i+1])\H(ts[i+1])*νs[i+1])^2*dt)  
        cs[i] = cs[i+1] + Δc
    end
    (;Ys, cs, νs, Ps)
end


function make_obj(ts, x0, Ys, νT, PT, N = 50)
    d = length(νT)
    function (θ::Vector{T}) where {T}
        local B = SymTridiagonal(θ[1:d], θ[d+1:2d-1])
        local β = θ[2d:end]
        (;νs, Ps) = guide = backward(T, B, β, ts, Ys, νT, PT)
        ll0 = logU(guide.cs[1], guide.νs[1], guide.Ps[1])(x0)
        o = 0.0
        for _ in 1:N
            ll = zero(ll0)
            x = x0 
            for i in eachindex(ts)[1:end-1]
                dt = ts[i+1] - ts[i]
                dy = Ys[i+1] - Ys[i] # observe X[i+1] in Y[i+1] - Y[i], knowledge in ν[i]
                dw = sqrt(dt)*randn(d)
                r = Ps[i]\(νs[i] - x)
                u = σ(ts[i], x)'*r
                Δll = (b(ts[i], x) - (B*x + β))'*r*dt 
                if a(ts[i], x) ≠ A(ts[i])
                    Δll += -0.5*tr((a(ts[i], x) - A(ts[i]))/Ps[i])*dt + 0.5*r'*(a(ts[i], x) - A(ts[i]))*r*dt
                end    
                x = value.(x + b(ts[i], x)*dt + σ(ts[i], x)*(u*dt + dw))
                ll = ll + Δll
            end
        o += (ll0 + ll)*exp(value(ll0 + ll))/N
        end
        return o
    end
end

function make_obj_rev(ts, x0, Ys, νT, PT, N = 100)
    d = length(νT)
    function (θ::Vector{T}) where {T}
        local B = SymTridiagonal(θ[1:d], θ[d+1:2d-1])
        local β = θ[2d:end]
        (;νs, Ps) = guide = backward(T, B, β, ts, Ys, νT, PT)

        ll0 = logU(guide.cs[1], guide.νs[1], guide.Ps[1])(x0)
        o = zero(ll0)
        for _ in 1:N
            ϕ = ll = zero(ll0)
            x = x0 
            for i in eachindex(ts)[1:end-1]
                dt = ts[i+1] - ts[i]
                dy = Ys[i+1] - Ys[i] # observe X[i+1] in Y[i+1] - Y[i], knowledge in ν[i]
                dw = sqrt(dt)*randn(d)
                dϕ = (R(ts[i])\H(ts[i])*x)'*dy - 0.5*norm(ς(ts[i])\H(ts[i])*x)^2*dt
               
                r = Ps[i]\(νs[i] - x)
                u = σ(ts[i], x)'*r
                dx = b(ts[i], x)*dt + σ(ts[i], x)*(u*dt + dw)
                dll = u'*dw + 0.5u'*u*dt # dw is the new/P^u brownian motion
              
                x += dx
                ϕ += dϕ
                ll += dll
            end
            o += (-(ϕ + logU(0.0, νT, PT)(x) - ll)/N)
        end
        return o
    end
end

function guided_mean(ts, x0, Ys, νT, PT, θ)
    local B = SymTridiagonal(θ[1:d], θ[d+1:2d-1])
    local β = θ[2d:end]
    (;νs, Ps) = guide = backward(Float64, B, β, ts, Ys, νT, PT)
    x = x0 
    μᵒs = [x0]
    for i in eachindex(ts)[1:end-1]
        dt = ts[i+1] - ts[i]
        r = Ps[i]\(νs[i] - x)
        u = σ(ts[i], x)'*r
        x = x + b(ts[i], x)*dt + σ(ts[i], x)*(u*dt)
        push!(μᵒs, x)
    end
    (;ts, μᵒs)
end


using Functors
struct StochasticModel{S <: AbstractArray, T}
    obj::T
    θ::S
end
@functor StochasticModel (θ,)
checknan(θ) = (any(isnan.(θ)) && error(); θ)
function stochastic_gradient(m::StochasticModel)
    fmap(p -> checknan(ForwardDiff.gradient(m.obj, m.θ)), m)
end


function optimize!(obj, θ, hist, nodeθ; η = 0.01, N = 20000, M = 10)
    model = StochasticModel(obj, θ)
    state = Optimisers.setup(Optimisers.Adam(η), model)  # just once

    pm = Progress(N÷M)
    for iter in 1:N 
        Optimisers.update!(state, model, stochastic_gradient(model))
        if iter %  M == 0
            ob = obj(θ)
            push!(hist, ob)
            next!(pm; showvalues = [(:iter,iter), (:θ, extrema(θ)), (:obj, ob)])
            REMOTE || notify(nodeθ)
        end
    end
    θ
end


Random.seed!(1177) # help?
ts = range(0.0, 1.0, step=0.001)

const d = 20

const Λ_ = SymTridiagonal([1.0; 2ones(d-2); 1.0], -ones(d-1))
const β_ = zeros(d)
const σ_= 1.0I(d)
const a_ = σ_*σ_'


B(t) = -10*0.5*Λ_ # not time dependent in paper
β(t) = β_
b(t, x) = 2*(x - x.^3) + B(t)*x + β(t)

A(t) = 1.0a_
σ(t, x) = σ_
a(t, x) = a_

H(t) = 1.0I(d)
ς(t) = 0.2I(d)
R(t) = ς(t)*ς(t)'


# terminal condition
PT = Matrix(0.1I(d))
x0 = zeros(d)
(;Xs, Ys, Bs) = forward(ts, x0)
xT = Xs[end]
νT = xT + cholesky(PT).L*randn(length(xT))

fig = Figure()
ax = Axis(fig[1,1])
hm = heatmap!(ts, 1:d, stack(Xs, dims=1), colorrange=(-2,2))
Colorbar(fig[1,2], hm)
save("forwardvar.png", fig)
fig1 = fig
REMOTE || display(fig1)



θ = θtrue = [diag(B(0)); diag(B(0),1); β_] 
obj = make_obj_rev(ts, x0, Ys, νT, PT, 1)
@show obj(θtrue )
θ0 = zeros(3d-1)

θ = copy(θ0)
history = Float64[]

fig_live = Figure()
ax = Axis(fig_live[1,1])
nodeθ = Observable(θ)
hm = map(θ -> [SymTridiagonal(θ[1:d], θ[d+1:2d-1]); θ[2d:end]'; B(0.0)], nodeθ)
heatmap!(ax, hm, colorrange=(-15,15))
REMOTE || display(fig_live)

M = 10
optimize!(obj, θ, history, nodeθ; N = 5000, η = 0.01, M)
optimize!(obj, θ, history, nodeθ; N = 50, η = 0.01, M)


guide = backward(ts, Ys, νT, PT)
(; Xˣs, μˣs, lls, acc) = gui_cr = guided_cr(ts, x0, guide, 10000, 0.3)

begin
    REL = true
    (;μᵒs) = guided_mean(ts, x0, Ys, νT, PT, θ)
    comp = guided_mean(ts, x0, Ys, νT, PT, θtrue)
    local B_ = SymTridiagonal(θ[1:d], θ[d+1:2d-1])
    local β = θ[2d:end]
    (;νs, Ps) = guide = backward(Float64, B_, β, ts, Ys, νT, PT)


    fig = Figure()
    ax = Axis(fig[1,1])
    hm = heatmap!(ts, 1:d, stack(Xs, dims=1))
    Colorbar(fig[1,2], hm)
    ax = Axis(fig[1,3])
    if REL
        hm = heatmap!(ts, 1:d, abs.(stack(μᵒs, dims=1)-stack(μˣs, dims=1)), colorrange=(0.0,0.1))
    else
        hm = heatmap!(ts, 1:d, stack(μᵒs, dims=1), colorrange=(-2,2))
    end
    Colorbar(fig[1,4], hm)
    ax = Axis(fig[1,5])
    hm = heatmap!(ts, 1:d, stack(μˣs, dims=1), colorrange=(-2,2))
    Colorbar(fig[1,6], hm)

    fig2 = fig
    ax = Axis(fig[2,1])
    heatmap!(ax, [SymTridiagonal(θ[1:d], θ[d+1:2d-1]); B(0.0)], colorrange=(-15,15))
    ax = Axis(fig[2,3])
    lines!(ax, history)
    ax = Axis(fig[2,5])
    if REL
        hm = heatmap!(ts, 1:d, abs.(stack(comp.μᵒs, dims=1)-stack(μˣs, dims=1)), colorrange=(0.0,0.1))
    else
        hm = heatmap!(ts, 1:d, stack(comp.μᵒs, dims=1), colorrange=(-2,2))
    end
    Colorbar(fig[2,6], hm)

    display(fig2)
end

begin 
    (;μᵒs) = guided_mean(ts, x0, Ys, νT, PT, θ)
    comp = guided_mean(ts, x0, Ys, νT, PT, θtrue)
    local B_ = SymTridiagonal(θ[1:d], θ[d+1:2d-1])
    local β = θ[2d:end]
    (;νs, Ps) = guide = backward(Float64, B_, β,  ts, Ys, νT, PT)

    fig = Figure()
    ax = Axis(fig[1,1])
    hm = heatmap!(ts, 1:d, stack(Xs, dims=1), colorrange=(-2,2))
    Colorbar(fig[1,2], hm)
    save("forwardvar.png", fig)

    fig = Figure()
    ax = Axis(fig[1,1])
    heatmap!(ax, [SymTridiagonal(θ[1:d], θ[d+1:2d-1]);], colorrange=(-15,15))
    ax = Axis(fig[1,end+1])
    hm = heatmap!(ax, [B(0.0);], colorrange=(-15,15))
    Colorbar(fig[1,end+1], hm)
    save("BversusBvar.png", fig)

    fig = Figure()
    ax = Axis(fig[1,1], limits=((1-0.5,d+0.5), (-2, 2)))
    scatter!(ax, 1:d, θ[2d:end], markersize=20)
    save("betavar.png", fig)
      
    fig = Figure()
    ax = Axis(fig[1,1])
    lines!(ax, range(1, step=10, length=length(history)), history)
    save("convergencevar.png", fig)

    fig = Figure()
    ax = Axis(fig[1,1])
    hm = heatmap!(ts, 1:d, stack(μˣs, dims=1), colorrange=(-2,2))
    Colorbar(fig[1,end+1], hm)
    save("posteriormeanvar.png", fig)

    fig = Figure()
    ax = Axis(fig[1,1])
    hm = heatmap!(ts, 1:d, stack(comp.μᵒs, dims=1), colorrange=(-2,2))
    Colorbar(fig[1,end+1], hm)
    save("proposalmeanvar.png", fig)

    fig = Figure()
    ax = Axis(fig[1,1])
    hm = heatmap!(ts, 1:d, stack(μᵒs, dims=1), colorrange=(-2,2))
    Colorbar(fig[1,end+1], hm)
    save("variationalvar.png", fig)

    fig = Figure()
    ax = Axis(fig[1,1])
    hm = heatmap!(ts, 1:d, abs.(stack(comp.μᵒs, dims=1)-stack(μˣs, dims=1)), colorrange=(0.0,0.1))
    Colorbar(fig[1,end+1], hm)
    save("proposalerrvar.png", fig)

    fig = Figure()
    ax = Axis(fig[1,1])
    hm = heatmap!(ts, 1:d, abs.(stack(μᵒs, dims=1)-stack(μˣs, dims=1)), colorrange=(0.0,0.1))
    Colorbar(fig[1,end+1], hm)
    save("variationalerr.png", fig)
end 