using LinearAlgebra
using Random, Statistics
using Test
using ProgressMeter
using GLMakie

using GuiSDE
using GuiSDE: forward, backward, logU, kallianpur_str_is, guided_is, guided_cr
import GuiSDE: b, β, B, A, σ, a, H, ς, R
center(xs) = xs .- mean(xs)
nth(x) = getindex(x, d÷2)

const d = 100
N = 5000
Random.seed!(1177) # help?
ts = range(0.0, 1.0, step=0.001)

const Λ_ = SymTridiagonal([1.0; 2ones(d-2); 1.0], -ones(d-1))
const β_ = zeros(d)
const σ_= 1.0I(d)
const a_ = σ_*σ_'

#= Define the system

    dXₜ = b(t, Xₜ)dt + σ(t, Xₜ)dWₜ
    dX̃ₜ = B(t)X̃ₜdt + β(t)dt + √A(t))dWₜ
    dYₜ = HXₜdt + ςdBₜ 
=#

B(t) = -10*0.5*Λ_
β(t) = β_
b(t, x) = 2*(x - x.^3) + B(t)*x + β(t)

A(t) = a_
σ(t, x) = σ_
a(t, x) = a_

H(t) = 1.0I(d)
ς(t) = 0.2I(d)
R(t) = ς(t)*ς(t)'


# terminal condition
PT = Matrix(0.1I(d))
x0 = zeros(d)


(;Xs, Ys) = forward(ts, x0)
fig = Figure()
ax = Axis(fig[1,1])
hm = heatmap!(ts, 1:d, stack(Xs, dims=1))
Colorbar(fig[1,2], hm)
display(fig)
save("forward.png", fig)

fig = Figure()
ax = Axis(fig[1,1])
hm = heatmap!(ts, 1:d, stack(Ys, dims=1))
Colorbar(fig[1,2], hm)
display(fig)
save("observation.png", fig)

fig = Figure()
ax = Axis(fig[1,1])
lines!(ax, ts, nth.(Xs), linewidth=1.0, color=nth.(Xs), colorrange=(-2,2))
lines!(ax, ts, nth.(Ys), linewidth=2.5, color=nth.(Ys), colorrange=(-2,2))
save("truthandobs.png", fig)
display(fig)

xT = Xs[end]
νT = xT + cholesky(PT).L*randn(length(xT))


guide = backward(ts, Ys, νT, PT)
(;cs, νs, Ps) = guide
u0 = logU(0.0, νT, PT)(x0)

fig = Figure()
ax = Axis(fig[1,1])
hm = heatmap!(ts, 1:d, stack(νs, dims=1), colorrange=(-2,2))
Colorbar(fig[1,2], hm)
display(fig)
save("guide.png", fig)

ka_str_is_pre = kallianpur_str_is(ts, x0, Ys, logU(0.0, νT, PT), N÷10, 0.0)
ka_str_is = kallianpur_str_is(ts, x0, Ys, logU(0.0, νT, PT), N, -mean(ka_str_is_pre.lls))

fig = Figure()
ax = Axis(fig[1,1])
hm = heatmap!(ts, 1:d, stack(ka_str_is.μˣs, dims=1), colorrange=(-2.0,2.0))
Colorbar(fig[1,2], hm)
display(fig)
save("unguided.png", fig)


(; Xᵒs, μˣs, μᵒs, lls) = gui_is = guided_is(ts, x0, guide, N)
fig = Figure()
ax = Axis(fig[1,1])
hm = heatmap!(ts, 1:d, stack(Xᵒs, dims=1), colorrange=(-2,2))
Colorbar(fig[1,2], hm)
display(fig)
save("proposal.png", fig)


(; Xˣs, μˣs, lls, acc) = gui_cr = guided_cr(ts, x0, guide, N, 0.9)

fig = Figure()
ax = Axis(fig[1,1])
hm = heatmap!(ts, 1:d, stack(Xˣs, dims=1), colorrange=(-2,2))
Colorbar(fig[1,2], hm)
display(fig)
save("samplecr.png", fig)

fig = Figure()
ax = Axis(fig[1,1])
hm = heatmap!(ts, 1:d, stack(μˣs, dims=1), colorrange=(-2,2))
Colorbar(fig[1,2], hm)
display(fig)
save("meancr.png", fig)

fig = Figure()
ax = Axis(fig[1,end+1]); lines!(ax, center(sort(ka_str_is.lls)), range(0, 1, length=length(ka_str_is.lls)))
ax = Axis(fig[1,end+1]); lines!(ax, center(sort(gui_is.lls)), range(0, 1, length=length(gui_is.lls)))
ax = Axis(fig[1,end+1]); lines!(ax, center(sort(gui_cr.lls)), range(0, 1, length=length(gui_cr.lls)))
display(fig)
save("lls.png", fig)


begin 
    fig = Figure()
    ax = Axis(fig[1,1])
    heatmap!(stack(Xs, dims=1), colorrange=(-2,2))
    ax = Axis(fig[2,1])
    lines!(ax, ts, nth.(Xs), linewidth=1.0, color=nth.(Xs), colorrange=(-2,2))

    ax = Axis(fig[1,2])
    heatmap!(stack(gui_cr.μˣs, dims=1), colorrange=(-2,2))
    ax = Axis(fig[2,2])
    lines!(ax, ts, nth.(gui_cr.μˣs), linewidth=1.0, color=nth.(gui_cr.μˣs), colorrange=(-2,2))


    ax = Axis(fig[1,3])
    heatmap!(stack(gui_cr.Xˣs, dims=1), colorrange=(-2,2))
    ax = Axis(fig[2,3])
    lines!(ax, ts, nth.(gui_cr.Xˣs), linewidth=1.0, color=nth.(gui_cr.Xˣs), colorrange=(-2,2))
    
    ax = Axis(fig[1,end+1])
    heatmap!(stack(gui_is.μˣs, dims=1), colorrange=(-2,2))
    ax = Axis(fig[2,end])
    lines!(ax, ts, nth.(gui_is.μˣs), linestyle=:dot, linewidth=1.0, color=nth.(gui_is.μˣs), colorrange=(-2,2))
    
    ax = Axis(fig[1,end+1])
    heatmap!(stack(ka_str_is.μˣs, dims=1), colorrange=(-2,2))
    ax = Axis(fig[2,end])
    lines!(ax, ts, nth.(ka_str_is.μˣs), linestyle=:dot, linewidth=1.0, color=nth.(ka_str_is.μˣs), colorrange=(-2,2))
 

    ax = Axis(fig[1,end+1])
    heatmap!(stack(νs, dims=1), colorrange=(-2,2))

    ax = Axis(fig[2,end])
    N = length(lls)
    center(xs) = xs .- mean(xs)
    #lines!(ax, center(sort(ka_str_is.lls)), range(0, 1, length=length(ka_str_is.lls))) # does not fit
    lines!(ax, center(sort(gui_is.lls)), range(0, 1, length=length(gui_is.lls)))
    lines!(ax, center(sort(gui_cr.lls)), range(0, 1, length=length(gui_cr.lls)))
  
    display(fig)
end
