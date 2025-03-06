"""
    GuiSDE

Guided smoothing and control for diffusion processes.
"""
module GuiSDE

using LinearAlgebra
using Random, Statistics
using ProgressMeter

function B end
function β end
function b end

function A end
function σ end
function a end

function H end
function ς end
function R end

"""
    U(c, ν, P)

Density of `N(ν, P)`` scaled by `exp(c)`.
"""
U(c, ν, P) = x -> exp(c)*(2pi)^(-length(x)/2)*(det(P))^(-1/2)*exp(-(ν - x)'*(P\(ν - x))/2)

"""
    U(c, ν, P)

Log-density of `N(ν, P)`` scaled by `exp(c)`.
"""
logU(c, ν, P) = x -> c  - (length(x)/2)*log(2pi) - logdet(P)/2  -(ν - x)'*(P\(ν - x))/2


"""
    forward(ts, x0, y0=zero(x0)) = (;Xs, Ys)

Simulate system of SDEs dXₜ = b(t, Xₜ)dt + σ(t, Xₜ)dWₜ, dYₜ = HXₜdt + ςdBₜ with the Euler scheme.
`ts` is the time grid, `x0` and `y0` are starting values.
"""
function forward(ts, x0, y0=zero(x0))
    d = length(x0)
    d2 = length(y0)
    x = x0
    Xs = [x]
    Ws = [zero(x)]
    Ys = [y0]
    Bs = [zero(y0)]
    @showprogress for i in eachindex(ts)[1:end-1]
        dt = ts[i+1] - ts[i]
        dw = sqrt(dt)*randn(d)
        x = x + b(ts[i], x)*dt + σ(ts[i], x)*dw
        # observe Xs[2:end] in diff(Ys)
        db = sqrt(dt)*randn(d2)
        push!(Ws, Ws[end] + dw)
        push!(Xs, x)
        push!(Bs, Bs[end] + db)        
        push!(Ys, Ys[end] + H(ts[i])*x*dt + ς(ts[i])*db)
    end
    (;Xs, Ys, Ws, Bs)
end


"""
    backward(ts, Ys, νT, PT, cT=0.0) = (;Ys, cs, νs, Ps)

Compute the linear Kalman-Bucy backward filter on the grid `ts` for the system 
dXₜ = (B(t)Xₜ + β(t)) dt + √A(t))dWₜ, dYₜ = HXₜdt + ςdBₜ 
with observation `Ys` with terminal value `U(cT, νT, PT)`.
"""
function backward(ts, Ys, νT, PT, cT=0.0)
    cs = zeros(eachindex(ts))
    d = length(νT)
    νs = Vector{typeof(νT)}(undef, length(ts))
    Ps = Vector{typeof(PT)}(undef, length(ts))
    νs[end] = νT
    Ps[end] = PT
    cs[end] = cT

    i = reverse(eachindex(ts))[2]
    @showprogress for i in reverse(eachindex(ts))[2:end]
        dt = ts[i+1] - ts[i]
        @assert dt > 0
        K = Ps[i+1]*(H(ts[i+1])'/R(ts[i+1]))
        Ps[i] = Ps[i+1] + (-B(ts[i+1])*Ps[i+1] - Ps[i+1]*B(ts[i+1])' + A(ts[i+1]) - K*R(ts[i+1])*K')*dt # ok
        dy = Ys[i+1] - Ys[i] # careful
        νs[i] = νs[i+1] -(β(ts[i+1]) + B(ts[i+1])*νs[i+1])*dt  + K*(dy - H(ts[i+1])*νs[i+1]*dt) # dy surprisingly large? vs[i] > vs[i+1]
        Δc = (-tr(B(ts[i+1])))*dt + ((R(ts[i+1])\H(ts[i+1])*νs[i+1])'*dy - 0.5norm(ς(ts[i+1])\H(ts[i+1])*νs[i+1])^2*dt)  
        cs[i] = cs[i+1] + Δc
    end
    (;Ys, cs, νs, Ps)
end


"""
    kallianpur_str_is(ts, x0, Ys, logU, N, ϕ0=0.0) = (; μˣs, lls)

Use importance sampling with the Kallianpur-Striebel formula and `N` samples 
to compute μˣₜ = E[Xₜ | Y, V] 
where `logU` corresponds to the boundary condition in the backward Kalman-Bucy filter.
Log-importance weights `lls` can be offset by `ϕ0` to avoid underflow.
"""
function kallianpur_str_is(ts, x0, Ys, logU, N, ϕ0=0.0)
    xxT = zero(x0)
    lls = Float64[]
    d = length(x0)
    Xs = [zero(x0) for t in ts]
    
    μˣs = [zero(x0) for t in ts]

    @showprogress for _ in 1:N  
        φ = ϕ0
        Xs[1] = x = x0
        for i in eachindex(ts)[1:end-1]
            dt = ts[i+1] - ts[i]
            dw = sqrt(dt)*randn(d)
            xnew = x + b(ts[i], x)*dt + σ(ts[i], x)*dw
            dy = Ys[i+1] - Ys[i]
            dϕ = (R(ts[i+1])\H(ts[i+1])*x)'*dy - 0.5*norm(ς(ts[i+1])\H(ts[i+1])*x)^2*dt
            φ += dϕ
            Xs[i+1] = x = xnew    
        end
        μˣs .+= Xs*(exp(φ + logU(x)))/N 
        push!(lls, φ + logU(x))
    end
    μˣs =  μˣs/mean(exp.(lls))
    (; μˣs, lls)
end


"""
    guided_is(ts, x0, guide, N) = (; Xᵒs, μˣs, μᵒs, lls)

Use guided importance sampling with `N` samples to compute μˣₜ = E[Xₜ | Y, V].
`guide` is the output of the backward filter. 
Also returns a random proposal `Xᵒs`and the proposal mean `μᵒs` and log-weights `lls`.
"""
function guided_is(ts, x0, guide, N)
    (;Ys, cs, νs) = guide
    Ps = cholesky.(Symmetric.(guide.Ps))
    d = length(x0)
    Xᵒs = [zero(x0) for t in ts]
    μˣs = zero.(Xᵒs)
    μᵒs = zero.(Xᵒs)
    u0 = logU(guide.cs[1], guide.νs[1], guide.Ps[1])(x0)

    ll = 0.0
    lls = Float64[]
    ll2 = 0.0
    l2s = 0.0
    @showprogress for _ in 1:N
        Xᵒs[1] .= x = x0 # todo
        ll = 0.0
        ll2 = 0.0
        for i in eachindex(ts)[1:end-1]
            dt = ts[i+1] - ts[i]
            dy = Ys[i+1] - Ys[i] # observe X[i+1] in Y[i+1] - Y[i], knowledge in ν[i]
            dw = sqrt(dt)*randn(d)
            r = Ps[i]\(νs[i] - x)
            u = σ(ts[i], x)'*r

            Δll = (b(ts[i], x) - (B(ts[i])*x + β(ts[i])))'*r*dt 
            if a(ts[i], x) ≠ A(ts[i])
                Δll += -0.5*tr((a(ts[i], x) - A(ts[i]))/Ps[i])*dt + 0.5*r'*(a(ts[i], x) - A(ts[i]))*r*dt
            end
            
            Δll2 = -u'*dw - 0.5u'*u*dt + (R(ts[i])\H(ts[i])*x)'*(dy) - 0.5*norm(ς(ts[i])\H(ts[i])*x)^2*dt 
            x = x + b(ts[i], x)*dt + σ(ts[i], x)*(u*dt + dw)
            Xᵒs[i+1] .= x

            ll = ll + Δll
            ll2 = ll2 + Δll2

        end
        μˣs .+= Xᵒs*(exp(ll + u0))/N
        μᵒs .+= Xᵒs/N    
        push!(lls, ll + u0)
        l2s += exp(ll2)*U(cs[end], νs[end], Ps[end])(x)/N
    end
    μˣs = μˣs/mean(exp.(lls))
    @show mean(exp.(lls)), l2s;
    (; Xᵒs, μˣs, μᵒs, lls)
end


"""
    guided_cr(ts, x0, guide, N, ϱ=0.0) = (; Xˣs, μˣs, lls, acc)

Use Metropolis-Hastings sampling with `N` samples to compute μˣₜ = E[Xₜ | Y, V] using
independent proposals (`ϱ=0.0`) or a preconditioned Crank-Nicolson scheme with autocorrelation 
`ϱ > 0.0`. `guide` is the output of the backward filter. 
Also returns a random bridge `Xˣs`, the log-weights `lls` and the acceptance probability `acc`.
"""
function guided_cr(ts, x0, guide, N, ϱ=0.0)
    (;Ys, cs, νs) = guide
    Ps = cholesky.(Symmetric.(guide.Ps))
    d = length(x0)
    Xᵒs = [zero(x0) for t in ts]
    Xˣs = zero.(Xᵒs)
    μˣs = zero.(Xᵒs)
    u0 = logU(guide.cs[1], guide.νs[1], guide.Ps[1])(x0)
    js = sqrt(1-ϱ^2)*sqrt.(diff(ts))
    dWs = sqrt.(diff(ts)) .* [randn(d) for s in js]
    dWᵒs = zero.(dWs)
    llold = -Inf
    lls = Float64[]
    acc = 0
    @showprogress for _ in 1:N
        Xᵒs[1] .= x = x0 # todo
        ll = 0.0
        for i in eachindex(ts)[1:end-1]
            dt = ts[i+1] - ts[i]
            dy = Ys[i+1] - Ys[i]
            dw = ϱ*dWs[i] + js[i]*randn(d)
            dWᵒs[i] .= dw
            r = Ps[i]\(νs[i] - x)
            u = σ(ts[i], x)'*r

            Δll =  (b(ts[i], x) - (B(ts[i])*x + β(ts[i])))'*r*dt 
            if a(ts[i], x) ≠ A(ts[i])
                Δll += -0.5*tr((a(ts[i], x) - A(ts[i]))/Ps[i])*dt + 0.5*r'*(a(ts[i], x) - A(ts[i]))*r*dt
            end
            x = x + b(ts[i], x)*dt + σ(ts[i], x)*(u*dt + dw)
            Xᵒs[i+1] .= x

            ll = ll + Δll
        end
        if rand() < exp(ll-llold)
            acc += 1
            Xˣs, Xᵒs = Xᵒs, Xˣs
            dWs, dWᵒs = dWᵒs, dWs
            llold = ll
        end
        μˣs .+= Xˣs/N 
        push!(lls, ll + u0)   
    end
    acc /= N
    @show mean(exp.(lls)), acc
    (; Xˣs, μˣs, lls, acc)
end

end
