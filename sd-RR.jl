using Pkg
Pkg.activate(".")

using Graphs, SimpleWeightedGraphs
using Random
using Arpack
using DataFrames
using CSV
using Statistics
using Formatting

# ======== Functions =======================================================================================================================================

NUM_DIGITS = 8

struct Superdiffusion_Results
       model1::String
       model2::String
       N::Int64
       num_duplex::Int64
       sorted::Bool
       reversed::Bool
       ks1::Vector{Float64}
       ks2::Vector{Float64}
       df::DataFrame
end

function duplex_superdiffusion_info(G1, G2)
       L1 = laplacian_matrix(G1)
       L2 = laplacian_matrix(G2)
       La = (L1 + L2) / 2
       λ1 = eigs(L1, nev=3, which=:SM)[1][2]
       λ2 = eigs(L2, nev=3, which=:SM)[1][2]
       λa = eigs(La, nev=3, which=:SM)[1][2]
       ζ = (λa - max(λ1, λ2)) / max(λ1, λ2)
       has_superdiffusion = (ζ > 0)
       ζ = round(ζ, digits=NUM_DIGITS)
       λ1 = round(λ1, digits=NUM_DIGITS)
       λ2 = round(λ2, digits=NUM_DIGITS)
       λa = round(λa, digits=NUM_DIGITS)
       return ζ, has_superdiffusion, λa, [λ1, λ2]
end

function my_erdos_renyi(N::Int64, k::Float64, sorted::Bool, reversed::Bool)
       p = k / (N - 1)
       G = erdos_renyi(N, p)
       while !is_connected(G)
              G = erdos_renyi(N, p)
       end
       if sorted
              v_sorted = sortperm(degree(G), rev=!reversed)
              G, vmap = induced_subgraph(G, v_sorted)
       end
       return G
end

function my_random_regular(N::Int64, k::Float64)
       k = round(Int, k)
       G = random_regular_graph(N, k)
       return G
end

function my_random_semiregular_graph(N::Int64, k::Float64, sorted::Bool, reversed::Bool)
       k_up = ceil(Int, k)
       degree_list = [k_up for i in 1:N]
       m = k - mean(degree_list)
       while abs(m) >= 0.005
              i = rand(1:N)
              if degree_list[i] == k_up
                     degree_list[i] -= 1
              end
              m = k - mean(degree_list)
       end
       if sum(degree_list) % 2 != 0
              degree_list[rand(1:N)] += 1
       end
       G = random_configuration_model(N, degree_list)

       if sorted
              v_sorted = sortperm(degree(G), rev=!reversed)
              G, vmap = induced_subgraph(G, v_sorted)
       end

       return G
end

function superdiffusion_analysis(model1::String, model2::String, N::Int64, ks1::Vector{Float64}, ks2::Vector{Float64}, num_duplex::Int64, sorted::Bool, reversed::Bool)

       if model1 ∉ ["ER", "RR", "RSR"]
              throw(DomainError(model1, "Incorrect Model 1"))
       end
       if model2 ∉ ["ER", "RR", "RSR"]
              throw(DomainError(model2, "Incorrect Model 2"))
       end

       rep_abort = 100
       unit_vector = ones(N)

       num_ks1 = length(ks1)
       num_ks2 = length(ks2)

       df = DataFrame(
              k1=Float64[], k2=Float64[], ka=Float64[],
              k1_avg=Float64[], k2_avg=Float64[], ka_avg=Float64[],
              ζ_avg=Float64[], prob_avg=Float64[],
              λ1_avg=Float64[], λ2_avg=Float64[], λa_avg=Float64[],
              σ_λ1=Float64[], σ_λ2=Float64[], σ_λa=Float64[])

       for i in 1:num_ks1
              k1 = ks1[i]

              for j in 1:num_ks2
                     k2 = ks2[j]

                     println("\u1b[1F")
                     println(string(" k1 = ", k1, " k2 = ", k2, "\u1b[1F"))

                     num_duplex_shortened = num_duplex

                     k1_avg, k2_avg, ka_avg = 0.0, 0.0, 0.0
                     ζ_avg, prob_avg = 0.0, 0.0
                     λ1_avg, λ2_avg, λa_avg = 0.0, 0.0, 0.0
                     list_λ1, list_λ2, list_λa = Float64[], Float64[], Float64[]

                     for rep in 1:num_duplex

                            succeeded = false
                            while !succeeded

                                   ### LAYER 1 ###
                                   if model1 == "ER"
                                          G1 = my_erdos_renyi(N, k1, sorted, false)
                                   elseif model1 == "RR"
                                          G1 = random_regular_graph(N, k1)
                                   elseif model1 == "BA"
                                          G1 = my_barabasi_albert(N, k1, sorted, false)
                                   elseif model1 == "RSR"
                                          G1 = my_random_semiregular_graph(N, l1, sorted, false)
                                   end

                                   ### LAYER 2 ###
                                   if model2 == "ER"
                                          G2 = my_erdos_renyi(N, k2, sorted, reversed)
                                   elseif model2 == "RR"
                                          G2 = random_regular_graph(N, k2)
                                   elseif model2 == "BA"
                                          G2 = my_barabasi_albert(N, k2, sorted, reversed)
                                   elseif model2 == "RSR"
                                          G2 = my_random_semiregular_graph(N, k2, sorted, reversed)
                                   end

                                   try
                                          A1 = adjacency_matrix(G1)
                                          A2 = adjacency_matrix(G2)
                                          Aa = (A1 + A2) / 2
                                          degs1 = A1 * unit_vector
                                          degs2 = A2 * unit_vector
                                          degsa = Aa * unit_vector

                                          k1_avg += mean(degs1)
                                          k2_avg += mean(degs2)
                                          ka_avg += mean(degsa)

                                          ζ, has_superdiffusion, λa, λs = duplex_superdiffusion_info(G1, G2)

                                          ζ_avg += ζ
                                          if has_superdiffusion
                                                 prob_avg += 1
                                          end

                                          λ1_avg += λs[1]
                                          λ2_avg += λs[2]
                                          λa_avg += λa

                                          push!(list_λ1, λs[1])
                                          push!(list_λ2, λs[2])
                                          push!(list_λa, λa)

                                          succeeded = true
                                   catch e
                                          succeeded = false
                                   end
                            end

                            if rep == rep_abort && prob_avg == 0.0
                                   num_duplex_shortened = rep_abort
                                   break
                            end
                     end

                     ka = (k1 + k2) / 2

                     k1_avg /= num_duplex_shortened
                     k2_avg /= num_duplex_shortened
                     ka_avg /= num_duplex_shortened

                     ζ_avg /= num_duplex_shortened
                     prob_avg /= num_duplex_shortened

                     λ1_avg /= num_duplex_shortened
                     λ2_avg /= num_duplex_shortened
                     λa_avg /= num_duplex_shortened

                     σ_λ1 = std(list_λ1)
                     σ_λ2 = std(list_λ2)
                     σ_λa = std(list_λa)

                     ζ_avg = round(ζ_avg, digits=NUM_DIGITS)
                     prob_avg = round(prob_avg, digits=NUM_DIGITS)

                     λ1_avg = round(λ1_avg, digits=NUM_DIGITS)
                     λ2_avg = round(λ2_avg, digits=NUM_DIGITS)
                     λa_avg = round(λa_avg, digits=NUM_DIGITS)

                     σ_λ1 = round(σ_λ1, digits=NUM_DIGITS)
                     σ_λ2 = round(σ_λ2, digits=NUM_DIGITS)
                     σ_λa = round(σ_λa, digits=NUM_DIGITS)

                     push!(df, (k1, k2, ka, k1_avg, k2_avg, ka_avg, ζ_avg, prob_avg, λ1_avg, λ2_avg, λa_avg, σ_λ1, σ_λ2, σ_λa))
              end
       end

       return Superdiffusion_Results(model1, model2, N, num_duplex, sorted, reversed, ks1, ks2, df)
end

function save_results(fn::String, results::Superdiffusion_Results)
       CSV.write(fn, results.df)
end

# ======== Initializations =================================================================================================================================

cd(Base.dirname(Base.source_path()))
pwd()

Random.seed!(555)

N = 500
num_duplex = 20

ks = collect(5.0:2.0:400.0)

# ======== Main 1 ==========================================================================================================================================

# -------- Parameters --------

model1 = "RR"
model2 = "RR"

sorted = false
reversed = false

fn_results_er = string(model1, "-", model2, "-N", N, "-D", num_duplex, "-S", sorted ? "1" : "0", "-R", reversed ? "1" : "0", ".csv")

println(fn_results_er)

# -------- Experiment --------

@time fn_results_er (
       results_er = superdiffusion_analysis(model1, model2, N, ks, ks, num_duplex, sorted, reversed)
)
save_results(fn_results_er, results_er)
