struct ReLU
end;
forward(z::Array{Float64, 2}, act_fn::ReLU)::Array{Float64,2} = z.*(z.>0)
gradient(z::Array{Float64, 2}, act_fn::ReLU)::Array{Float64,2} = Array{Float64, 2}(z.>0)

struct Softmax
end;
forward(z::Array{Float64, 2}, act_fn::Softmax)::Array{Float64,2} = softmax(z)
gradient(z::Array{Float64, 2}, act_fn::Softmax)::Array{Float64,2} = (softmax(z)) .* (1 .- softmax(z))

function softmax(z::Array{Float64,2})::Array{Float64,2}
    #converts real numbers to probabilities
    c=maximum(z)
    p = z .- log.( sum( exp.(z .- c) ) ) .-c
    p = exp.(z)
    return p
end