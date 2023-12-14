# xe_losses assume inputs as probabilities (real valued and 
# normalised, ie. after a Softmax).
xe_loss(y::Array{Float64,2}, Y::Array{Float64,2}) = -sum(Y.*log.(y))
xe_loss_derivative(y::Array{Float64,2}, Y::Array{Float64,2}) = y - Y