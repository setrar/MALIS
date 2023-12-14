function forward(x::Array{Float64,2}, net::Dict)::Array{Float64,2}
    # Feed the input through a network.
    # Keep track of the activations both pre- and post- activation function.
    # Store these values in the network's dictionary.
    A=[x]
    Z=[]
    for n in 1:length(net["Layers"])
        z = net["Layers"][n].weight*x + net["Layers"][n].bias
        x = forward(z, net["Layers"][n].act_fn)
        append!(Z, [z])
        append!(A, [x])
    end
    net["A"]=A
    net["Z"]=Z
    return x
end;