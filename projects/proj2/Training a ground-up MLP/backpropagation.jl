# n is the layer number
dA_prev = xe_loss_derivative(y, Y)./mb_size

for n in 1:depth
    n_curr = depth-(n-1);
    n_prev = depth-n;

    dA = dA_prev;
    Z = net["Z"][n_curr];
    A_prev = net["A"][n_curr];

    W = net["Layers"][n_curr].weight;
    B = net["Layers"][n_curr].bias;
    act_fn = net["Layers"][n_curr].act_fn;

    out = calculate_gradient(dA, W, B, Z, A_prev, act_fn)     

    dA_prev = out[1]
    append!(dW, [out[2]])
    append!(dB, [out[3]])   
end

# put them back in forward order
dW=reverse(dW)
dB=reverse(dB)
