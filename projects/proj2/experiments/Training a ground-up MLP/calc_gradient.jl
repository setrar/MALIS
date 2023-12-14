function calculate_gradient(dA, W, B, Z, A_prev, act_fn)
    dZ = dA.*gradient(Z, act_fn)
    dW = (dZ * A_prev')
    dB = dZ
    dA_prev = W'*dZ
    out=[dA_prev, dW, dB]
    return out
end
