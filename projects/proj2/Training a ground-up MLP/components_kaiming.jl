function kaiming(type, dim_out,dim_in)
    # Weight init for a layer with ReLU activation function.
    matrix = randn(type, dim_out, dim_in).*sqrt(2/dim_in)
    return matrix
end